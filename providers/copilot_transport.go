package providers

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	copilotauth "github.com/xibodev/llm-provider-auth/copilot"
	core "github.com/xibodev/llmgw-core"
)

const (
	copilotMaxResponseBytes = 64 << 20
	copilotMaxCatalogBytes  = 8 << 20
	copilotMaxErrorBytes    = 64 << 10
)

// copilotIdentity is the editor identity every Copilot request presents.
type copilotIdentity struct {
	integrationID, editorVersion, pluginVersion, userAgent string
}

// copilotCall is one Copilot API request. accept replaces the Accept header
// the gateway sends by default, and vision adds the header without which
// Copilot rejects images. timeout, when set, bounds the attempt.
type copilotCall struct {
	method, path string
	body         []byte
	accept       string
	vision       bool
	timeout      time.Duration
}

// send performs call with the credential's session. Copilot rejecting the
// session with 401 is retried once with a new session, as the gateway does;
// any other answer is the caller's to read. It returns the session of the
// last attempt.
func (p *Copilot) send(ctx context.Context, credential *core.Credential, call copilotCall) (*http.Response, *copilotauth.Session, error) {
	session, err := p.sessions.session(ctx, credential, "")
	if err != nil {
		return nil, nil, err
	}
	response, err := p.do(ctx, session, call)
	if err != nil || response.StatusCode != http.StatusUnauthorized {
		return response, session, err
	}
	response.Body.Close()
	if session, err = p.sessions.session(ctx, credential, session.Token); err != nil {
		return nil, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, &core.ProviderError{Message: "the Copilot request was canceled", Cause: err}
	}
	response, err = p.do(ctx, session, call)
	return response, session, err
}

// do sends one attempt with the gateway's Copilot headers, at the session's
// API base.
func (p *Copilot) do(ctx context.Context, session *copilotauth.Session, call copilotCall) (*http.Response, error) {
	attempt, cancel := ctx, context.CancelFunc(func() {})
	if call.timeout > 0 {
		attempt, cancel = context.WithTimeout(ctx, call.timeout)
	}
	var body io.Reader
	if call.body != nil {
		body = bytes.NewReader(call.body)
	}
	request, err := http.NewRequestWithContext(attempt, call.method, strings.TrimRight(session.ChatBaseURL, "/")+call.path, body)
	if err != nil {
		cancel()
		return nil, core.NewConfigurationError("the Copilot request could not be created", err)
	}
	request.Header.Set("Authorization", "Bearer "+session.Token)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Copilot-Integration-Id", p.identity.integrationID)
	request.Header.Set("Editor-Version", p.identity.editorVersion)
	request.Header.Set("Editor-Plugin-Version", p.identity.pluginVersion)
	request.Header.Set("OpenAI-Intent", "conversation-panel")
	request.Header.Set("User-Agent", p.identity.userAgent)
	if call.accept != "" {
		request.Header.Set("Accept", call.accept)
	}
	if call.vision {
		request.Header.Set("Copilot-Vision-Request", "true")
	}
	response, err := p.client.Do(request)
	if err != nil {
		cancel()
		return nil, copilotTransportFailure(ctx, "Copilot transport failed", err)
	}
	response.Body = copilotBody{ReadCloser: response.Body, cancel: cancel}
	return response, nil
}

// copilotBody ends an attempt's deadline when its body is closed.
type copilotBody struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (b copilotBody) Close() error {
	defer b.cancel()
	return b.ReadCloser.Close()
}

// copilotRead reads a successful body, bounded as the gateway bounds it. A
// body that breaks off is repeated like a transport failure.
func copilotRead(ctx context.Context, response *http.Response, limit int64) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if int64(len(raw)) > limit {
		return nil, copilotMalformed("the Copilot response exceeds its size limit", nil)
	}
	if err != nil {
		return nil, copilotTransportFailure(ctx, "the Copilot response could not be read", err)
	}
	return raw, nil
}

// copilotStatusFailure reports an upstream error status, classified as the
// gateway classifies its Copilot failures: a transient status permits a
// retry and failover and counts against the provider; any other status,
// 401 and 403 included, ends the request. The message names the status and
// the error's identifiers in raw, never the upstream's text.
func copilotStatusFailure(response *http.Response, raw []byte) *core.ProviderError {
	message := fmt.Sprintf("Copilot upstream request failed with status %d", response.StatusCode)
	if diagnostic := codexErrorDiagnostic(raw); diagnostic != "" {
		message += " (" + diagnostic + ")"
	}
	var retryAfter time.Duration
	if response.StatusCode != http.StatusUnauthorized && response.StatusCode != http.StatusForbidden {
		retryAfter = parseRetryAfterHeader(response.Header.Get("Retry-After"), time.Now())
	}
	return copilotStatusError(response.StatusCode, message, retryAfter, nil)
}

// copilotErrorBody reads enough of an error body to explain it, or nothing.
func copilotErrorBody(response *http.Response) []byte {
	raw, err := readLimited(response.Body, copilotMaxErrorBytes)
	if err != nil {
		return nil
	}
	return raw
}

func copilotStatusError(status int, message string, retryAfter time.Duration, cause error) *core.ProviderError {
	transient := false
	switch status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusInternalServerError,
		http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		transient = true
	}
	return &core.ProviderError{
		Message: message, Class: core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: status}).ErrorClass,
		Classification: core.ProviderErrorClassification{
			StatusCode: status, Retryable: transient, FailoverEligible: transient, CircuitFailure: transient, RetryAfter: retryAfter,
		},
		Cause: cause,
	}
}

// copilotTransportFailure reports a request that got no answer. A caller
// that gave up permits nothing and counts against nothing; a client timeout
// while the caller still waits is a slow upstream, which another attempt may
// get past.
func copilotTransportFailure(ctx context.Context, message string, err error) *core.ProviderError {
	failure := &core.ProviderError{
		Message: message, Class: core.ProviderErrorTransport,
		Classification: core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}, Cause: err,
	}
	if callerCancellation(err) && ctx.Err() != nil {
		failure.Classification = core.ProviderErrorClassification{}
	}
	return failure
}

// copilotMalformed reports an answer the gateway could not use. Another
// target may still serve the request, and it counts against the provider.
func copilotMalformed(message string, err error) *core.ProviderError {
	return &core.ProviderError{
		Message: message, Class: core.ProviderErrorUpstream,
		Classification: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}, Cause: err,
	}
}

// copilotUnsupported reports a request, or an answer, that the Responses
// conversion cannot carry without a material loss. Another target may serve
// it natively.
func copilotUnsupported(message string, err error) *core.ProviderError {
	return &core.ProviderError{
		Message: message, Class: core.ProviderErrorUnsupported,
		Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
	}
}

// copilotInvalid reports a request no target could serve.
func copilotInvalid(message string, err error) *core.ProviderError {
	return &core.ProviderError{Message: message, Class: core.ProviderErrorInvalidRequest, Cause: err}
}
