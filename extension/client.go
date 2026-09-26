package extension

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
)

// DefaultTimeout bounds an invoke when the caller's context has no deadline,
// and bounds how long the default client waits for the daemon to answer any
// request, a stream included.
const DefaultTimeout = 180 * time.Second

// DefaultControlTimeout bounds info, models, refresh and every oauth step when
// the caller's context has no deadline. It keeps a refresh well inside
// tokenstore.Coordinator's default lease wait, which must exceed it.
const DefaultControlTimeout = 30 * time.Second

const (
	// maxJSONBody bounds every JSON answer but a catalog.
	maxJSONBody = 1 << 20
	// maxCatalogBody bounds a models answer.
	maxCatalogBody = 16 << 20
	// maxResponseBody bounds an invoke answer, which may be an image or
	// audio.
	maxResponseBody = 64 << 20
	// maxMessageRunes bounds a message taken from a daemon's answer.
	maxMessageRunes = 512
	// maxProviderIDLength bounds a provider id.
	maxProviderIDLength = 128
)

// Config configures a Client. Products supply every value; the package reads
// no environment.
type Config struct {
	// BaseURL is the daemon's origin, such as "http://127.0.0.1:18888". A
	// path, for a daemon behind a reverse proxy, is kept; PathPrefix is
	// appended to it.
	BaseURL string
	// Secret is the daemon's shared secret, sent as a bearer token. A daemon
	// started without one accepts any caller.
	Secret string
	// HTTPClient performs every request. Nil uses a client that does not
	// follow redirects and waits DefaultTimeout for response headers. A
	// client with a Timeout also cuts off streams that run longer.
	HTTPClient *http.Client
}

// Client talks to one extension daemon. It keeps no state between requests
// and is safe for concurrent use.
type Client struct {
	base       string
	secret     string
	httpClient *http.Client
}

// NewClient returns a Client for the daemon config names.
func NewClient(config Config) (*Client, error) {
	base := strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	parsed, err := url.Parse(base)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, errors.New("extension: base URL must be an absolute http or https URL without credentials, query or fragment")
	}
	httpClient := config.HTTPClient
	if httpClient == nil {
		transport := &http.Transport{}
		if shared, ok := http.DefaultTransport.(*http.Transport); ok {
			transport = shared.Clone()
		}
		transport.ResponseHeaderTimeout = DefaultTimeout
		httpClient = &http.Client{
			Transport: transport,
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}
	}
	return &Client{base: base, secret: strings.TrimSpace(config.Secret), httpClient: httpClient}, nil
}

// Info returns the providers the daemon serves.
func (c *Client) Info(ctx context.Context) (InfoResponse, error) {
	ctx, cancel := withDeadline(ctx, DefaultControlTimeout)
	defer cancel()
	request, err := c.newRequest(ctx, http.MethodGet, c.base+PathPrefix+"info", nil)
	if err != nil {
		return InfoResponse{}, err
	}
	request.Header.Set("Accept", core.ContentTypeJSON)
	response, err := c.httpClient.Do(request)
	if err != nil {
		return InfoResponse{}, transportError(ctx, "", "info", err)
	}
	defer response.Body.Close()
	if !succeeded(response) {
		return InfoResponse{}, statusError(response, "", "info")
	}
	var info InfoResponse
	if err := decodeBounded(response.Body, maxJSONBody, &info); err != nil {
		return InfoResponse{}, invalidAnswer("", "info", err)
	}
	return info, nil
}

// ListModels returns provider's catalog for credential, which may be nil.
func (c *Client) ListModels(ctx context.Context, provider string, credential *core.Credential) ([]core.ModelInfo, error) {
	endpoint, err := c.providerURL(provider, "models")
	if err != nil {
		return nil, err
	}
	ctx, cancel := withDeadline(ctx, DefaultControlTimeout)
	defer cancel()
	request, err := c.newRequest(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Accept", core.ContentTypeJSON)
	SetCredentialHeaders(request.Header, credential)
	response, err := c.httpClient.Do(request)
	if err != nil {
		return nil, transportError(ctx, provider, "models", err)
	}
	defer response.Body.Close()
	if !succeeded(response) {
		return nil, statusError(response, provider, "models")
	}
	var catalog ModelsResponse
	if err := decodeBounded(response.Body, maxCatalogBody, &catalog); err != nil {
		return nil, invalidAnswer(provider, "models", err)
	}
	return catalog.Models, nil
}

// Invoke performs one non-streaming operation on provider. The response's
// ContentType is the daemon's, and its Losses are the ones the daemon
// reported.
func (c *Client) Invoke(ctx context.Context, provider string, request core.Request) (core.Response, error) {
	endpoint, err := c.providerURL(provider, "invoke")
	if err != nil {
		return core.Response{}, err
	}
	ctx, cancel := withDeadline(ctx, DefaultTimeout)
	defer cancel()
	httpRequest, err := c.newRequest(ctx, http.MethodPost, endpoint, bytes.NewReader(request.Body))
	if err != nil {
		return core.Response{}, err
	}
	setOperationHeaders(httpRequest.Header, request)
	response, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return core.Response{}, transportError(ctx, provider, "invoke", err)
	}
	defer response.Body.Close()
	if !succeeded(response) {
		return core.Response{}, statusError(response, provider, "invoke")
	}
	body, err := readBounded(response.Body, maxResponseBody)
	if err != nil {
		if errors.Is(err, errBodyTooLarge) {
			return core.Response{}, invalidAnswer(provider, "invoke", err)
		}
		return core.Response{}, transportError(ctx, provider, "invoke", err)
	}
	return core.Response{
		Body:        body,
		ContentType: response.Header.Get("Content-Type"),
		Losses:      LossesFromHeader(response.Header),
	}, nil
}

// Stream performs one streaming operation on provider. Each frame is one
// complete SSE record that carries data. No deadline applies but the
// caller's context. The stream reports the losses the daemon put on its
// answer, as core.StreamLosses reads them.
func (c *Client) Stream(ctx context.Context, provider string, request core.Request) (core.StreamIter, error) {
	endpoint, err := c.providerURL(provider, "stream")
	if err != nil {
		return nil, err
	}
	httpRequest, err := c.newRequest(ctx, http.MethodPost, endpoint, bytes.NewReader(request.Body))
	if err != nil {
		return nil, err
	}
	setOperationHeaders(httpRequest.Header, request)
	httpRequest.Header.Set("Accept", core.ContentTypeEventStream)
	response, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return nil, transportError(ctx, provider, "stream", err)
	}
	if !succeeded(response) {
		defer response.Body.Close()
		return nil, statusError(response, provider, "stream")
	}
	return newStream(ctx, provider, response), nil
}

// Refresh asks the daemon to renew record for provider and returns the
// renewed record. The product calls it under its own lease; RefreshFunc
// adapts it to tokenstore.Coordinator. A daemon's refusal is a *RefreshError.
func (c *Client) Refresh(ctx context.Context, provider string, record tokenstore.Record) (tokenstore.Record, error) {
	status, body, err := c.postJSON(ctx, provider, "refresh", RefreshRequest{Record: record})
	if err != nil {
		return tokenstore.Record{}, err
	}
	var answer RefreshResponse
	decodeErr := json.Unmarshal(body, &answer)
	if status < 200 || status > 299 || answer.Error != "" || decodeErr != nil {
		message := answer.Error
		if message == "" {
			message = errorMessage(body)
		}
		return tokenstore.Record{}, &RefreshError{
			Provider:   provider,
			StatusCode: status,
			Message:    safeMessage(message),
			Permanent:  answer.Terminal,
		}
	}
	return answer.Record, nil
}

// RefreshFunc returns the refresh of provider, for tokenstore.NewCoordinator.
func (c *Client) RefreshFunc(provider string) tokenstore.RefreshFunc {
	return func(ctx context.Context, current tokenstore.Record) (tokenstore.Record, error) {
		return c.Refresh(ctx, provider, current)
	}
}

// RefreshError reports a refresh the daemon refused.
type RefreshError struct {
	Provider string
	// StatusCode is the daemon's answer status.
	StatusCode int
	// Message is the daemon's reason, bounded and on one line.
	Message string
	// Permanent reports that the provider rejected the grant for good, so
	// the credential should be revoked.
	Permanent bool
}

func (e *RefreshError) Error() string {
	text := "extension " + e.Provider + " refresh failed"
	if e.Message != "" {
		return text + ": " + e.Message
	}
	if e.StatusCode != 0 {
		return fmt.Sprintf("%s (HTTP %d)", text, e.StatusCode)
	}
	return text
}

// Terminal reports a grant the provider rejected for good, as
// tokenstore.IsTerminal asks.
func (e *RefreshError) Terminal() bool { return e != nil && e.Permanent }

// newRequest builds a request to the daemon with the secret.
func (c *Client) newRequest(ctx context.Context, method, endpoint string, body io.Reader) (*http.Request, error) {
	request, err := http.NewRequestWithContext(ctx, method, endpoint, body)
	if err != nil {
		return nil, core.NewConfigurationError("extension request is invalid", err)
	}
	if c.secret != "" {
		request.Header.Set("Authorization", "Bearer "+c.secret)
	}
	return request, nil
}

// postJSON posts a JSON body to one of provider's routes and returns the
// answer's status and body, whatever the status. Only a request that got no
// answer fails.
func (c *Client) postJSON(ctx context.Context, provider, route string, in any) (int, []byte, error) {
	endpoint, err := c.providerURL(provider, route)
	if err != nil {
		return 0, nil, err
	}
	raw, err := json.Marshal(in)
	if err != nil {
		return 0, nil, fmt.Errorf("extension: encode %s request: %w", route, err)
	}
	ctx, cancel := withDeadline(ctx, DefaultControlTimeout)
	defer cancel()
	request, err := c.newRequest(ctx, http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return 0, nil, err
	}
	request.Header.Set("Content-Type", core.ContentTypeJSON)
	request.Header.Set("Accept", core.ContentTypeJSON)
	response, err := c.httpClient.Do(request)
	if err != nil {
		return 0, nil, transportError(ctx, provider, route, err)
	}
	defer response.Body.Close()
	body, err := readBounded(response.Body, maxJSONBody)
	if err != nil {
		if errors.Is(err, errBodyTooLarge) {
			return 0, nil, invalidAnswer(provider, route, err)
		}
		return 0, nil, transportError(ctx, provider, route, err)
	}
	return response.StatusCode, body, nil
}

// providerURL returns the URL of one of provider's routes.
func (c *Client) providerURL(provider, route string) (string, error) {
	if !validProviderID(provider) {
		return "", core.NewConfigurationError(fmt.Sprintf("extension provider id %q is invalid", provider), nil)
	}
	return c.base + PathPrefix + provider + "/" + route, nil
}

// validProviderID admits ids that are one safe path segment.
func validProviderID(id string) bool {
	if id == "" || id == "." || id == ".." || id == "info" || len(id) > maxProviderIDLength {
		return false
	}
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-', r == '.':
		default:
			return false
		}
	}
	return true
}

// setOperationHeaders puts an operation's surface, model, encoding and
// credential on its headers.
func setOperationHeaders(header http.Header, request core.Request) {
	header.Set(HeaderSurface, string(request.Surface))
	header.Set(HeaderModel, request.Model)
	if request.ContentType != "" {
		header.Set("Content-Type", request.ContentType)
	}
	SetCredentialHeaders(header, request.Credential)
}

// withDeadline bounds ctx by timeout unless it already has a deadline.
func withDeadline(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

func succeeded(response *http.Response) bool {
	return response.StatusCode >= 200 && response.StatusCode <= 299
}

var errBodyTooLarge = errors.New("extension: answer exceeds its size limit")

// readBounded reads body up to limit bytes.
func readBounded(body io.Reader, limit int64) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > limit {
		return nil, errBodyTooLarge
	}
	return raw, nil
}

// decodeBounded decodes a JSON body of at most limit bytes into out.
func decodeBounded(body io.Reader, limit int64, out any) error {
	raw, err := readBounded(body, limit)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, out)
}

// statusError classifies a failed answer the way core classifies an upstream
// failure. Its message keeps the daemon's reason but never its raw body.
func statusError(response *http.Response, provider, route string) error {
	raw, _ := readBounded(response.Body, maxJSONBody)
	failure := core.ProviderFailure{
		StatusCode: response.StatusCode,
		RetryAfter: response.Header.Get("Retry-After"),
		ObservedAt: time.Now(),
	}
	health := core.ClassifyProviderFailure(failure)
	message := label(provider, route) + " failed"
	if reason := safeMessage(errorMessage(raw)); reason != "" {
		message += ": " + reason
	} else {
		message += fmt.Sprintf(" (HTTP %d)", response.StatusCode)
	}
	return &core.ProviderError{
		Message: message,
		Class:   health.ErrorClass,
		Classification: core.ProviderErrorClassification{
			StatusCode:       response.StatusCode,
			Retryable:        health.Retryable,
			FailoverEligible: health.Retryable,
			CircuitFailure:   health.ErrorClass == core.ProviderErrorTransport || health.ErrorClass == core.ProviderErrorUpstream,
			RetryAfter:       health.RetryAfter,
		},
		Cause: &core.ProviderOperationError{Failure: failure, Op: label(provider, route)},
	}
}

// transportError reports a request that got no answer. Another target may
// get past it, unless the caller gave up.
func transportError(ctx context.Context, provider, route string, err error) error {
	failure := &core.ProviderError{
		Message: label(provider, route) + " failed: the extension did not answer",
		Class:   core.ProviderErrorTransport,
		Classification: core.ProviderErrorClassification{
			Retryable: true, FailoverEligible: true, CircuitFailure: true,
		},
		Cause: err,
	}
	if ctx.Err() != nil {
		failure.Message = label(provider, route) + " was canceled"
		failure.Classification = core.ProviderErrorClassification{}
	}
	return failure
}

// invalidAnswer reports an answer the client cannot use.
func invalidAnswer(provider, route string, err error) error {
	return &core.ProviderError{
		Message: label(provider, route) + " failed: the extension's answer is invalid",
		Class:   core.ProviderErrorUpstream,
		Cause:   err,
	}
}

func label(provider, route string) string {
	if provider == "" {
		return "extension " + route
	}
	return "extension " + provider + " " + route
}

// errorMessage reads the reason from an error body in either of its shapes,
// {"error": {"message": ...}} or {"error": "..."}.
func errorMessage(raw []byte) string {
	var envelope struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(raw, &envelope) != nil || len(envelope.Error) == 0 {
		return ""
	}
	var text string
	if json.Unmarshal(envelope.Error, &text) == nil {
		return text
	}
	var detail ErrorDetail
	if json.Unmarshal(envelope.Error, &detail) == nil {
		return detail.Message
	}
	return ""
}

// safeMessage puts a daemon's reason on one line and bounds its length.
func safeMessage(text string) string {
	text = strings.Join(strings.Fields(text), " ")
	if utf8.RuneCountInString(text) > maxMessageRunes {
		text = string([]rune(text)[:maxMessageRunes]) + "..."
	}
	return text
}
