package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// CodexConfig configures Codex. Instructions and ClientVersion are required.
type CodexConfig struct {
	// Instructions precede the instructions of every request. The Codex
	// backend refuses a request without instructions.
	Instructions string
	// ClientVersion is sent to the catalog as client_version. It names the
	// product making the call, so no library can supply an honest default.
	ClientVersion string
	// ResponsesURL and ModelsURL default to llm-provider-auth's canonical
	// Codex endpoints.
	ResponsesURL string
	ModelsURL    string
	// Client performs every request. Nil uses a client that times out after
	// 90 seconds.
	Client *http.Client
	// Now stamps catalog discovery. Nil uses time.Now.
	Now func() time.Time
}

// Codex implements core.Provider for the official Codex backend. With
// NewCodexRefresh it is the whole vertical: the Responses transport, the
// catalog and what its rows mean, and the credential refresh.
//
// Responses is the only native surface, and it streams. A
// translation.Adapter in front of Codex serves Chat Completions over it.
//
// Each operation authenticates with the credential it is given: Token is
// the bearer, and AccountID the ChatGPT account it acts for. A credential
// without a token fails before anything is sent. Codex never refreshes: a
// rejected token fails with status 401, which a Runtime refreshes once
// before it replays the request.
type Codex struct {
	transport codexTransport
}

var _ core.Provider = (*Codex)(nil)

// NewCodex returns a Codex provider.
func NewCodex(config CodexConfig) (*Codex, error) {
	transport, err := newCodexTransport(config)
	if err != nil {
		return nil, err
	}
	return &Codex{transport: transport}, nil
}

// NativeSurfaces reports Responses for every model: the Codex backend has
// no other surface.
func (p *Codex) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceResponses}
}

// Invoke performs one Responses request. Codex always streams upstream, so
// Invoke reads the stream to its terminal event and returns that response.
func (p *Codex) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	payload, authorize, err := p.prepare(ctx, request)
	if err != nil {
		return core.Response{}, err
	}
	response, err := p.transport.post(ctx, payload, authorize)
	if err != nil {
		return core.Response{}, codexFailure(ctx, err)
	}
	defer response.Body.Close()
	final, err := readCodexFinalResponse(response.Body)
	if err != nil {
		return core.Response{}, codexFailure(ctx, invocationDecodeError(err))
	}
	body, err := json.Marshal(final)
	if err != nil {
		return core.Response{}, codexFailure(ctx, invocationDecodeError(err))
	}
	return core.Response{Body: body, ContentType: core.ContentTypeJSON}, nil
}

// Stream performs one Responses request and returns its events as Codex
// sends them, each frame byte for byte, up to the terminal event.
func (p *Codex) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	payload, authorize, err := p.prepare(ctx, request)
	if err != nil {
		return nil, err
	}
	response, err := p.transport.post(ctx, payload, authorize)
	if err != nil {
		return nil, codexFailure(ctx, err)
	}
	return &codexStream{ctx: ctx, events: newCodexResponsesStreamIter(response.Body)}, nil
}

// ListModels returns the models the credential's account can use: catalog
// rows supported in the API and listed, each with the Responses endpoint
// when the catalog omits its endpoints.
func (p *Codex) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	authorize, err := codexCredential(credential)
	if err != nil {
		return nil, err
	}
	// The catalog is not a Responses endpoint, so it gets no Responses beta
	// header. The gateway always removed the one CodexProvider sends.
	models, err := p.transport.catalog(ctx, authorize, false, parseCodexModels)
	if err != nil {
		return nil, codexFailure(ctx, err)
	}
	return models, nil
}

// prepare refuses what Codex cannot serve before anything is sent, then
// shapes the request exactly as CodexProvider.CompleteResponses does.
func (p *Codex) prepare(ctx context.Context, request core.Request) (map[string]any, codexAuthorizer, error) {
	if request.Surface != core.ModelSurfaceResponses {
		return nil, nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	authorize, err := codexCredential(request.Credential)
	if err != nil {
		return nil, nil, err
	}
	payload, err := codexPayload(request)
	if err != nil {
		return nil, nil, err
	}
	// The official client always sends a list of input items. Shorthand
	// string input is the single user message it abbreviates, and the
	// gateway expanded it before it reached CodexProvider.
	if text, ok := payload["input"].(string); ok {
		payload["input"] = []any{map[string]any{"role": "user", "content": text}}
	}
	shaped, err := p.transport.nativeRequest(request.Model, payload)
	if err != nil {
		return nil, nil, codexFailure(ctx, err)
	}
	return shaped, authorize, nil
}

// codexCredential authorizes with a request's credential. Codex serves only
// ChatGPT OAuth tokens, so a credential without one, such as an API key, is
// a configuration error.
func codexCredential(credential *core.Credential) (codexAuthorizer, error) {
	if credential == nil || strings.TrimSpace(credential.Token) == "" {
		return nil, core.NewConfigurationError("Codex needs a credential with a ChatGPT access token", nil)
	}
	authorization := codexAuthorization{
		scheme: "Bearer", token: strings.TrimSpace(credential.Token), accountID: strings.TrimSpace(credential.AccountID),
	}
	return func(context.Context) (codexAuthorization, error) { return authorization, nil }, nil
}

// codexPayload decodes the JSON object body. Numbers stay json.Number, so a
// value passed through, such as a bound in a tool schema, reaches Codex as
// the caller wrote it.
func codexPayload(request core.Request) (map[string]any, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return nil, &core.ProviderError{Message: "a Codex Responses request must be JSON", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	decoder := json.NewDecoder(bytes.NewReader(request.Body))
	decoder.UseNumber()
	var payload map[string]any
	err = decoder.Decode(&payload)
	if err == nil {
		if _, trailing := decoder.Token(); trailing != io.EOF {
			err = errors.New("the body continues after its JSON object")
		}
	}
	if err != nil || payload == nil {
		return nil, &core.ProviderError{Message: "the Codex Responses request body is not a JSON object", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	return payload, nil
}

// codexStream reports a failure mid-stream as a *core.ProviderError too.
type codexStream struct {
	ctx    context.Context
	events *codexResponsesStreamIter
}

func (s *codexStream) Next() ([]byte, error) {
	frame, err := s.events.Next()
	if err != nil && err != io.EOF {
		return nil, codexFailure(s.ctx, err)
	}
	return frame, err
}

func (s *codexStream) Close() error { return s.events.Close() }

// codexFailure returns a transport failure as the canonical
// *core.ProviderError. The transport's *InvocationError or *CatalogError
// remains its cause, so errors.As still finds it.
//
// A failure because the caller gave up permits nothing and counts against
// nothing. Go's client timeout wraps context.DeadlineExceeded too, but
// while the caller still waits it is a slow upstream, which another
// attempt may get past.
func codexFailure(ctx context.Context, err error) error {
	var invocation *InvocationError
	var catalog *CatalogError
	var failure *core.ProviderError
	switch {
	case errors.As(err, &invocation):
		failure = codexInvocationFailure(invocation)
	case errors.As(err, &catalog):
		failure = codexCatalogFailure(catalog)
	default:
		return err
	}
	if callerCancellation(err) {
		failure.Classification = core.ProviderErrorClassification{}
		if ctx.Err() == nil {
			failure.Class = core.ProviderErrorTransport
			failure.Classification = core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}
		}
	}
	return failure
}

func codexInvocationFailure(e *InvocationError) *core.ProviderError {
	failure := &core.ProviderError{Message: e.Msg, Class: e.class, Classification: e.ProviderErrorClassification(), Cause: e}
	switch {
	case e.Status != 0:
		failure.Class = core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: e.Status}).ErrorClass
	case e.class == core.ProviderErrorUnsupported, e.class == core.ProviderErrorConfiguration:
		// Another target may serve the request.
		failure.Classification = core.ProviderErrorClassification{FailoverEligible: true}
	case e.class == core.ProviderErrorUpstream:
		// Codex answered with a malformed stream. Another target may still
		// serve the request.
		failure.Classification = core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}
	}
	return failure
}

func codexCatalogFailure(e *CatalogError) *core.ProviderError {
	failure := &core.ProviderError{Message: e.Detail, Cause: e}
	switch e.Code {
	case "http_error":
		// Classified like an inference failure with the same status.
		failure.Class = core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: e.Status}).ErrorClass
		failure.Classification = (&InvocationError{Status: e.Status, RetryAfter: e.RetryAfter}).ProviderErrorClassification()
	case "transport_error":
		failure.Class = core.ProviderErrorTransport
		failure.Classification = core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}
	case "invalid_response":
		failure.Class = core.ProviderErrorUpstream
		failure.Classification = core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}
	default:
		failure.Class = core.ProviderErrorConfiguration
		failure.Classification = core.ProviderErrorClassification{FailoverEligible: true}
	}
	return failure
}
