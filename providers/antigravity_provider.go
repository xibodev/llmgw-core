package providers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// AntigravityConfig configures Antigravity. Every field is optional.
type AntigravityConfig struct {
	// BaseURL is the Cloud Code Assist endpoint. Empty uses Google's.
	BaseURL string
	// Client performs every request. Nil uses a client that times out after
	// 120 seconds.
	Client *http.Client
	// ProjectResolved receives the Code Assist project that Antigravity
	// discovered for a credential whose Metadata named none, so that the
	// product can store it under core.CredentialMetadataProjectID and later
	// operations skip discovery. The credential is the one the operation
	// used, which the hook must not modify. A product fences its write on
	// the credential's token, as the gateway does, because a credential
	// reauthorized meanwhile may act for another account.
	//
	// It runs only once the operation has succeeded. Storing the project
	// changes the credential's revision, and a Runtime refreshes a rejected
	// credential only while it keeps the revision the request used. A
	// product that does not store the project discovers it again next time.
	ProjectResolved func(ctx context.Context, credential *core.Credential, projectID string)
}

// Antigravity implements core.Provider for Google Antigravity, over Google's
// undocumented Cloud Code Assist v1internal API. With NewAntigravityRefresh
// it is the whole vertical: the Chat transport, image generation, the
// catalog and what its rows mean, the project each credential uses, and
// the credential refresh.
//
// Chat Completions is its only surface. Antigravity maps the messages,
// max_tokens, temperature and tools to Gemini exactly as the gateway does,
// and drops every other field, reporting it in Response.Losses. Cloud Code
// Assist always streams, and Antigravity reads the stream to its end, so
// Invoke serves a request and Stream refuses one, as the gateway does. A
// translation.Adapter in front serves Messages and Responses over Chat,
// without streaming. GenerateImages implements core.ImageGenerator.
//
// Each operation authenticates with the credential it is given: Token is
// the bearer, and the project is the one its Metadata names under
// core.CredentialMetadataProjectID, or else the one loadCodeAssist
// discovers. A credential without a token fails before anything is sent.
// Antigravity never refreshes: a rejected token fails with status 401,
// which a Runtime refreshes once before it replays the request.
type Antigravity struct {
	transport       antigravityTransport
	projectResolved func(context.Context, *core.Credential, string)
}

var (
	_ core.Provider       = (*Antigravity)(nil)
	_ core.ImageGenerator = (*Antigravity)(nil)
)

// NewAntigravity returns an Antigravity provider.
func NewAntigravity(config AntigravityConfig) (*Antigravity, error) {
	base := strings.TrimSpace(config.BaseURL)
	if base != "" {
		if parsed, err := url.Parse(base); err != nil || (parsed.Scheme != "https" && parsed.Scheme != "http") || parsed.Host == "" {
			return nil, errors.New("Antigravity base URL must be an absolute HTTP URL")
		}
	}
	return &Antigravity{transport: newAntigravityTransport(config.Client, base), projectResolved: config.ProjectResolved}, nil
}

// NativeSurfaces reports Chat Completions for every model.
func (p *Antigravity) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

// Invoke performs one Chat Completions request and returns the completion
// Cloud Code Assist streamed, read to its end.
func (p *Antigravity) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	if request.Surface != core.ModelSurfaceChatCompletions {
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	token, err := antigravityToken(request.Credential)
	if err != nil {
		return core.Response{}, err
	}
	if strings.TrimSpace(request.Model) == "" {
		return core.Response{}, &core.ProviderError{Message: "an Antigravity request needs a model", Class: core.ProviderErrorInvalidRequest}
	}
	payload, losses, err := antigravityChatPayload(request)
	if err != nil {
		return core.Response{}, err
	}
	mapped, err := mapAntigravityRequest(payload)
	if err != nil {
		return core.Response{}, &core.ProviderError{
			Message: "Antigravity cannot serve this Chat request: " + err.Error(), Class: core.ProviderErrorUnsupported,
			Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
		}
	}
	project, err := p.project(ctx, request.Credential, token)
	if err != nil {
		return core.Response{}, err
	}
	result, err := p.transport.complete(ctx, token, project.id, request.Model, mapped)
	if err != nil {
		return core.Response{}, antigravityFailure(ctx, err)
	}
	body, err := json.Marshal(result)
	if err != nil {
		return core.Response{}, antigravityFailure(ctx, core.NewProviderOperationError("antigravity completion response decode", http.StatusOK, "", err))
	}
	p.resolved(ctx, request.Credential, project)
	return core.Response{Body: body, ContentType: core.ContentTypeJSON, Losses: losses}, nil
}

// Stream refuses every request, as the gateway does, because Antigravity
// reads Cloud Code Assist's stream to its end rather than relaying it.
// Nothing is sent. The refusal wraps
// ErrExperimentalAntigravityStreamingUnsupported and permits failover to a
// target that streams.
func (p *Antigravity) Stream(_ context.Context, request core.Request) (core.StreamIter, error) {
	if request.Surface != core.ModelSurfaceChatCompletions {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	return nil, &core.ProviderError{
		Message: "Antigravity does not stream", Class: core.ProviderErrorUnsupported,
		Classification: core.ProviderErrorClassification{FailoverEligible: true},
		Cause:          ErrExperimentalAntigravityStreamingUnsupported,
	}
}

// ListModels returns the models the credential's project may use, with the
// rows and capabilities the gateway lists for them.
func (p *Antigravity) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	token, err := antigravityToken(credential)
	if err != nil {
		return nil, err
	}
	project, err := p.project(ctx, credential, token)
	if err != nil {
		return nil, err
	}
	raw, status, err := p.transport.catalog(ctx, token, project.id)
	if err != nil {
		return nil, antigravityFailure(ctx, err)
	}
	models, err := parseAntigravityModels(raw)
	if err != nil {
		return nil, antigravityFailure(ctx, antigravityCatalogDecodeError(status, err))
	}
	p.resolved(ctx, credential, project)
	return models, nil
}

// GenerateImages generates one image with a model that the root roster and
// the image roster of the credential's catalog both name, read in the same
// operation, as the gateway generates it. A request Antigravity refuses
// before sending anything keeps the status 400 of the legacy adapter in its
// cause, where the gateway reads it.
func (p *Antigravity) GenerateImages(ctx context.Context, request core.GenerateImagesRequest, credential *core.Credential) (core.GenerateImagesResult, error) {
	request, err := antigravityImageRequest(request)
	if err != nil {
		return core.GenerateImagesResult{}, &core.ProviderError{
			Message: "the image request cannot be sent to Antigravity: " + err.Error(), Class: core.ProviderErrorInvalidRequest,
			Cause: antigravityImageRefusal(err),
		}
	}
	token, err := antigravityToken(credential)
	if err != nil {
		return core.GenerateImagesResult{}, err
	}
	project, err := p.project(ctx, credential, token)
	if err != nil {
		return core.GenerateImagesResult{}, err
	}
	catalog, err := p.transport.decodedCatalog(ctx, token, project.id)
	if err != nil {
		return core.GenerateImagesResult{}, antigravityFailure(ctx, err)
	}
	if err := antigravityImageModel(catalog, request.Model); err != nil {
		return core.GenerateImagesResult{}, &core.ProviderError{
			Message: "the Antigravity catalog does not list this model for image generation", Class: core.ProviderErrorUnsupported,
			Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
		}
	}
	result, err := p.transport.generateImage(ctx, token, project.id, request)
	if err != nil {
		return core.GenerateImagesResult{}, antigravityFailure(ctx, err)
	}
	p.resolved(ctx, credential, project)
	return result, nil
}

// antigravityToken is the bearer of a request's credential. Antigravity
// serves only Google OAuth tokens, so a credential without one, such as an
// API key, is a configuration error.
func antigravityToken(credential *core.Credential) (string, error) {
	if credential == nil || strings.TrimSpace(credential.Token) == "" {
		return "", core.NewConfigurationError("Antigravity needs a credential with a Google OAuth access token", nil)
	}
	return strings.TrimSpace(credential.Token), nil
}

// antigravityProject is the project one operation uses, and whether the
// operation discovered it.
type antigravityProject struct {
	id         string
	discovered bool
}

// project returns the project credential names, or else the one
// loadCodeAssist discovers for token, as the gateway resolves it.
func (p *Antigravity) project(ctx context.Context, credential *core.Credential, token string) (antigravityProject, error) {
	if id := strings.TrimSpace(credential.Metadata[core.CredentialMetadataProjectID]); id != "" {
		return antigravityProject{id: id}, nil
	}
	id, err := p.transport.loadCodeAssist(ctx, token)
	if errors.Is(err, errAntigravityNoProject) {
		// No request can succeed for an account without a project, and
		// another target may serve it.
		return antigravityProject{}, core.NewConfigurationError("Antigravity found no Code Assist project for this account", err)
	}
	if err != nil {
		return antigravityProject{}, antigravityFailure(ctx, err)
	}
	return antigravityProject{id: id, discovered: true}, nil
}

// resolved reports a discovered project once its operation has succeeded.
func (p *Antigravity) resolved(ctx context.Context, credential *core.Credential, project antigravityProject) {
	if project.discovered && p.projectResolved != nil {
		p.projectResolved(ctx, credential, project.id)
	}
}

// antigravityFailure returns a transport failure as the canonical
// *core.ProviderError. The transport's *core.ProviderOperationError stays
// its cause, so a product that reads the upstream status and Retry-After
// from it, as the gateway does, still finds them.
//
// A failure because the caller gave up permits nothing and counts against
// nothing. Go's client timeout wraps context.DeadlineExceeded too, but
// while the caller still waits it is a slow upstream, which another attempt
// may get past.
func antigravityFailure(ctx context.Context, err error) error {
	var operation *core.ProviderOperationError
	if !errors.As(err, &operation) {
		// Only building a request fails before it is sent.
		return core.NewConfigurationError("the Antigravity request could not be built", err)
	}
	failure := &core.ProviderError{Message: operation.Error(), Cause: err}
	switch status := operation.Failure.StatusCode; {
	case status == 0:
		failure.Class = core.ProviderErrorTransport
		failure.Classification = core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}
	case status < 200 || status >= 300:
		failure.Class = core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: status}).ErrorClass
		retryAfter := parseRetryAfterHeader(operation.Failure.RetryAfter, time.Now())
		failure.Classification = (&InvocationError{Status: status, RetryAfter: retryAfter}).ProviderErrorClassification()
	default:
		// Cloud Code Assist answered with a body Antigravity cannot read.
		// Another target may still serve the request.
		failure.Class = core.ProviderErrorUpstream
		failure.Classification = core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}
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
