package providers

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers/zen"
)

// ZenConfig configures Zen. Every field is optional.
type ZenConfig struct {
	// BaseURL and MetadataURL default to zen.DefaultBaseURL and
	// zen.DefaultMetadataURL.
	BaseURL     string
	MetadataURL string
	// ProviderID owns the models anonymous access admits. Empty uses
	// zen.DefaultProviderID, as the gateway does.
	ProviderID string
	// Client performs inference. Nil uses a client that times out after 90
	// seconds.
	Client *http.Client
	// CatalogClient lists models. Nil uses a client that times out after 10
	// seconds, the gateway's bound on catalog requests.
	CatalogClient *http.Client
	// Models returns the row a product's catalog holds for a model, as
	// ListModels returned it. Zen serves the model on the surfaces the row
	// lists, and without a credential only if the row is tagged
	// ModelTagFree. A model the catalog lacks, or a nil Models, is served as
	// the gateway serves it with a cold catalog: a muse-spark- model over
	// Responses, any other over Chat Completions, and anonymously alike.
	Models func(model string) (core.ModelInfo, bool)
	// Now stamps catalogs and completions that carry no creation time. Nil
	// uses time.Now.
	Now func() time.Time
	// NewID makes the session and request IDs of an invocation whose
	// context carries no zen.InvocationIdentity. Nil uses random IDs.
	NewID func(prefix string) (string, error)
	// CapabilityTTL is how long listed capabilities stay fresh. Zero or
	// less uses one hour.
	CapabilityTTL time.Duration
	// MaxCatalogBytes bounds each catalog document. Zero or less uses 8
	// MiB, the gateway's bound.
	MaxCatalogBytes int64
}

// Zen implements core.Provider for OpenCode Zen. It is the whole vertical:
// request shaping, anonymous access and its invocation identity, both
// streams, and the catalog and what its rows mean. Requests are shaped
// byte for byte as the gateway shapes them.
//
// A model is served on its native surface: Chat Completions, Responses, or
// both, as its catalog row lists them (see ZenConfig.Models). Zen serves
// Chat natively for a Responses-only model too, converting it to
// Responses as the gateway does; a translation.Adapter in front serves
// Responses for a Chat model and Messages for any.
//
// A credential's API key, or its token, authenticates as a bearer. Without
// a credential, or with one of the gateway's anonymous sentinels, Zen
// sends the anonymous identity: the public bearer, and admission shaping
// that makes a request look like the OpenCode CLI's. Every request carries
// the invocation identity of zen.InvocationIdentityFromContext, or a new
// one per operation; put one in the context to keep it across retries.
type Zen struct {
	baseURL, metadataURL, providerID string
	client, catalogClient            *http.Client
	models                           func(string) (core.ModelInfo, bool)
	now                              func() time.Time
	newID                            func(string) (string, error)
	capabilityTTL                    time.Duration
	maxCatalogBytes                  int64
}

var _ core.Provider = (*Zen)(nil)

// NewZen returns a Zen provider.
func NewZen(config ZenConfig) (*Zen, error) {
	p := &Zen{
		providerID: strings.TrimSpace(config.ProviderID), client: config.Client, catalogClient: config.CatalogClient,
		models: config.Models, now: config.Now, newID: config.NewID,
		capabilityTTL: config.CapabilityTTL, maxCatalogBytes: config.MaxCatalogBytes,
	}
	var err error
	if p.baseURL, err = zenURL(config.BaseURL, zen.DefaultBaseURL); err != nil {
		return nil, core.NewConfigurationError("the OpenCode Zen base URL must be an absolute URL", err)
	}
	if p.metadataURL, err = zenURL(config.MetadataURL, zen.DefaultMetadataURL); err != nil {
		return nil, core.NewConfigurationError("the models.dev URL must be an absolute URL", err)
	}
	if p.providerID == "" {
		p.providerID = zen.DefaultProviderID
	}
	if p.client == nil {
		p.client = &http.Client{Timeout: 90 * time.Second}
	}
	if p.catalogClient == nil {
		p.catalogClient = &http.Client{Timeout: 10 * time.Second}
	}
	if p.now == nil {
		p.now = time.Now
	}
	if p.capabilityTTL <= 0 {
		p.capabilityTTL = time.Hour
	}
	if p.maxCatalogBytes <= 0 {
		p.maxCatalogBytes = 8 << 20
	}
	return p, nil
}

func zenURL(value, fallback string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		value = fallback
	}
	parsed, err := url.Parse(value)
	if err == nil && (parsed.Scheme == "" || parsed.Host == "") {
		err = fmt.Errorf("%q is not absolute", value)
	}
	return strings.TrimRight(value, "/"), err
}

// zenRoute is how Zen serves one model.
type zenRoute struct {
	row   core.ModelInfo
	known bool
	// responses reports Responses served natively; chatOverResponses, Chat
	// served by converting it to Responses, for a Responses-only model.
	responses, chatOverResponses bool
}

// route reads the model's catalog row as the gateway does. Without one,
// the gateway's cold-catalog contract, kept since v0.6.6, holds: Muse
// models serve Responses only.
func (p *Zen) route(model string) zenRoute {
	if p.models != nil {
		if row, ok := p.models(model); ok {
			chat, responses := zenRowSurfaces(row)
			return zenRoute{row: row, known: true, responses: responses, chatOverResponses: responses && !chat}
		}
	}
	muse := strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "muse-spark-")
	return zenRoute{responses: muse, chatOverResponses: muse}
}

// NativeSurfaces reports Chat Completions for every model, and Responses
// for a model whose catalog row lists it, first when it is the only one.
func (p *Zen) NativeSurfaces(model string) []core.ModelSurface {
	route := p.route(model)
	switch {
	case route.chatOverResponses:
		return []core.ModelSurface{core.ModelSurfaceResponses, core.ModelSurfaceChatCompletions}
	case route.responses:
		return []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses}
	}
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

// zenCall is one operation Zen has checked and is about to send.
type zenCall struct {
	model    string
	payload  map[string]any
	access   zenAccess
	route    zenRoute
	identity zen.InvocationIdentity
}

// Invoke performs one Chat Completions or Responses request.
func (p *Zen) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	call, err := p.prepare(ctx, request)
	if err != nil {
		return core.Response{}, err
	}
	switch {
	case request.Surface == core.ModelSurfaceResponses:
		return p.invokeResponses(ctx, call)
	case call.route.chatOverResponses:
		return p.invokeChatOverResponses(ctx, call)
	}
	return p.invokeChat(ctx, call)
}

// Stream performs one Chat Completions or Responses request. A native
// stream passes Zen's records through byte for byte; see zenResponsesStream
// for which Responses records it keeps. Chat over Responses is converted
// from the complete response, as the gateway converts it, and ends in
// [DONE]. The stream reports the request's losses through
// core.LossReporter.
func (p *Zen) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	call, err := p.prepare(ctx, request)
	if err != nil {
		return nil, err
	}
	switch {
	case request.Surface == core.ModelSurfaceResponses:
		return p.streamResponses(ctx, call)
	case call.route.chatOverResponses:
		return p.streamChatOverResponses(ctx, call)
	}
	return p.streamChat(ctx, call)
}

// prepare refuses what Zen cannot serve before anything is sent.
func (p *Zen) prepare(ctx context.Context, request core.Request) (zenCall, error) {
	route := p.route(request.Model)
	if request.Surface != core.ModelSurfaceChatCompletions && (request.Surface != core.ModelSurfaceResponses || !route.responses) {
		return zenCall{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	if strings.TrimSpace(request.Model) == "" {
		return zenCall{}, &core.ProviderError{Message: "an OpenCode Zen request needs a model", Class: core.ProviderErrorInvalidRequest}
	}
	access := zenCredential(request.Credential)
	if access.anonymous && route.known && !slices.Contains(route.row.Tags, ModelTagFree) {
		// The catalog knows the model and anonymous access does not admit
		// it, so the anonymous identity is not sent for it. Nothing
		// reached Zen, so the refusal carries no status.
		return zenCall{}, &core.ProviderError{
			Message: zenAnonymousRefusal(request.Model), Class: core.ProviderErrorAuth,
			Classification: core.ProviderErrorClassification{FailoverEligible: true},
		}
	}
	payload, err := zenPayload(request)
	if err != nil {
		return zenCall{}, err
	}
	identity, ok := zen.InvocationIdentityFromContext(ctx)
	if !ok {
		if identity, err = zen.NewInvocationIdentity(nil, p.newID); err != nil {
			return zenCall{}, core.NewConfigurationError("OpenCode Zen could not create an invocation identity", err)
		}
	}
	return zenCall{model: request.Model, payload: payload, access: access, route: route, identity: identity}, nil
}

// ListModels returns what the credential can use. A keyed credential lists
// Zen's catalog as the gateway reads it. Anonymous access lists the models
// it admits, each tagged ModelTagFree: those the models.dev catalog prices
// at exactly zero and the live catalog, asked with a new anonymous
// identity, lists.
func (p *Zen) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	access := zenCredential(credential)
	if !access.anonymous {
		header := http.Header{}
		header.Set("Content-Type", core.ContentTypeJSON)
		header.Set("Authorization", access.authorization)
		raw, err := p.fetch(ctx, p.baseURL+"/models", header, "")
		if err != nil {
			return nil, err
		}
		return parseZenModels(raw, p.now())
	}
	observedAt := p.now().UTC()
	metadata, err := p.fetch(ctx, p.metadataURL, http.Header{"Accept": {core.ContentTypeJSON}}, "metadata_")
	if err != nil {
		return nil, err
	}
	identity, err := zen.NewInvocationIdentity(nil, p.newID)
	if err != nil {
		return nil, core.NewConfigurationError("OpenCode Zen could not create an invocation identity", err)
	}
	header := zenHeaders(zenAccess{anonymous: true, authorization: zenPublicBearer}, identity)
	header.Set("Accept", core.ContentTypeJSON)
	live, err := p.fetch(ctx, p.baseURL+"/models", header, "")
	if err != nil {
		return nil, err
	}
	return parseZenAnonymousModels(metadata, live, zen.NormalizeOptions{
		ProviderID: p.providerID, ObservedAt: observedAt, CapabilityTTL: p.capabilityTTL,
	})
}

// zenAnonymousRefusal is the gateway's guidance for a model anonymous
// access does not serve.
func zenAnonymousRefusal(model string) string {
	return fmt.Sprintf("OpenCode Zen model %q is not available through the current anonymous catalog; configure an OpenCode Zen API key for paid models.", model)
}

// zenFailure returns a transport failure as the canonical
// *core.ProviderError, classified as Codex classifies its own: the
// transport's *InvocationError stays the cause, a caller that gave up
// permits nothing, and a client timeout while the caller still waits is a
// slow upstream.
func zenFailure(ctx context.Context, err error) error {
	return codexFailure(ctx, err)
}
