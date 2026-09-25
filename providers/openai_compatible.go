package providers

import (
	"context"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// The gateway's default OpenAI-compatible timeout, and its bound on a
// catalog request.
const (
	openAICompatibleTimeout        = 300 * time.Second
	openAICompatibleCatalogTimeout = 10 * time.Second
)

// OpenAICompatibleConfig configures OpenAICompatible. BaseURL is required.
type OpenAICompatibleConfig struct {
	// BaseURL is the API base the operation paths follow, such as
	// https://api.openai.com/v1: /chat/completions, /responses and /models.
	BaseURL string
	// RegistryID is the registry entry the instance serves, as a Registry
	// names it. It selects the wire facts the gateway keys on an entry:
	// "openai" serves Responses for every model, and kilo_code, llm7,
	// ovh_ai_endpoints and pollinations list, without a key, only the
	// models anonymous access admits. Pollinations also takes Chat at
	// /v1/chat/completions and lists its catalog as a bare array.
	RegistryID string
	// Models returns the row a product's catalog holds for a model, as
	// ListModels returned it. A model whose row lists a Responses endpoint
	// serves Responses natively, and adaptation reads the row too. A model
	// the catalog lacks, or a nil Models, serves Chat Completions only,
	// unless RegistryID is "openai".
	Models func(model string) (core.ModelInfo, bool)
	// Headers are sent with every request, before a credential's own
	// headers. Neither replaces a header the transport sets.
	Headers map[string]string
	// ForwardAllFields sends every Chat field a request sets to the Chat
	// endpoint. By default only the fields the gateway's transport forwards
	// are sent, and each other field is dropped and reported as an advisory
	// loss. Chat served over Responses carries only the fields the
	// gateway's Chat facade forwards, either way.
	ForwardAllFields bool
	// ForceAPISupport turns the gateway's API adaptation on for every Chat
	// request, as its provider-level force_api_support does: a model whose
	// row lists Responses but not Chat Completions is served over
	// Responses, and a model whose row lists reasoning efforts takes
	// max_tokens as max_completion_tokens. A request's boolean
	// force_api_support turns it on or off for that request.
	ForceAPISupport bool
	// Client performs inference. Nil uses a client that times out after
	// 300 seconds, the gateway's default.
	Client *http.Client
	// CatalogClient lists models. Nil uses a client that times out after
	// 10 seconds, the gateway's bound on catalog requests.
	CatalogClient *http.Client
	// Now stamps catalog discovery and reads Retry-After dates. Nil uses
	// time.Now.
	Now func() time.Time
}

// OpenAICompatible implements core.Provider for an upstream that speaks the
// OpenAI wire: the gateway's openai_compatible, openai and litellm
// instances, and the anonymous catalogs the registry curates. It is the
// gateway's OpenAI transport without its OpenCode Zen and GitHub Copilot
// branches, which Zen and Copilot own.
//
// Chat Completions is native for every model, and Responses for a model
// whose catalog row lists it or that the openai registry entry serves. A
// translation.Adapter in front serves Messages over Chat, and Responses
// over Chat for the other models.
//
// A credential's API key, or else its token, is the bearer, without any
// "Bearer " prefix. No key, "free" or "none" sends no Authorization, and
// to an anonymous registry entry it is anonymous access, as "public" is.
// A credential's headers are sent too.
//
// Errors are *core.ProviderError. A refusal is classified by the gateway's
// status set with the upstream's Retry-After, and its message never quotes
// the upstream; a catalog failure's *CatalogError has a CatalogCode* code.
type OpenAICompatible struct {
	baseURL, registryID, label string
	models                     func(string) (core.ModelInfo, bool)
	headers                    map[string]string
	forwardAll, adapt          bool
	client, catalogClient      *http.Client
	now                        func() time.Time
}

var (
	_ core.Provider      = (*OpenAICompatible)(nil)
	_ core.WirePreserver = (*OpenAICompatible)(nil)
)

// NewOpenAICompatible returns an OpenAICompatible provider.
func NewOpenAICompatible(config OpenAICompatibleConfig) (*OpenAICompatible, error) {
	base := strings.TrimSpace(config.BaseURL)
	parsed, err := url.Parse(base)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, core.NewConfigurationError("the OpenAI-compatible base URL must be an absolute URL", err)
	}
	p := &OpenAICompatible{
		baseURL: strings.TrimRight(base, "/"), registryID: strings.ToLower(strings.TrimSpace(config.RegistryID)),
		label: "OpenAI-compatible", models: config.Models, headers: maps.Clone(config.Headers),
		forwardAll: config.ForwardAllFields, adapt: config.ForceAPISupport,
		client: config.Client, catalogClient: config.CatalogClient, now: config.Now,
	}
	if p.client == nil {
		p.client = &http.Client{Timeout: openAICompatibleTimeout}
	}
	if p.catalogClient == nil {
		p.catalogClient = &http.Client{Timeout: openAICompatibleCatalogTimeout}
	}
	if p.now == nil {
		p.now = time.Now
	}
	return p, nil
}

// NativeSurfaces reports Chat Completions for every model, and Responses
// for a model the openai registry entry serves or whose row lists it.
func (p *OpenAICompatible) NativeSurfaces(model string) []core.ModelSurface {
	if p.responsesNative(model) {
		return []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses}
	}
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

// PreservesWire implements core.WirePreserver. Responses is forwarded as
// sent, and so is Chat on the Chat endpoint: the model and stream flag are
// set, and fields the transport does not forward are dropped, but nothing
// is converted. Chat that the configured adaptation serves over Responses
// is converted, so it is not preserved; neither is a request whose own
// force_api_support turns adaptation on, which this declaration cannot see.
func (p *OpenAICompatible) PreservesWire(model string, surface core.ModelSurface) bool {
	switch surface {
	case core.ModelSurfaceResponses:
		return p.responsesNative(model)
	case core.ModelSurfaceChatCompletions:
		return !p.adaptsToResponses(model, p.adapt)
	}
	return false
}

// Invoke performs one Chat Completions or Responses request and returns
// the upstream's answer as sent, or, for Chat that adaptation serves over
// Responses, the answer converted to Chat.
func (p *OpenAICompatible) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	call, err := p.prepare(request)
	if err != nil {
		return core.Response{}, err
	}
	if request.Surface == core.ModelSurfaceResponses {
		return p.invokeResponses(ctx, call)
	}
	return p.invokeChat(ctx, call)
}

// Stream performs one streaming Chat Completions or Responses request. Its
// frames are the upstream's SSE records as sent, [DONE] included; Chat that
// adaptation serves over Responses streams the complete answer as Chat
// chunks ending in [DONE], as the gateway does. The stream reports the
// request's losses through core.LossReporter.
func (p *OpenAICompatible) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	call, err := p.prepare(request)
	if err != nil {
		return nil, err
	}
	if request.Surface == core.ModelSurfaceResponses {
		return p.streamResponses(ctx, call)
	}
	return p.streamChat(ctx, call)
}

// openAICompatibleCall is one operation checked and about to be shaped.
type openAICompatibleCall struct {
	model   string
	payload map[string]any
	access  openAIAccess
}

// prepare refuses what the provider cannot serve before anything is sent.
func (p *OpenAICompatible) prepare(request core.Request) (openAICompatibleCall, error) {
	if request.Surface != core.ModelSurfaceChatCompletions && (request.Surface != core.ModelSurfaceResponses || !p.responsesNative(request.Model)) {
		return openAICompatibleCall{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	if strings.TrimSpace(request.Model) == "" {
		return openAICompatibleCall{}, openAIInvalid("an OpenAI-compatible request needs a model", nil)
	}
	access, err := p.access(request.Credential)
	if err != nil {
		return openAICompatibleCall{}, err
	}
	payload, err := openAIPayload(request)
	if err != nil {
		return openAICompatibleCall{}, err
	}
	return openAICompatibleCall{model: request.Model, payload: payload, access: access}, nil
}

func (p *OpenAICompatible) row(model string) (core.ModelInfo, bool) {
	if p.models == nil {
		return core.ModelInfo{}, false
	}
	return p.models(model)
}

// responsesNative reports Responses served natively, as the gateway
// decides: for every model of the openai registry entry, and for a model
// whose row lists a Responses endpoint, its websocket form included.
func (p *OpenAICompatible) responsesNative(model string) bool {
	if p.registryID == "openai" {
		return true
	}
	row, ok := p.row(model)
	return ok && slices.ContainsFunc(row.SupportedAPIs, openAIResponsesEndpoint)
}

// adaptsToResponses reports Chat that adaptation serves over Responses: a
// model whose row lists Responses but not Chat Completions, as
// translate.PreferredEndpoint reads the row.
func (p *OpenAICompatible) adaptsToResponses(model string, adapt bool) bool {
	if !adapt {
		return false
	}
	row, ok := p.row(model)
	return ok && translate.PreferredEndpoint(row.SupportedAPIs) == "responses"
}

func openAIResponsesEndpoint(endpoint string) bool {
	switch strings.ToLower(strings.TrimSpace(endpoint)) {
	case "/responses", "/v1/responses", "ws:/responses":
		return true
	}
	return false
}
