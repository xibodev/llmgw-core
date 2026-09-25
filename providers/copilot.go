package providers

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	copilotauth "github.com/xibodev/llm-provider-auth/copilot"
	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// The editor identity the gateway presents to Copilot unless configured.
const (
	copilotDefaultIntegrationID = "vscode-chat"
	copilotDefaultEditorVersion = "vscode/1.95.3"
)

// The gateway's default Copilot timeout, and its bound on a catalog request.
const (
	copilotDefaultTimeout = 300 * time.Second
	copilotCatalogTimeout = 10 * time.Second
)

// CopilotConfig configures Copilot. Auth, EditorPluginVersion and UserAgent
// are required.
type CopilotConfig struct {
	// Auth is the product's llm-provider-auth Copilot client. It exchanges
	// GitHub OAuth tokens for session tokens, and it resolves the product's
	// own token, configured, cached or from the gh CLI, for a request that
	// carries no credential. Every operation fails with a configuration
	// error unless its AllowProxy setting is on.
	Auth *copilotauth.Client
	// IntegrationID and EditorVersion identify the editor integration in the
	// Copilot-Integration-Id and Editor-Version headers of API requests.
	// Empty fields use the gateway's defaults, vscode-chat and vscode/1.95.3.
	// The session exchange presents Auth's own editor identity, which the
	// gateway leaves at llm-provider-auth's defaults.
	IntegrationID string
	EditorVersion string
	// EditorPluginVersion and UserAgent name the product making the call, so
	// no library can supply an honest default. The gateway sends
	// llm-gateway/0.1 and GithubCopilotChat/llm-gateway.
	EditorPluginVersion string
	UserAgent           string
	// Client performs every API request. Nil uses a client that times out
	// after 300 seconds, the gateway's default Copilot timeout. A catalog
	// request is bounded to 10 seconds besides, as in the gateway.
	Client *http.Client
	// Now stamps catalog discovery and ages sessions. Nil uses time.Now.
	Now func() time.Time
	// DisableAdaptation sends every Chat request to Chat Completions, as the
	// gateway does when its Copilot API adaptation is off. By default a Chat
	// request for a model whose catalog row lists only Responses is served
	// over Responses. A request's force_api_support field overrides either.
	DisableAdaptation bool
}

// Copilot implements core.Provider for GitHub Copilot. It is the whole
// vertical the gateway ran in its OpenAI transport: session handling, request
// shaping, the catalog and what its rows mean.
//
// A request's credential is the caller's GitHub OAuth token. Copilot
// exchanges it for a session token through Auth and keeps the session per
// credential until it nears expiry. A request without a credential uses the
// product's own token through Auth. A credential without a token is a
// configuration error, and nothing is sent.
//
// Chat Completions is native for every model and streams; for a model whose
// catalog row lists only Responses, Copilot serves Chat over Responses
// itself, as the gateway's adaptation does. Responses is native for a model
// whose catalog row lists a Responses endpoint, as the gateway decides, so
// list the models first; a translation.Adapter serves Responses for the rest
// over Chat, and Messages over Chat for all.
//
// Copilot rejecting a session with 401 is retried once with a new session.
// A second 401, or GitHub rejecting the OAuth token itself, fails with
// status 401, which a Runtime refreshes or reports.
type Copilot struct {
	sessions *copilotSessions
	identity copilotIdentity
	client   *http.Client
	now      func() time.Time
	adapt    bool

	mu     sync.RWMutex
	routes map[string]copilotRoute
}

var _ core.Provider = (*Copilot)(nil)

// NewCopilot returns a Copilot provider.
func NewCopilot(config CopilotConfig) (*Copilot, error) {
	if config.Auth == nil {
		return nil, errors.New("Copilot auth client is required")
	}
	if strings.TrimSpace(config.EditorPluginVersion) == "" {
		return nil, errors.New("Copilot editor plugin version is required")
	}
	if strings.TrimSpace(config.UserAgent) == "" {
		return nil, errors.New("Copilot user agent is required")
	}
	identity := copilotIdentity{
		integrationID: config.IntegrationID, editorVersion: config.EditorVersion,
		pluginVersion: config.EditorPluginVersion, userAgent: config.UserAgent,
	}
	if identity.integrationID == "" {
		identity.integrationID = copilotDefaultIntegrationID
	}
	if identity.editorVersion == "" {
		identity.editorVersion = copilotDefaultEditorVersion
	}
	client := config.Client
	if client == nil {
		client = &http.Client{Timeout: copilotDefaultTimeout}
	}
	now := config.Now
	if now == nil {
		now = time.Now
	}
	return &Copilot{
		sessions: newCopilotSessions(config.Auth, now), identity: identity,
		client: client, now: now, adapt: !config.DisableAdaptation,
		routes: map[string]copilotRoute{},
	}, nil
}

// NativeSurfaces reports Chat Completions for every model, and Responses
// too for a model whose latest catalog row lists a Responses endpoint.
func (p *Copilot) NativeSurfaces(model string) []core.ModelSurface {
	if route, ok := p.route(model); ok && route.responses() {
		return []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses}
	}
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

// Invoke performs one Chat Completions or Responses request. Chat returns
// Copilot's answer, or, for a model served over Responses, the answer
// converted to Chat.
func (p *Copilot) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	switch {
	case request.Surface == core.ModelSurfaceChatCompletions:
		return p.invokeChat(ctx, request)
	case request.Surface == core.ModelSurfaceResponses && core.ServesNatively(p, request.Model, request.Surface):
		return p.invokeResponses(ctx, request)
	}
	return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
}

// Stream performs one streaming Chat Completions or Responses request. Its
// frames are Copilot's SSE records as sent, or, for a model served over
// Responses, Chat chunks ending in [DONE]. The stream reports the request's
// losses through core.LossReporter.
func (p *Copilot) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	switch {
	case request.Surface == core.ModelSurfaceChatCompletions:
		return p.streamChat(ctx, request)
	case request.Surface == core.ModelSurfaceResponses && core.ServesNatively(p, request.Model, request.Surface):
		return p.streamResponses(ctx, request)
	}
	return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
}

// ListModels returns the catalog the credential's account can use, every
// row the gateway lists, and remembers what each row says about serving its
// model.
func (p *Copilot) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	response, _, err := p.send(ctx, credential, copilotCall{method: http.MethodGet, path: "/models", timeout: copilotCatalogTimeout})
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, copilotStatusFailure(response, copilotErrorBody(response))
	}
	raw, err := copilotRead(ctx, response, copilotMaxCatalogBytes)
	if err != nil {
		return nil, err
	}
	models, err := parseCopilotCatalog(raw, p.now())
	if err != nil {
		return nil, copilotMalformed("the Copilot catalog is invalid", err)
	}
	p.learn(models)
	infos := make([]core.ModelInfo, len(models))
	for index, model := range models {
		infos[index] = model.info
	}
	return infos, nil
}

// copilotRoute is what the latest catalog row of a model says about serving
// it: the endpoints it lists and whether it lists reasoning efforts.
type copilotRoute struct {
	endpoints       []string
	reasoningEffort bool
}

// responses reports a listed Responses endpoint, which the gateway serves
// natively.
func (r copilotRoute) responses() bool {
	return slices.ContainsFunc(r.endpoints, func(endpoint string) bool {
		return copilotResponsesEndpoint(strings.ToLower(strings.TrimSpace(endpoint)))
	})
}

// preferred is the endpoint the gateway's API adaptation sends Chat to:
// "responses" for a row that lists Responses but not Chat Completions.
func (r copilotRoute) preferred() string { return translate.PreferredEndpoint(r.endpoints) }

func (p *Copilot) route(model string) (copilotRoute, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	route, ok := p.routes[model]
	return route, ok
}

// learn keeps the routes of a listed catalog. Rows another credential's
// catalog listed stay known: a model's endpoints do not depend on the
// account that lists it.
func (p *Copilot) learn(models []copilotModel) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, model := range models {
		p.routes[model.info.ID] = copilotRoute{
			endpoints: slices.Clone(model.info.SupportedAPIs), reasoningEffort: model.reasoningEffort,
		}
	}
}
