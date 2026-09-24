// Package zen implements the reusable, storage-neutral OpenCode Zen anonymous
// admission contract. It deliberately does not cache catalogs or schedule
// refreshes; callers decide when and where evidence is retained.
package zen

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

const (
	DefaultBaseURL       = "https://opencode.ai/zen/v1"
	DefaultMetadataURL   = "https://models.dev/api.json"
	DefaultProviderID    = "opencode_zen"
	AnonymousUserAgent   = "opencode/1.18.31 ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14"
	defaultResponseLimit = 10 << 20
)

// HTTPClient is implemented by *http.Client and permits deterministic tests and
// embedding-specific transports without coupling Zen to storage or scheduling.
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

type Endpoints struct {
	BaseURL     string
	MetadataURL string
}

type Config struct {
	ProviderID      string
	Endpoints       Endpoints
	HTTPClient      HTTPClient
	Now             func() time.Time
	NewID           func(prefix string) (string, error)
	CapabilityTTL   time.Duration
	MaxResponseSize int64
	UserAgent       string
}

// Client performs live discovery, request admission, and inference calls. It
// retains only the stable anonymous project identity; it stores no catalog.
type Client struct {
	providerID      string
	baseURL         string
	metadataURL     string
	httpClient      HTTPClient
	now             func() time.Time
	newID           func(string) (string, error)
	capabilityTTL   time.Duration
	maxResponseSize int64
	userAgent       string
}

type invocationIdentityKey struct{}

// InvocationIdentity is the logical OpenCode invocation, independent of any
// authentication attempt. Keeping it in context makes all retries reuse it.
type InvocationIdentity struct {
	Project, Session, Request, Client, UserAgent string
}

func NewInvocationIdentity(headers http.Header, newID func(string) (string, error)) (InvocationIdentity, error) {
	if newID == nil {
		newID = randomID
	}
	identity := InvocationIdentity{
		Project:   headers.Get("x-opencode-project"),
		Session:   headers.Get("x-opencode-session"),
		Request:   headers.Get("x-opencode-request"),
		Client:    headers.Get("x-opencode-client"),
		UserAgent: AnonymousUserAgent,
	}
	if identity.Project == "" {
		identity.Project = "global"
	}
	if identity.Client == "" {
		identity.Client = "cli"
	}
	var err error
	if identity.Session == "" {
		identity.Session, err = newID("ses")
	}
	if err == nil && identity.Request == "" {
		identity.Request, err = newID("msg")
	}
	if err != nil {
		return InvocationIdentity{}, fmt.Errorf("create Zen invocation identity: %w", err)
	}
	return identity, nil
}

func WithInvocationIdentity(ctx context.Context, identity InvocationIdentity) context.Context {
	return context.WithValue(ctx, invocationIdentityKey{}, identity)
}

func InvocationIdentityFromContext(ctx context.Context) (InvocationIdentity, bool) {
	identity, ok := ctx.Value(invocationIdentityKey{}).(InvocationIdentity)
	return identity, ok
}

func EnsureInvocationIdentity(ctx context.Context, headers http.Header) (context.Context, error) {
	if _, ok := InvocationIdentityFromContext(ctx); ok {
		return ctx, nil
	}
	identity, err := NewInvocationIdentity(headers, nil)
	if err != nil {
		return ctx, err
	}
	return WithInvocationIdentity(ctx, identity), nil
}

func ApplyInvocationHeaders(header http.Header, identity InvocationIdentity) {
	header.Set("x-opencode-project", identity.Project)
	header.Set("x-opencode-session", identity.Session)
	header.Set("x-opencode-request", identity.Request)
	header.Set("x-opencode-client", identity.Client)
	header.Set("User-Agent", identity.UserAgent)
}

func New(config Config) (*Client, error) {
	providerID := strings.TrimSpace(config.ProviderID)
	if providerID == "" {
		providerID = DefaultProviderID
	}
	baseURL, err := endpoint(config.Endpoints.BaseURL, DefaultBaseURL)
	if err != nil {
		return nil, fmt.Errorf("Zen base URL: %w", err)
	}
	metadataURL, err := endpoint(config.Endpoints.MetadataURL, DefaultMetadataURL)
	if err != nil {
		return nil, fmt.Errorf("models.dev URL: %w", err)
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	now := config.Now
	if now == nil {
		now = time.Now
	}
	newID := config.NewID
	if newID == nil {
		newID = randomID
	}
	ttl := config.CapabilityTTL
	if ttl <= 0 {
		ttl = time.Hour
	}
	limit := config.MaxResponseSize
	if limit <= 0 {
		limit = defaultResponseLimit
	}
	userAgent := strings.TrimSpace(config.UserAgent)
	if userAgent == "" {
		userAgent = AnonymousUserAgent
	}
	return &Client{
		providerID: providerID, baseURL: baseURL, metadataURL: metadataURL,
		httpClient: client, now: now, newID: newID, capabilityTTL: ttl,
		maxResponseSize: limit, userAgent: userAgent,
	}, nil
}

func endpoint(value, fallback string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		value = fallback
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", errors.New("must be an absolute URL")
	}
	return strings.TrimRight(value, "/"), nil
}

func randomID(prefix string) (string, error) {
	var raw [13]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", err
	}
	return prefix + "_" + hex.EncodeToString(raw[:]), nil
}

// AnonymousHeaders returns headers for one standalone logical invocation.
func (c *Client) AnonymousHeaders() (http.Header, error) {
	identity, err := NewInvocationIdentity(nil, c.newID)
	if err != nil {
		return nil, err
	}
	identity.Client = "cli"
	identity.UserAgent = c.userAgent
	return c.AnonymousHeadersFor(identity), nil
}

func (c *Client) AnonymousHeadersFor(identity InvocationIdentity) http.Header {
	header := http.Header{}
	header.Set("Authorization", "Bearer public")
	header.Set("Content-Type", "application/json")
	ApplyInvocationHeaders(header, identity)
	return header
}

func (c *Client) anonymousHeadersForContext(ctx context.Context) (http.Header, error) {
	if identity, ok := InvocationIdentityFromContext(ctx); ok {
		return c.AnonymousHeadersFor(identity), nil
	}
	return c.AnonymousHeaders()
}

type metadataProvider struct {
	NPM    string                   `json:"npm"`
	Models map[string]metadataModel `json:"models"`
}

type metadataModel struct {
	ID               string                     `json:"id"`
	Name             string                     `json:"name"`
	Status           string                     `json:"status"`
	Cost             map[string]json.RawMessage `json:"cost"`
	Reasoning        *bool                      `json:"reasoning"`
	ToolCall         *bool                      `json:"tool_call"`
	StructuredOutput *bool                      `json:"structured_output"`
	Modalities       struct {
		Input []string `json:"input"`
	} `json:"modalities"`
	Limit struct {
		Context int64 `json:"context"`
		Output  int64 `json:"output"`
	} `json:"limit"`
	Provider *struct {
		NPM string `json:"npm"`
	} `json:"provider"`
}

type catalogEnvelope struct {
	Data []struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	} `json:"data"`
}

// Discover returns the OpenCode-compatible public snapshot. OpenCode exposes
// snapshot models with zero input cost, then applies its ordinary status rules.
func (c *Client) Discover(ctx context.Context) (core.CatalogEvidence, error) {
	evidence, _, err := c.discoverSnapshot(ctx)
	return evidence, err
}

func (c *Client) discoverSnapshot(ctx context.Context) (core.CatalogEvidence, metadataProvider, error) {
	observedAt := c.now().UTC()
	metadataRaw, err := c.get(ctx, c.metadataURL, false)
	if err != nil {
		status := core.CatalogFailed
		if isCallerCancellation(ctx, err) {
			status = core.CatalogNotProbed
		}
		return core.CatalogEvidence{Status: status, ObservedAt: observedAt}, metadataProvider{}, err
	}

	var root map[string]json.RawMessage
	if err := json.Unmarshal(metadataRaw, &root); err != nil {
		return core.CatalogEvidence{Status: core.CatalogFailed, ObservedAt: observedAt}, metadataProvider{}, errors.New("decode models.dev catalog")
	}
	providerRaw, ok := root["opencode"]
	if !ok {
		return core.CatalogEvidence{Status: core.CatalogFailed, ObservedAt: observedAt}, metadataProvider{}, errors.New("models.dev catalog has no opencode provider")
	}
	var metadata metadataProvider
	if err := json.Unmarshal(providerRaw, &metadata); err != nil || metadata.Models == nil {
		return core.CatalogEvidence{Status: core.CatalogFailed, ObservedAt: observedAt}, metadataProvider{}, errors.New("decode models.dev opencode provider")
	}

	models := make([]core.ModelInfo, 0, len(metadata.Models))
	for key, model := range metadata.Models {
		if model.ID != key || model.ID == "" {
			continue
		}
		if !admittedStatus(model.Status) || !inputCostZero(model.Cost) {
			continue
		}
		surface, ok := nativeSurface(metadata.NPM, model)
		if !ok {
			continue
		}
		models = append(models, core.ModelInfo{
			ID: key, Object: "model", OwnedBy: c.providerID,
			Description: model.Name, Capabilities: capabilities(model, surface, observedAt, c.capabilityTTL),
		})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	status := core.CatalogDiscovered
	if len(models) == 0 {
		status = core.CatalogEmpty
	}
	return core.CatalogEvidence{Status: status, Models: models, ObservedAt: observedAt}, metadata, nil
}

// DiscoverVerified is the optional strict policy: a snapshot model must also
// appear in the live Zen catalog and have every described monetary cost at zero.
func (c *Client) DiscoverVerified(ctx context.Context) (core.CatalogEvidence, error) {
	public, metadata, err := c.discoverSnapshot(ctx)
	if err != nil {
		return public, err
	}
	raw, err := c.get(ctx, c.baseURL+"/models", true)
	if err != nil {
		return core.CatalogEvidence{Status: core.CatalogFailed, ObservedAt: public.ObservedAt}, err
	}
	var catalog catalogEnvelope
	if json.Unmarshal(raw, &catalog) != nil || catalog.Data == nil {
		return core.CatalogEvidence{Status: core.CatalogFailed, ObservedAt: public.ObservedAt}, errors.New("decode Zen catalog")
	}
	live := make(map[string]bool, len(catalog.Data))
	for _, row := range catalog.Data {
		live[strings.TrimSpace(row.ID)] = true
	}
	verified := public.Models[:0]
	for _, model := range public.Models {
		metadataModel := metadata.Models[model.ID]
		if live[model.ID] && verifiedStatus(metadataModel.Status) && exactZeroCost(metadataModel.Cost) {
			verified = append(verified, model)
		}
	}
	public.Models = verified
	if len(verified) == 0 {
		public.Status = core.CatalogEmpty
	}
	return public, nil
}

func verifiedStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "", "active", "beta":
		return true
	default:
		return false
	}
}

type catalogEnvelopeRow struct {
	object  string
	created int64
}

func (c *Client) get(ctx context.Context, endpoint string, anonymous bool) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, errors.New("create catalog request")
	}
	request.Header.Set("Accept", "application/json")
	if anonymous {
		headers, headerErr := c.AnonymousHeaders()
		if headerErr != nil {
			return nil, headerErr
		}
		request.Header = headers
		request.Header.Set("Accept", "application/json")
	}
	response, err := c.httpClient.Do(request)
	if err != nil {
		if cause := context.Cause(ctx); cause != nil {
			return nil, cause
		}
		return nil, errors.New("catalog request failed")
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("catalog request returned HTTP %d", response.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, c.maxResponseSize+1))
	if err != nil {
		if cause := context.Cause(ctx); cause != nil {
			return nil, cause
		}
		return nil, errors.New("read catalog response")
	}
	if int64(len(data)) > c.maxResponseSize {
		return nil, errors.New("catalog response exceeds size limit")
	}
	return data, nil
}

func admittedStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "", "active", "beta":
		return true
	default:
		return false
	}
}

// monetaryCostKey reports the price fields that must be exactly zero.
func monetaryCostKey(key string) bool {
	switch key {
	case "input", "output", "reasoning", "cache_read", "cache_write", "input_audio", "output_audio":
		return true
	}
	return false
}

// exactZeroCost rejects absent required prices, strings, unknown fields,
// non-zero values, and malformed nested tiers rather than guessing that a
// partially described model is free.
func exactZeroCost(cost map[string]json.RawMessage) bool {
	if cost == nil || !rawZero(cost["input"]) || !rawZero(cost["output"]) {
		return false
	}
	for key, raw := range cost {
		if monetaryCostKey(key) {
			if !rawZero(raw) {
				return false
			}
			continue
		}
		switch key {
		case "context_over_200k":
			var nested map[string]json.RawMessage
			if json.Unmarshal(raw, &nested) != nil || !exactZeroCost(nested) {
				return false
			}
		case "tiers":
			var tiers []map[string]json.RawMessage
			if json.Unmarshal(raw, &tiers) != nil || len(tiers) == 0 {
				return false
			}
			for _, tier := range tiers {
				delete(tier, "tier")
				if !exactZeroCost(tier) {
					return false
				}
			}
		default:
			return false
		}
	}
	return true
}

func rawZero(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if decoder.Decode(&value) != nil {
		return false
	}
	number, ok := value.(json.Number)
	if !ok {
		return false
	}
	rational, ok := new(big.Rat).SetString(number.String())
	return ok && rational.Sign() == 0
}

func inputCostZero(cost map[string]json.RawMessage) bool {
	raw, ok := cost["input"]
	return !ok || rawZero(raw)
}

func nativeSurface(providerNPM string, model metadataModel) (core.ModelSurface, bool) {
	npm := strings.TrimSpace(providerNPM)
	if model.Provider != nil && strings.TrimSpace(model.Provider.NPM) != "" {
		npm = strings.TrimSpace(model.Provider.NPM)
	}
	switch npm {
	case "@ai-sdk/openai":
		return core.ModelSurfaceResponses, true
	case "@ai-sdk/openai-compatible":
		return core.ModelSurfaceChatCompletions, true
	default:
		return "", false
	}
}

func capabilities(model metadataModel, surface core.ModelSurface, observedAt time.Time, ttl time.Duration) *core.ModelCapabilities {
	expiresAt := observedAt.Add(ttl)
	chat, responses := core.SupportUnsupported, core.SupportUnsupported
	if surface == core.ModelSurfaceResponses {
		responses = core.SupportSupported
	} else {
		chat = core.SupportSupported
	}
	return &core.ModelCapabilities{
		SchemaVersion: core.ModelCapabilitiesSchemaVersion,
		Operations: core.ModelOperationCapabilities{
			Chat: core.SupportSupported, Embeddings: core.SupportUnsupported, Image: core.SupportUnsupported,
			AudioIn: core.SupportUnsupported, AudioOut: core.SupportUnsupported, Video: core.SupportUnsupported,
			TokenCount: core.SupportUnsupported,
		},
		Surfaces: core.ModelSurfaceCapabilities{ChatCompletions: chat, Responses: responses, Messages: core.SupportUnsupported},
		Inputs:   core.ModelInputCapabilities{Text: modalitySupport(model.Modalities.Input, "text"), Image: modalitySupport(model.Modalities.Input, "image")},
		Tools:    reportedBoolSupport(model.ToolCall), Reasoning: reportedBoolSupport(model.Reasoning), StructuredOutput: reportedBoolSupport(model.StructuredOutput),
		Streaming: core.SupportSupported, StatefulResponses: core.SupportUnsupported,
		Limits:     core.ModelCapabilityLimits{ContextTokens: positive(model.Limit.Context), MaxOutputTokens: positive(model.Limit.Output)},
		Provenance: core.ModelCapabilityProvenance{Source: core.ModelCapabilitySourceModelsDev, Confidence: core.ModelCapabilityConfidenceHigh},
		Freshness:  core.ModelCapabilityFreshness{DiscoveredAt: &observedAt, ExpiresAt: &expiresAt},
	}
}

func boolSupport(value bool) core.Support {
	if value {
		return core.SupportSupported
	}
	return core.SupportUnsupported
}

func reportedBoolSupport(value *bool) core.Support {
	if value == nil {
		return core.SupportUnknown
	}
	return boolSupport(*value)
}

func modalitySupport(modalities []string, wanted string) core.Support {
	if modalities == nil {
		return core.SupportUnknown
	}
	for _, modality := range modalities {
		if modality == wanted {
			return core.SupportSupported
		}
	}
	return core.SupportUnsupported
}

func positive(value int64) *int64 {
	if value <= 0 {
		return nil
	}
	return &value
}

// NativeSurface reads the catalog-derived native surface without model-name
// heuristics. Catalog rows not produced by this package remain unknown.
func NativeSurface(model core.ModelInfo) (core.ModelSurface, bool) {
	if model.Capabilities == nil {
		return "", false
	}
	chat := model.Capabilities.Surfaces.ChatCompletions == core.SupportSupported
	responses := model.Capabilities.Surfaces.Responses == core.SupportSupported
	if chat == responses {
		return "", false
	}
	if responses {
		return core.ModelSurfaceResponses, true
	}
	return core.ModelSurfaceChatCompletions, true
}

// CompleteNative sends an already surface-correct payload. It does not perform
// wire translation, making lossless Chat/Responses routing an explicit caller
// responsibility.
func (c *Client) CompleteNative(ctx context.Context, model string, surface core.ModelSurface, payload map[string]any) (map[string]any, error) {
	path := ""
	switch surface {
	case core.ModelSurfaceChatCompletions:
		path = "/chat/completions"
	case core.ModelSurfaceResponses:
		path = "/responses"
	default:
		return nil, fmt.Errorf("unsupported Zen surface %q", surface)
	}
	body := cloneMap(payload)
	switch surface {
	case core.ModelSurfaceChatCompletions:
		body = AdmitChat(messageMaps(body["messages"]), body)
	case core.ModelSurfaceResponses:
		body = AdmitResponses(body)
	}
	body["model"] = model
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, errors.New("encode Zen request")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(encoded))
	if err != nil {
		return nil, errors.New("create Zen request")
	}
	headers, err := c.anonymousHeadersForContext(ctx)
	if err != nil {
		return nil, err
	}
	request.Header = headers
	response, err := c.httpClient.Do(request)
	if err != nil {
		if cause := context.Cause(ctx); cause != nil {
			return nil, core.NewProviderOperationError("Zen completion", 0, "", cause)
		}
		return nil, core.NewProviderOperationError("Zen completion", 0, "", err)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, core.NewProviderOperationError("Zen completion", response.StatusCode, response.Header.Get("Retry-After"), nil)
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, c.maxResponseSize+1))
	if err != nil {
		if cause := context.Cause(ctx); cause != nil {
			return nil, core.NewProviderOperationError("Zen completion response read", 0, "", cause)
		}
		return nil, core.NewProviderOperationError("Zen completion response read", response.StatusCode, "", err)
	}
	if int64(len(data)) > c.maxResponseSize {
		return nil, core.NewProviderOperationError("Zen completion response size", response.StatusCode, "", errors.New("response exceeds size limit"))
	}
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, core.NewProviderOperationError("Zen completion response decode", response.StatusCode, "", err)
	}
	return result, nil
}

func messageMaps(raw any) []map[string]any {
	switch messages := raw.(type) {
	case []map[string]any:
		return messages
	case []any:
		out := make([]map[string]any, 0, len(messages))
		for _, rawMessage := range messages {
			if message, ok := rawMessage.(map[string]any); ok {
				out = append(out, message)
			}
		}
		return out
	default:
		return nil
	}
}

// Connect implements core.ProviderConnector. It discovers the admitted catalog,
// probes the first exact model on its metadata-derived native surface, and then
// applies the caller's standard exact-target publication policy.
func (c *Client) Connect(ctx context.Context, request core.ProviderConnectRequest) (core.ProviderConnectResult, error) {
	result := core.ProviderConnectResult{Connection: request.Connection}
	if err := request.Connection.Validate(); err != nil {
		return result, err
	}
	if strings.TrimSpace(request.Connection.ProviderID) != c.providerID {
		return result, fmt.Errorf("Zen connection provider id %q does not match client provider id %q", request.Connection.ProviderID, c.providerID)
	}
	if request.Connection.Kind != core.ProviderConnectionAnonymous || request.Connection.AuthKind != core.ProviderAuthAnonymous {
		return result, errors.New("Zen anonymous access requires an anonymous connection")
	}
	if _, err := core.PublishExactTargets(c.providerID, core.CatalogEvidence{}, nil, request.PublicationPolicy); err != nil {
		return result, err
	}

	catalog, err := c.DiscoverVerified(ctx)
	result.Catalog = catalog
	if err != nil {
		if isCallerCancellation(ctx, err) {
			result.Health = unknownHealth(catalog.ObservedAt)
			return result, err
		}
		result.Health = healthFromError(err, catalog.ObservedAt)
		return result, err
	}
	result.Health = core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: http.StatusOK, ObservedAt: catalog.ObservedAt})
	if len(catalog.Models) > 0 {
		model := catalog.Models[0]
		target := core.Target{Provider: c.providerID, Model: model.ID}
		surface, _ := NativeSurface(model)
		started := c.now().UTC()
		probeErr := c.probe(ctx, target.Model, surface)
		finished := c.now().UTC()
		probe := core.CompletionProbeEvidence{Target: target, Status: core.CompletionVerified, ObservedAt: finished, Latency: finished.Sub(started)}
		if probeErr != nil {
			probe.Status = core.CompletionFailed
		}
		result.Probes = []core.CompletionProbeEvidence{probe}
		if isCallerCancellation(ctx, probeErr) {
			result.Probes[0].Status = core.CompletionNotProbed
			result.Health = unknownHealth(finished)
			return result, probeErr
		}
		if probeErr != nil {
			result.Health = healthFromError(probeErr, finished)
		}
	}
	result.Targets, err = core.PublishExactTargets(c.providerID, catalog, result.Probes, request.PublicationPolicy)
	return result, err
}

func (c *Client) probe(ctx context.Context, model string, surface core.ModelSurface) error {
	payload := map[string]any{"stream": true, "max_tokens": 16}
	switch surface {
	case core.ModelSurfaceChatCompletions:
		payload["messages"] = []any{map[string]any{"role": "user", "content": "Reply with: ok"}}
	case core.ModelSurfaceResponses:
		delete(payload, "max_tokens")
		payload["max_output_tokens"] = 16
		payload["input"] = "Reply with: ok"
	default:
		return errors.New("Zen probe model has no admitted native surface")
	}
	body := cloneMap(payload)
	if surface == core.ModelSurfaceChatCompletions {
		body = AdmitChat(messageMaps(body["messages"]), body)
	} else {
		body = AdmitResponses(body)
	}
	body["model"] = model
	encoded, err := json.Marshal(body)
	if err != nil {
		return errors.New("encode Zen probe")
	}
	path := "/chat/completions"
	if surface == core.ModelSurfaceResponses {
		path = "/responses"
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(encoded))
	if err != nil {
		return errors.New("create Zen probe")
	}
	headers, err := c.anonymousHeadersForContext(ctx)
	if err != nil {
		return err
	}
	headers.Set("Accept", "text/event-stream")
	request.Header = headers
	response, err := c.httpClient.Do(request)
	if err != nil {
		if cause := context.Cause(ctx); cause != nil {
			return core.NewProviderOperationError("Zen probe", 0, "", cause)
		}
		return core.NewProviderOperationError("Zen probe", 0, "", err)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return core.NewProviderOperationError("Zen probe", response.StatusCode, response.Header.Get("Retry-After"), nil)
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, c.maxResponseSize+1))
	if err != nil {
		if cause := context.Cause(ctx); cause != nil {
			return core.NewProviderOperationError("Zen probe response read", 0, "", cause)
		}
		return core.NewProviderOperationError("Zen probe response read", response.StatusCode, "", err)
	}
	if int64(len(data)) > c.maxResponseSize {
		return core.NewProviderOperationError("Zen probe response size", response.StatusCode, "", errors.New("response exceeds size limit"))
	}
	if err := validateProbeSSE(surface, data); err != nil {
		return core.NewProviderOperationError("Zen probe response decode", response.StatusCode, "", err)
	}
	return nil
}

func validateProbeSSE(surface core.ModelSurface, data []byte) error {
	normalized := strings.ReplaceAll(strings.ReplaceAll(string(data), "\r\n", "\n"), "\r", "\n")
	if normalized != "" && !strings.HasSuffix(normalized, "\n\n") {
		return errors.New("Zen probe stream ended with a truncated SSE event")
	}
	lines := strings.Split(normalized, "\n")
	var fields []string
	terminal := false
	chatFinished := false
	events := 0
	consume := func() error {
		if len(fields) == 0 {
			return nil
		}
		value := strings.Join(fields, "\n")
		fields = fields[:0]
		events++
		if terminal {
			return errors.New("Zen probe stream contains an event after its terminal event")
		}
		if value == "[DONE]" {
			if surface != core.ModelSurfaceChatCompletions || !chatFinished {
				return errors.New("Zen probe stream ended without a successful terminal event")
			}
			terminal = true
			return nil
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(value), &event); err != nil {
			return fmt.Errorf("malformed Zen probe SSE event: %w", err)
		}
		if event["error"] != nil {
			return errors.New("Zen probe stream reported an error")
		}
		switch surface {
		case core.ModelSurfaceChatCompletions:
			finished, err := validateChatProbeEvent(event)
			if err != nil {
				return err
			}
			if chatFinished {
				return errors.New("Zen probe stream contains an event after its finish chunk")
			}
			chatFinished = finished
		case core.ModelSurfaceResponses:
			eventType, ok := event["type"].(string)
			if !ok || eventType == "" {
				return errors.New("Zen probe response event type is missing")
			}
			switch eventType {
			case "response.failed", "response.incomplete", "error":
				return errors.New("Zen probe response did not complete successfully")
			case "response.completed":
				response, ok := event["response"].(map[string]any)
				status, _ := response["status"].(string)
				if !ok || status != "completed" {
					return errors.New("Zen probe completion event is invalid")
				}
				terminal = true
			}
		default:
			return errors.New("Zen probe model has no admitted native surface")
		}
		return nil
	}
	for _, line := range lines {
		if line == "" {
			if err := consume(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, value, found := strings.Cut(line, ":")
		if !found {
			value = ""
		}
		if strings.HasPrefix(value, " ") {
			value = value[1:]
		}
		if field == "data" {
			fields = append(fields, value)
		}
	}
	if events == 0 {
		return errors.New("Zen probe stream contains no data events")
	}
	if !terminal {
		return errors.New("Zen probe stream ended without a successful terminal event")
	}
	return nil
}

func validateChatProbeEvent(event map[string]any) (bool, error) {
	choices, ok := event["choices"].([]any)
	if !ok || len(choices) == 0 {
		return false, errors.New("Zen probe chat event has no choices")
	}
	finished := false
	for _, raw := range choices {
		choice, ok := raw.(map[string]any)
		if !ok {
			return false, errors.New("Zen probe chat choice is invalid")
		}
		if reason := choice["finish_reason"]; reason != nil {
			value, ok := reason.(string)
			if !ok || strings.TrimSpace(value) == "" {
				return false, errors.New("Zen probe chat finish reason is invalid")
			}
			finished = true
		}
	}
	return finished, nil
}

func isCallerCancellation(ctx context.Context, err error) bool {
	cause := context.Cause(ctx)
	return cause != nil && err != nil && errors.Is(err, cause)
}

func unknownHealth(observedAt time.Time) core.ProviderHealthEvidence {
	return core.ProviderHealthEvidence{Status: core.ProviderHealthUnknown, ErrorClass: core.ProviderErrorNone, ObservedAt: observedAt}
}

func healthFromError(err error, observedAt time.Time) core.ProviderHealthEvidence {
	failure := core.ProviderFailure{ObservedAt: observedAt, Err: err}
	var operationError *core.ProviderOperationError
	if errors.As(err, &operationError) {
		failure = operationError.Failure
		failure.ObservedAt = observedAt
	}
	return core.ClassifyProviderFailure(failure)
}

func cloneMap(input map[string]any) map[string]any {
	output := make(map[string]any, len(input)+1)
	for key, value := range input {
		output[key] = value
	}
	return output
}
