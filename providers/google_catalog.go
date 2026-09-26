package providers

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// CatalogCodeConfigurationIncomplete is an instance configured too little
// to list a catalog, such as Vertex AI without a project.
const CatalogCodeConfigurationIncomplete = "configuration_incomplete"

// Vertex AI's unfiltered Model Garden can hold over 14,000 rows. The whole
// walk is bounded, duplicates and filtered rows included, as well as each
// answer: per-request timeouts and repeated-token detection do not bound
// unique tokens.
const (
	googleCatalogMaxPages  = 100
	googleCatalogMaxModels = 20000
)

// ListModels returns the models the deployment lists for the credential,
// with the capabilities the gateway derives for them.
//
// AI Studio's catalog is paged, and a model's supportedGenerationMethods
// say what it serves. Vertex AI lists publisher models only on the v1beta1
// collection route, scoped to the instance's location and never merged
// across locations, and only to an OAuth principal: with an API key alone
// its catalog is not discoverable. Its rows carry no capability field, so
// a model's family, read from its ID, says what it serves, and a family
// the classifier cannot place is left out rather than listed as a chat
// model. Only models an operator can call directly are listed. What the
// catalog omits stays unknown.
func (p *Google) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	access, err := p.access(credential)
	if err != nil {
		return nil, err
	}
	var models []core.ModelInfo
	if p.deployment == GoogleVertexAI {
		models, err = p.vertexModels(ctx, access)
	} else {
		models, err = p.aiStudioModels(ctx, access)
	}
	if err != nil {
		return nil, err
	}
	// The gateway infers a row's capabilities when it stores the catalog.
	discoveredAt := p.now()
	for index := range models {
		models[index].Capabilities = core.InferCapabilities(models[index], discoveredAt, time.Time{})
	}
	return models, nil
}

// googleCatalogRefusal reports a catalog this instance cannot list as
// configured. It says nothing about the upstream.
func googleCatalogRefusal(code, detail string) error {
	return core.NewConfigurationError(detail, &CatalogError{Code: code, Detail: detail})
}

// aiStudioModels walks AI Studio's catalog through its nextPageToken.
func (p *Google) aiStudioModels(ctx context.Context, access googleAccess) ([]core.ModelInfo, error) {
	authorization, err := p.authorization(access)
	if err != nil {
		return nil, err
	}
	if authorization.name == "" {
		return nil, googleCatalogRefusal(CatalogCodeAuthenticationFailed, "Provider API key is required for catalog access.")
	}
	models := make([]core.ModelInfo, 0)
	query := url.Values{"pageSize": {"1000"}}
	seen := map[string]bool{}
	count := 0
	for page := 0; ; page++ {
		if page >= googleCatalogMaxPages {
			return nil, catalogFailure(ctx, CatalogCodeNotDiscoverable, "Provider catalog listing exceeded the page limit.", 0, 0, nil)
		}
		decoded, err := p.discoverModels(ctx, authorization, p.baseURL+"/models?"+query.Encode(), "models")
		if err != nil {
			return nil, err
		}
		rows, _ := decoded["models"].([]any)
		count += len(rows)
		if count > googleCatalogMaxModels {
			return nil, catalogFailure(ctx, CatalogCodeNotDiscoverable, "Provider catalog listing exceeded the model limit.", 0, 0, nil)
		}
		models = append(models, parseAIStudioModels(decoded)...)
		next, _ := decoded["nextPageToken"].(string)
		if next == "" {
			return models, nil
		}
		if seen[next] {
			return nil, catalogFailure(ctx, CatalogCodeInvalidShape, "Provider catalog response repeated a page token.", http.StatusOK, 0, nil)
		}
		seen[next] = true
		query.Set("pageToken", next)
	}
}

// vertexModels lists Vertex AI's Google models. Google refuses API keys on
// ListPublisherModels by design ("API keys are not supported by this API.
// Expected OAuth2 access token."), so a key-only instance cannot have a
// catalog and says so rather than advertise one nobody measured.
func (p *Google) vertexModels(ctx context.Context, access googleAccess) ([]core.ModelInfo, error) {
	if access.project == "" {
		return nil, googleCatalogRefusal(CatalogCodeConfigurationIncomplete, "Vertex AI requires a Google Cloud project ID.")
	}
	authorization, err := p.authorization(access)
	if err != nil {
		return nil, googleCatalogTokenFailure(ctx, err)
	}
	if !authorization.bearer() {
		return nil, googleCatalogRefusal(CatalogCodeNotDiscoverable, "Vertex AI model discovery requires a service account credential; "+
			"the catalog is not discoverable with an API key alone.")
	}
	return p.vertexPublisherModels(ctx, authorization, "google")
}

// googleCatalogTokenFailure reports a token exchange that failed for a
// catalog with the gateway's codes: an exchange that may repeat is a
// transport error, and any other an authentication failure, which keeps
// the token endpoint's status.
func googleCatalogTokenFailure(ctx context.Context, err error) error {
	classification := core.ClassifyError(err)
	if classification.Retryable {
		return catalogFailure(ctx, CatalogCodeTransportError, "Provider credential refresh could not reach the token service.", classification.StatusCode, 0, err)
	}
	failure := catalogFailure(ctx, CatalogCodeAuthenticationFailed, "Provider credential refresh failed for catalog access.", classification.StatusCode, 0, err)
	failure.Class = core.ProviderErrorAuth
	return failure
}

// vertexPublisherModels lists a publisher's managed models for the
// instance's location, as measured against the live API:
//
//   - The collection route exists only on v1beta1; every v1 variant answers
//     with a generic HTML 404. Inference stays on v1, so Google speaks two
//     API versions.
//   - The project-scoped form has no list method. The project comes from the
//     OAuth token, so the route carries none.
//   - The catalog is regional and not nested: "global" carries IDs a region
//     like us-central1 lacks, and the other way round.
func (p *Google) vertexPublisherModels(ctx context.Context, authorization googleAuthorization, publisher string) ([]core.ModelInfo, error) {
	base := "https://" + vertexHost(p.location)
	if p.baseURL != "" {
		base = vertexDiscoveryBase(p.baseURL)
	}
	models := make([]core.ModelInfo, 0, 64)
	pageToken := ""
	seen := map[string]bool{}
	count := 0
	for page := 0; ; page++ {
		if page >= googleCatalogMaxPages {
			return nil, catalogFailure(ctx, CatalogCodeNotDiscoverable, "Provider catalog listing exceeded the page limit.", 0, 0, nil)
		}
		// Upstream rejects pageSize=1000; 200 is the measured working value.
		query := url.Values{"pageSize": {"200"}}
		if pageToken != "" {
			// A page token is opaque and not URL-safe: sent raw, a '+' in it
			// decodes upstream as a space, asking for a page nobody issued.
			query.Set("pageToken", pageToken)
		}
		endpoint := fmt.Sprintf("%s/v1beta1/publishers/%s/models?%s", base, publisher, query.Encode())
		decoded, err := p.discoverModels(ctx, authorization, endpoint, "publisherModels")
		if err != nil {
			return nil, err
		}
		rows, _ := decoded["publisherModels"].([]any)
		count += len(rows)
		if count > googleCatalogMaxModels {
			return nil, catalogFailure(ctx, CatalogCodeNotDiscoverable, "Provider catalog listing exceeded the model limit.", 0, 0, nil)
		}
		models = append(models, vertexManagedModels(decoded, p.location == vertexDefaultLocation)...)
		next, _ := decoded["nextPageToken"].(string)
		if next == "" {
			return models, nil
		}
		if seen[next] {
			return nil, catalogFailure(ctx, CatalogCodeInvalidShape, "Provider catalog response repeated a page token.", http.StatusOK, 0, nil)
		}
		seen[next] = true
		pageToken = next
	}
}

// vertexDiscoveryBase is the root the discovery route hangs off a
// configured base URL. That base is an inference root ending in /v1, and
// discovery exists only on v1beta1, so appending to it unstripped asked for
// <base>/v1/v1beta1/... and failed discovery on every proxied instance while
// inference worked. A base that does not end in the inference version is
// the caller's own path shape, and is left alone.
func vertexDiscoveryBase(baseURL string) string {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	return strings.TrimSuffix(base, "/"+vertexDefaultAPI)
}

// discoverModels fetches and validates one catalog page as the gateway
// does. Beyond naming their models, the rows' fields that drive filtering
// must be well formed, because a malformed value is no evidence that a
// model is unsupported. Unknown methods and actions stay legitimate.
func (p *Google) discoverModels(ctx context.Context, authorization googleAuthorization, endpoint, field string) (map[string]any, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		detail := "Provider catalog request could not be created."
		return nil, core.NewConfigurationError(detail, &CatalogError{Code: CatalogCodeTransportError, Detail: detail, Cause: err})
	}
	authorization.apply(request.Header)
	response, err := p.client.Do(request)
	if err != nil {
		return nil, catalogFailure(ctx, CatalogCodeTransportError, "Provider catalog request could not reach the upstream service.", 0, 0, err)
	}
	decoded, err := decodeCatalogResponse(ctx, response, p.now(), field, "name")
	if err != nil {
		return nil, err
	}
	invalid := func() error {
		return catalogFailure(ctx, CatalogCodeInvalidShape, "Provider catalog response contained invalid model metadata.", response.StatusCode, 0, nil)
	}
	if token, exists := decoded["nextPageToken"]; exists {
		if _, ok := token.(string); !ok {
			return nil, invalid()
		}
	}
	rows, _ := decoded[field].([]any)
	for _, raw := range rows {
		row, _ := raw.(map[string]any)
		name, _ := row["name"].(string)
		if strings.TrimSpace(name[strings.LastIndex(name, "/")+1:]) == "" || !googleCatalogRowValid(row, field) {
			return nil, invalid()
		}
	}
	return decoded, nil
}

// googleCatalogRowValid checks the field of a row that drives filtering:
// AI Studio's supportedGenerationMethods is a list of nonblank strings, and
// each native action Vertex AI's supportedActions names is a message, empty
// or not.
func googleCatalogRowValid(row map[string]any, field string) bool {
	if field == "models" {
		value, exists := row["supportedGenerationMethods"]
		if !exists {
			return true
		}
		methods, ok := value.([]any)
		if !ok {
			return false
		}
		for _, method := range methods {
			if text, ok := method.(string); !ok || strings.TrimSpace(text) == "" {
				return false
			}
		}
		return true
	}
	value, exists := row["supportedActions"]
	if !exists {
		return true
	}
	actions, ok := value.(map[string]any)
	if !ok {
		return false
	}
	for _, key := range []string{"openGenerationAiStudio", "requestAccess", "deploy", "deployGke", "multiDeployVertex"} {
		if action, exists := actions[key]; exists {
			if _, ok := action.(map[string]any); !ok {
				return false
			}
		}
	}
	return true
}

// googleModel is one catalog row, as the gateway lists it.
func googleModel(id, displayName string, capabilities map[string]any, surfaces []string) core.ModelInfo {
	return core.ModelInfo{
		ID: id, Object: "model", OwnedBy: "google", Vendor: "google", DisplayName: displayName,
		LegacyCapabilities: capabilities, SupportedAPIs: surfaces,
	}
}

// parseAIStudioModels keeps the models whose supportedGenerationMethods
// place them.
func parseAIStudioModels(decoded map[string]any) []core.ModelInfo {
	rawModels, _ := decoded["models"].([]any)
	models := make([]core.ModelInfo, 0, len(rawModels))
	for _, raw := range rawModels {
		entry, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		name, _ := entry["name"].(string)
		id := strings.TrimPrefix(name, "models/")
		if id == "" {
			continue
		}
		methods := make([]string, 0, 4)
		if raw, ok := entry["supportedGenerationMethods"].([]any); ok {
			for _, method := range raw {
				if text, ok := method.(string); ok {
					methods = append(methods, text)
				}
			}
		}
		capabilities, surfaces := googleCapabilities(id, methods)
		if len(capabilities) == 0 {
			continue
		}
		displayName, _ := entry["displayName"].(string)
		models = append(models, googleModel(id, displayName, capabilities, surfaces))
	}
	return models
}

// googleCapabilities maps Google's supportedGenerationMethods onto the
// gateway's capability vocabulary. Image models advertise the same
// generateContent method as text models, so their name identifies them.
func googleCapabilities(id string, methods []string) (map[string]any, []string) {
	capabilities := map[string]any{}
	surfaces := []string{}
	has := func(name string) bool {
		for _, method := range methods {
			if method == name {
				return true
			}
		}
		return false
	}
	lower := strings.ToLower(id)
	switch {
	case strings.Contains(lower, "veo") && has("predictLongRunning"):
		capabilities["video"] = true
		surfaces = append(surfaces, "/v1/videos/generations")
	case strings.Contains(lower, "image") && has("generateContent"):
		capabilities["image"] = true
		surfaces = append(surfaces, "/v1/images/generations")
	case has("generateContent"):
		capabilities["chat"] = true
		surfaces = append(surfaces, "/v1/chat/completions", "/v1/messages")
	case has("embedContent"):
		capabilities["embedding"] = true
	}
	return capabilities, surfaces
}

// vertexManagedModels keeps the entries an operator can call directly:
// those whose supportedActions offer openGenerationAiStudio, or
// requestAccess, which is gated but callable once granted. The rest are
// Model Garden entries that only deploy, so the operator must stand up an
// endpoint first; one measured region carried 11,841 of those against
// about 78 managed models. The global location also lists a model without
// actions; in a region, missing actions are no evidence of availability.
func vertexManagedModels(decoded map[string]any, allowEmptyActions bool) []core.ModelInfo {
	raw, _ := decoded["publisherModels"].([]any)
	models := make([]core.ModelInfo, 0, len(raw))
	for _, entry := range raw {
		fields, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		name, _ := fields["name"].(string)
		id := name[strings.LastIndex(name, "/")+1:]
		if id == "" {
			continue
		}
		rawActions, actionsPresent := fields["supportedActions"]
		actions, _ := rawActions.(map[string]any)
		_, openGeneration := actions["openGenerationAiStudio"]
		_, gated := actions["requestAccess"]
		// An ID the classifier cannot place is left out, not listed bare:
		// a row without capabilities reads as a chat model, and routing it
		// to generateContent gets the 404 the classifier exists to prevent.
		// A caller who knows the ID can still address it.
		capabilities, surfaces := vertexModelCapability(id)
		if len(capabilities) == 0 {
			continue
		}
		if !openGeneration && !gated && !(allowEmptyActions && (!actionsPresent || len(actions) == 0)) {
			continue
		}
		// The list carries no display name, so none is guessed.
		models = append(models, googleModel(id, "", capabilities, surfaces))
	}
	return models
}

// vertexModelCapability infers a discovered model's capability from its ID:
// ListPublisherModels carries no capability field, so this is a floor, not
// a measurement. Embedding is checked first, because "gemini-embedding-*"
// IDs name both families, and an embedding model routed to generateContent
// fails upstream. An ID that matches no family has no capability.
func vertexModelCapability(id string) (map[string]any, []string) {
	lower := strings.ToLower(id)
	switch {
	case strings.Contains(lower, "embed"):
		return map[string]any{"embedding": true}, nil
	case strings.Contains(lower, "veo"):
		return map[string]any{"video": true}, []string{"/v1/videos/generations"}
	case strings.Contains(lower, "image"):
		return map[string]any{"image": true}, []string{"/v1/images/generations"}
	case strings.Contains(lower, "gemini") || strings.Contains(lower, "bison") ||
		strings.Contains(lower, "chat") || strings.Contains(lower, "codey"):
		return map[string]any{"chat": true}, []string{"/v1/chat/completions", "/v1/messages"}
	}
	return nil, nil
}
