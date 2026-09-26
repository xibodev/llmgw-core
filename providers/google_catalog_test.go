package providers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

func googleModelIDs(models []core.ModelInfo) []string {
	ids := make([]string, 0, len(models))
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	return ids
}

// googleCatalogCode returns a catalog failure's code, detail and status.
func googleCatalogCode(t *testing.T, err error) (string, string, int) {
	t.Helper()
	var catalog *CatalogError
	if !errors.As(err, &catalog) {
		t.Fatalf("error = %v, want a catalog error", err)
	}
	return catalog.Code, catalog.Detail, catalog.Status
}

// Ported from the gateway's TestCapabilitiesComeFromGoogleMetadata.
func TestGoogleCapabilitiesComeFromGoogleMetadata(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		id      string
		methods []string
		want    string
	}{
		{"veo-3.1-fast-generate-preview", []string{"predictLongRunning"}, "video"},
		{"gemini-3.1-flash-image", []string{"generateContent"}, "image"},
		{"gemini-3.5-flash", []string{"generateContent", "countTokens"}, "chat"},
		{"text-embedding-004", []string{"embedContent"}, "embedding"},
	} {
		capabilities, endpoints := googleCapabilities(testCase.id, testCase.methods)
		if _, ok := capabilities[testCase.want]; !ok {
			t.Fatalf("%s capabilities = %+v, want %s", testCase.id, capabilities, testCase.want)
		}
		if testCase.want != "embedding" && len(endpoints) == 0 {
			t.Fatalf("%s declared no endpoints", testCase.id)
		}
	}
}

// Ported from the gateway's TestAIStudioCatalogSkipsModelsWithNoUsableCapability,
// with the rows the gateway lists and the capabilities it infers for them.
func TestGoogleAIStudioCatalogSkipsModelsWithNoUsableCapability(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, `{"models":[
          {"name":"models/gemini-3.5-flash","displayName":"Gemini 3.5 Flash","supportedGenerationMethods":["generateContent"]},
          {"name":"models/veo-3.1-fast-generate-preview","supportedGenerationMethods":["predictLongRunning"]},
          {"name":"models/text-embedding-004","supportedGenerationMethods":["embedContent"]},
          {"name":"models/legacy-thing","supportedGenerationMethods":["generateMessage"]}
        ]}`))
	discovered := time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC)
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base, Now: func() time.Time { return discovered }})
	models, err := provider.ListModels(context.Background(), googleKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	if ids := googleModelIDs(models); !slices.Equal(ids, []string{"gemini-3.5-flash", "veo-3.1-fast-generate-preview", "text-embedding-004"}) {
		t.Fatalf("models = %v, want the unusable one skipped", ids)
	}
	chat := models[0]
	if chat.Object != "model" || chat.OwnedBy != "google" || chat.Vendor != "google" || chat.DisplayName != "Gemini 3.5 Flash" ||
		!reflect.DeepEqual(chat.LegacyCapabilities, map[string]any{"chat": true}) ||
		!slices.Equal(chat.SupportedAPIs, []string{"/v1/chat/completions", "/v1/messages"}) {
		t.Fatalf("chat row = %+v", chat)
	}
	// What the gateway's catalog infers when it stores the row.
	if want := core.AdaptModelCapabilities(map[string]any{"chat": true}, chat.SupportedAPIs, discovered, time.Time{}); !reflect.DeepEqual(chat.Capabilities, want) {
		t.Fatalf("capabilities = %+v, want %+v", chat.Capabilities, want)
	}
	if models[1].Capabilities.Operations.Video != core.SupportSupported || models[2].Capabilities.Operations.Embeddings != core.SupportSupported ||
		models[2].Capabilities.Operations.Chat != core.SupportUnknown || models[0].Capabilities.Streaming != core.SupportUnknown {
		t.Fatalf("capabilities = %+v / %+v", models[1].Capabilities, models[2].Capabilities)
	}
	if calls := fake.take(); len(calls) != 1 || calls[0].method != http.MethodGet || calls[0].path != "/models" || calls[0].query != "pageSize=1000" ||
		calls[0].apiKey != "k" || calls[0].requestType != "" {
		t.Fatalf("upstream = %+v", calls)
	}
}

// Ported from the gateway's TestAIStudioCatalogReportsAuthenticationFailure.
func TestGoogleAIStudioCatalogReportsAuthenticationFailure(t *testing.T) {
	t.Parallel()
	_, base := newGoogleFake(t, googleAnswer(http.StatusForbidden, `{"error":{"status":"PERMISSION_DENIED","message":"denied"}}`))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	_, err := provider.ListModels(context.Background(), googleKey("bad-key"))
	if code, detail, status := googleCatalogCode(t, err); code != CatalogCodeAuthenticationFailed || status != http.StatusForbidden ||
		!strings.Contains(detail, "rejected") {
		t.Fatalf("catalog failure=(%q,%q,%d)", code, detail, status)
	}
	// Without a credential there is nothing to ask with.
	_, err = provider.ListModels(context.Background(), nil)
	if code, detail, status := googleCatalogCode(t, err); code != CatalogCodeAuthenticationFailed || status != 0 ||
		detail != "Provider API key is required for catalog access." || core.ClassifyError(err).CircuitFailure {
		t.Fatalf("catalog failure=(%q,%q,%d)", code, detail, status)
	}
}

// Ported from the gateway's TestVertexDetailedCatalogRequiresProject.
func TestGoogleVertexCatalogRequiresProject(t *testing.T) {
	t.Parallel()
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI})
	_, err := provider.ListModels(context.Background(), googleKey("key"))
	var failure *core.ProviderError
	if code, detail, status := googleCatalogCode(t, err); code != CatalogCodeConfigurationIncomplete || status != 0 ||
		!strings.Contains(detail, "project") || !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration {
		t.Fatalf("catalog failure=(%q,%q,%d)", code, detail, status)
	}
}

// Ported from the gateway's TestVertexWithAPIKeyReportsCatalogUndiscoverable:
// Google refuses key auth on ListPublisherModels by design, so the catalog
// is reported undiscoverable rather than invented.
func TestGoogleVertexWithAPIKeyReportsCatalogUndiscoverable(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, `{"publisherModels":[]}`))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "test-project"})
	_, err := provider.ListModels(context.Background(), googleKey("test-api-key"))
	if code, detail, _ := googleCatalogCode(t, err); code != CatalogCodeNotDiscoverable || !strings.Contains(err.Error(), "not discoverable") ||
		!strings.Contains(detail, "service account") || core.ClassifyError(err).CircuitFailure {
		t.Fatalf("error %v does not report the catalog as undiscoverable", err)
	}
	if calls := fake.take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
}

// googleDiscovery lists a Vertex AI catalog through a fake answering with
// the publisher models body, with the OAuth bearer discovery requires.
func googleDiscovery(t *testing.T, suffix, location string, body map[string]any) ([]core.ModelInfo, []googleCall, error) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, string(raw)))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + suffix, Project: "test-project", Location: location})
	models, err := provider.ListModels(context.Background(), googleBearer("test-oauth-token"))
	return models, fake.take(), err
}

func googlePublisherModel(name string, actions map[string]any) map[string]any {
	row := map[string]any{"name": name}
	if actions != nil {
		row["supportedActions"] = actions
	}
	return row
}

var googleOpenGeneration = map[string]any{"openGenerationAiStudio": map[string]any{}}

// Ported from the gateway's TestVertexDiscoveryUsesV1Beta1CollectionRoute,
// TestVertexDiscoveryDoesNotDoubleTheVersionSegmentOnACustomBase and
// TestVertexDiscoveryKeepsABaseWithoutTheInferenceVersion. The v1
// collection route is not served, and the project-scoped path has no list
// method; the project comes from the token.
func TestGoogleVertexDiscoveryUsesTheV1Beta1CollectionRoute(t *testing.T) {
	t.Parallel()
	for suffix, path := range map[string]string{
		"":              "/v1beta1/publishers/google/models",
		"/v1":           "/v1beta1/publishers/google/models",
		"/v1/":          "/v1beta1/publishers/google/models",
		"/vertex-proxy": "/vertex-proxy/v1beta1/publishers/google/models",
	} {
		models, calls, err := googleDiscovery(t, suffix, "global", map[string]any{"publisherModels": []any{
			googlePublisherModel("publishers/google/models/gemini-2.5-flash", googleOpenGeneration),
		}})
		if err != nil {
			t.Fatalf("base %q: %v", suffix, err)
		}
		if len(calls) != 1 || calls[0].path != path || calls[0].query != "pageSize=200" || calls[0].authorization != "Bearer test-oauth-token" ||
			calls[0].apiKey != "" || calls[0].requestType != "" {
			t.Fatalf("base %q: upstream = %+v, want %s", suffix, calls, path)
		}
		if ids := googleModelIDs(models); !slices.Equal(ids, []string{"gemini-2.5-flash"}) || models[0].DisplayName != "" ||
			!slices.Equal(models[0].SupportedAPIs, []string{"/v1/chat/completions", "/v1/messages"}) {
			t.Fatalf("models = %+v, want the bare model id", models)
		}
	}
}

// Ported from the gateway's TestVertexDiscoveryKeepsOnlyManagedModels and
// TestVertexRegionalDiscoveryDoesNotInferEmptyActions. Self-deploy Model
// Garden entries need an endpoint first, and there are thousands of them.
func TestGoogleVertexDiscoveryKeepsOnlyManagedModels(t *testing.T) {
	t.Parallel()
	rows := []any{
		googlePublisherModel("publishers/google/models/gemini-managed-a", googleOpenGeneration),
		googlePublisherModel("publishers/google/models/gemini-gated-b", map[string]any{"requestAccess": map[string]any{}}),
		googlePublisherModel("publishers/google/models/gemini-global-empty-actions", map[string]any{}),
		googlePublisherModel("publishers/google/models/gemini-missing-actions", nil),
		googlePublisherModel("publishers/hf-someone/models/gemini-deployable-c", map[string]any{"deploy": map[string]any{}, "deployGke": map[string]any{}}),
	}
	models, _, err := googleDiscovery(t, "", "global", map[string]any{"publisherModels": rows})
	if err != nil {
		t.Fatal(err)
	}
	if ids := googleModelIDs(models); !slices.Equal(ids, []string{"gemini-managed-a", "gemini-gated-b", "gemini-global-empty-actions", "gemini-missing-actions"}) {
		t.Fatalf("models = %v", ids)
	}
	// In a region, missing actions are no availability evidence.
	models, _, err = googleDiscovery(t, "", "us-central1", map[string]any{"publisherModels": rows})
	if err != nil {
		t.Fatal(err)
	}
	if ids := googleModelIDs(models); !slices.Equal(ids, []string{"gemini-managed-a", "gemini-gated-b"}) {
		t.Fatalf("regional models = %v", ids)
	}
}

// Ported from the gateway's TestVertexDiscoveryOmitsUnclassifiableModels and
// TestVertexDiscoveryDoesNotAdvertiseEmbeddingModelsAsChat. A row without a
// capability reads as a chat model, so an unknown family is left out, and
// an embedding model is never listed on a chat surface.
func TestGoogleVertexDiscoveryClassifiesByFamily(t *testing.T) {
	t.Parallel()
	models, _, err := googleDiscovery(t, "", "global", map[string]any{"publisherModels": []any{
		googlePublisherModel("publishers/google/models/gemini-2.5-flash", googleOpenGeneration),
		googlePublisherModel("publishers/google/models/unknown-family-001", googleOpenGeneration),
		googlePublisherModel("publishers/google/models/text-embedding-005", googleOpenGeneration),
		googlePublisherModel("publishers/google/models/gemini-embedding-001", googleOpenGeneration),
		googlePublisherModel("publishers/google/models/imagen-4.0-generate-001", googleOpenGeneration),
		googlePublisherModel("publishers/google/models/veo-3.1-lite-generate-001", googleOpenGeneration),
	}})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"gemini-2.5-flash": "chat", "text-embedding-005": "embedding", "gemini-embedding-001": "embedding",
		"imagen-4.0-generate-001": "image", "veo-3.1-lite-generate-001": "video",
	}
	if len(models) != len(want) {
		t.Fatalf("models = %v", googleModelIDs(models))
	}
	for _, model := range models {
		if !reflect.DeepEqual(model.LegacyCapabilities, map[string]any{want[model.ID]: true}) {
			t.Fatalf("%s capabilities = %+v", model.ID, model.LegacyCapabilities)
		}
		if want[model.ID] == "embedding" && (model.SupportedAPIs != nil || model.Capabilities.Operations.Chat != core.SupportUnknown ||
			model.Capabilities.Operations.Embeddings != core.SupportSupported) {
			t.Fatalf("%s advertised as chat: %+v", model.ID, model)
		}
	}
}

// Ported from the gateway's TestVertexCatalogIsOnlyWhatUpstreamListed: an
// upstream that lists nothing yields nothing, with no floor of IDs.
func TestGoogleVertexCatalogIsOnlyWhatUpstreamListed(t *testing.T) {
	t.Parallel()
	models, calls, err := googleDiscovery(t, "", "global", map[string]any{"publisherModels": []any{}})
	if err != nil || len(models) != 0 || models == nil || len(calls) != 1 {
		t.Fatalf("models = %+v, err = %v, upstream = %+v", models, err, calls)
	}
}

// Ported from the gateway's TestVertexDiscoveryFollowsPagination, for both
// catalogs: pages follow nextPageToken, and the opaque token is
// query-escaped, because a '+' sent raw decodes upstream as a space.
func TestGoogleCatalogsFollowPagination(t *testing.T) {
	t.Parallel()
	const nextToken = "second+page/token=="
	for _, deployment := range []GoogleDeployment{GoogleAIStudio, GoogleVertexAI} {
		field, first, second := "models", "models/gemini-page-one", "models/gemini-page-two"
		if deployment == GoogleVertexAI {
			field, first, second = "publisherModels", "publishers/google/models/gemini-page-one", "publishers/google/models/gemini-page-two"
		}
		page := func(name string, next string) string {
			body := map[string]any{field: []any{map[string]any{
				"name": name, "supportedGenerationMethods": []any{"generateContent"}, "supportedActions": googleOpenGeneration,
			}}}
			if next != "" {
				body["nextPageToken"] = next
			}
			raw, _ := json.Marshal(body)
			return string(raw)
		}
		fake, base := newGoogleFake(t, func(r *http.Request) (int, string) {
			if r.URL.Query().Get("pageToken") == "" {
				return http.StatusOK, page(first, nextToken)
			}
			return http.StatusOK, page(second, "")
		})
		provider := newGoogleTest(t, GoogleConfig{Deployment: deployment, BaseURL: base, Project: "test-project"})
		models, err := provider.ListModels(context.Background(), googleBearer("test-oauth-token"))
		if err != nil {
			t.Fatal(err)
		}
		if ids := googleModelIDs(models); !slices.Equal(ids, []string{"gemini-page-one", "gemini-page-two"}) {
			t.Fatalf("%s models = %v, want both pages", deployment, ids)
		}
		size := map[GoogleDeployment]string{GoogleAIStudio: "1000", GoogleVertexAI: "200"}[deployment]
		calls := fake.take()
		if len(calls) != 2 || calls[0].query != "pageSize="+size || calls[1].query != "pageSize="+size+"&pageToken=second%2Bpage%2Ftoken%3D%3D" {
			t.Fatalf("%s upstream = %+v", deployment, calls)
		}
	}
}

// A page is validated whole before any row is used, as in the gateway, so a
// malformed page never replaces a usable catalog with a partial one.
func TestGoogleCatalogRefusesMalformedPages(t *testing.T) {
	t.Parallel()
	for name, testCase := range map[string]struct {
		deployment GoogleDeployment
		body, code string
	}{
		"blank model id":        {GoogleAIStudio, `{"models":[{"name":"models/ok","supportedGenerationMethods":["generateContent"]},{"name":"models/ "}]}`, CatalogCodeInvalidShape},
		"methods not a list":    {GoogleAIStudio, `{"models":[{"name":"models/ok","supportedGenerationMethods":"generateContent"}]}`, CatalogCodeInvalidShape},
		"blank method":          {GoogleAIStudio, `{"models":[{"name":"models/ok","supportedGenerationMethods":[" "]}]}`, CatalogCodeInvalidShape},
		"token not a string":    {GoogleAIStudio, `{"models":[],"nextPageToken":7}`, CatalogCodeInvalidShape},
		"repeated token":        {GoogleAIStudio, `{"models":[],"nextPageToken":"again"}`, CatalogCodeInvalidShape},
		"actions not an object": {GoogleVertexAI, `{"publisherModels":[{"name":"publishers/google/models/gemini-a","supportedActions":[]}]}`, CatalogCodeInvalidShape},
		"action not a message":  {GoogleVertexAI, `{"publisherModels":[{"name":"publishers/google/models/gemini-a","supportedActions":{"deploy":true}}]}`, CatalogCodeInvalidShape},
		"no model array":        {GoogleVertexAI, `{"models":[]}`, CatalogCodeInvalidShape},
		"not json":              {GoogleVertexAI, `<html>`, CatalogCodeInvalidJSON},
	} {
		_, base := newGoogleFake(t, googleAnswer(http.StatusOK, testCase.body))
		provider := newGoogleTest(t, GoogleConfig{Deployment: testCase.deployment, BaseURL: base, Project: "p"})
		models, err := provider.ListModels(context.Background(), googleBearer("t"))
		if code, _, status := googleCatalogCode(t, err); models != nil || code != testCase.code || status != http.StatusOK {
			t.Errorf("%s: models=%v code=%q status=%d", name, models, code, status)
		}
	}
	// A catalog request that gets no answer may repeat.
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, Client: &http.Client{Transport: googleUnreachable{}}})
	_, err := provider.ListModels(context.Background(), googleKey("k"))
	if code, _, _ := googleCatalogCode(t, err); code != CatalogCodeTransportError || !core.ClassifyError(err).Retryable {
		t.Fatalf("code = %q, classification = %+v", code, core.ClassifyError(err))
	}
}
