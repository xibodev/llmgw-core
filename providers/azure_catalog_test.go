package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// azureCatalog lists the fixture resource's deployments with a key.
func azureCatalog(t *testing.T, answer func(http.ResponseWriter, *http.Request, []byte)) (*azureBackend, []core.ModelInfo, error) {
	t.Helper()
	backend, server := newAzureBackend(t, answer)
	models, err := newTestAzure(t, server).ListModels(context.Background(), azureKey("fixture-key"))
	return backend, models, err
}

func writeAzureJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", core.ContentTypeJSON)
	_ = json.NewEncoder(w).Encode(value)
}

// Ported from the gateway's TestAzureCatalogUsesDeploymentsNeverModels and
// TestAzureAuthenticatesWithApiKeyHeader: the catalog is the deployments,
// read at the pinned api-version with the api-key.
func TestAzureCatalogUsesDeploymentsNeverModels(t *testing.T) {
	t.Parallel()
	backend, models, err := azureCatalog(t, func(w http.ResponseWriter, r *http.Request, _ []byte) {
		if r.URL.Path != "/openai/deployments" {
			writeAzureJSON(w, map[string]any{"data": []map[string]any{{"id": "vendor-catalogue-entry", "status": "succeeded"}}})
			return
		}
		writeAzureJSON(w, map[string]any{"data": []map[string]any{{"id": "gpt-5.6-sol", "status": "succeeded"}}})
	})
	if err != nil || len(models) != 1 || models[0].ID != "gpt-5.6-sol" {
		t.Fatalf("models = %+v, err = %v", models, err)
	}
	calls := backend.take()
	if len(calls) != 1 || calls[0].method != http.MethodGet || calls[0].path != "/openai/deployments" ||
		calls[0].query != "api-version=2023-03-15-preview" || calls[0].header.Get("api-key") != "fixture-key" ||
		calls[0].header.Get("Authorization") != "" {
		t.Fatalf("upstream = %+v", calls)
	}
}

// Ported from the gateway's TestAzureNeverFallsBackToModelsOnDeploymentFailure,
// TestAzureRejectedCredentialReportsAnAuthenticationFailure and
// TestAzureNonAuthDeploymentsFailureStaysUndiscoverable, with the
// classification a Runtime reads.
func TestAzureCatalogFailuresKeepTheGatewayCodes(t *testing.T) {
	t.Parallel()
	for _, check := range []struct {
		status int
		code   string
		class  core.ProviderErrorClass
		want   core.ProviderErrorClassification
	}{
		{http.StatusUnauthorized, CatalogCodeAuthenticationFailed, core.ProviderErrorAuth, core.ProviderErrorClassification{StatusCode: http.StatusUnauthorized}},
		{http.StatusForbidden, CatalogCodeAuthenticationFailed, core.ProviderErrorForbidden, core.ProviderErrorClassification{StatusCode: http.StatusForbidden}},
		{http.StatusNotFound, CatalogCodeNotDiscoverable, core.ProviderErrorUpstream, core.ProviderErrorClassification{StatusCode: http.StatusNotFound}},
		{http.StatusInternalServerError, CatalogCodeNotDiscoverable, core.ProviderErrorUpstream, core.ProviderErrorClassification{
			StatusCode: http.StatusInternalServerError, Retryable: true, FailoverEligible: true, CircuitFailure: true, RetryAfter: 2 * time.Second,
		}},
	} {
		t.Run(fmt.Sprint(check.status), func(t *testing.T) {
			t.Parallel()
			backend, models, err := azureCatalog(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
				if check.status == http.StatusInternalServerError {
					w.Header().Set("Retry-After", "2")
				}
				w.WriteHeader(check.status)
			})
			assertAzureFailure(t, err, check.class, check.want)
			if catalog := catalogCode(t, err); models != nil || catalog.Code != check.code || catalog.Status != check.status {
				t.Fatalf("catalog = %#v, models = %+v", catalog, models)
			}
			if calls := backend.take(); len(calls) != 1 || calls[0].path != "/openai/deployments" {
				t.Fatalf("the catalog fell back to another route: %+v", calls)
			}
		})
	}
}

// Ported from the gateway's TestAzureCatalogListsOnlyChatCallableDeployments
// and TestAzureCatalogExcludesUnrecognisedDeployments, with the rows the
// gateway stores.
func TestAzureCatalogListsOnlyChatCallableDeployments(t *testing.T) {
	t.Parallel()
	discoveredAt := time.Date(2026, time.September, 21, 12, 0, 0, 0, time.UTC)
	_, server := newAzureBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
		writeAzureJSON(w, map[string]any{"object": "list", "data": []any{
			map[string]any{"id": "chat-main", "model": "gpt-4o", "status": "succeeded", "object": "deployment"},
			map[string]any{"id": "gpt-4o-mini", "model": "gpt-4o-mini", "status": "succeeded"},
			map[string]any{"id": "o3-reasoner"},
			map[string]any{"id": "vectors", "model": "text-embedding-3-large", "status": "succeeded"},
			map[string]any{"id": "pictures", "model": "dall-e-3", "status": "succeeded"},
			map[string]any{"id": "listen", "model": "whisper", "status": "succeeded"},
			map[string]any{"id": "speak", "model": "gpt-4o-mini-tts", "status": "succeeded"},
			map[string]any{"id": "legacy", "model": "gpt-35-turbo-instruct", "status": "succeeded"},
			map[string]any{"id": "mystery", "model": "some-unreleased-family-1", "status": "succeeded"},
			map[string]any{"id": "pending", "model": "gpt-4o", "status": "creating"},
			map[string]any{"model": "gpt-4o", "status": "succeeded"},
			"not a row",
		}})
	})
	provider, err := NewAzureOpenAI(AzureOpenAIConfig{BaseURL: server.URL, CatalogClient: server.Client(), Now: func() time.Time { return discoveredAt }})
	if err != nil {
		t.Fatal(err)
	}
	models, err := provider.ListModels(context.Background(), azureKey("fixture-key"))
	if err != nil {
		t.Fatal(err)
	}
	row := func(id, display string) core.ModelInfo {
		model := core.ModelInfo{
			ID: id, Object: "model", OwnedBy: "azure-openai", Vendor: "azure-openai", DisplayName: display,
			LegacyCapabilities: map[string]any{"chat": true}, SupportedAPIs: []string{"/v1/chat/completions", "/v1/messages"},
		}
		model.Capabilities = core.AdaptModelCapabilities(model.LegacyCapabilities, model.SupportedAPIs, discoveredAt, time.Time{})
		return model
	}
	if want := []core.ModelInfo{row("chat-main", "gpt-4o"), row("gpt-4o-mini", ""), row("o3-reasoner", "")}; !reflect.DeepEqual(models, want) {
		t.Fatalf("models = %+v", models)
	}
	if capabilities := models[0].Capabilities; capabilities.Operations.Chat != core.SupportSupported ||
		capabilities.Surfaces.ChatCompletions != core.SupportSupported || capabilities.Surfaces.Messages != core.SupportSupported {
		t.Fatalf("capabilities = %+v", capabilities)
	}
}

// Ported from the gateway's base URL tests: a portal origin, /openai and
// /openai/v1 all normalize to the inference endpoint, and anything else is
// refused when the provider is built, not when a request is made.
func TestAzureBaseURLIsAResourceEndpoint(t *testing.T) {
	t.Parallel()
	for _, configured := range []string{
		"https://resource.endpoint.invalid",
		"https://resource.endpoint.invalid/",
		" https://resource.endpoint.invalid/openai ",
		"https://resource.endpoint.invalid/openai/v1",
		"https://resource.endpoint.invalid/openai/v1/",
	} {
		provider, err := NewAzureOpenAI(AzureOpenAIConfig{BaseURL: configured})
		if err != nil || provider.baseURL != "https://resource.endpoint.invalid/openai/v1" || provider.origin != "https://resource.endpoint.invalid" {
			t.Fatalf("NewAzureOpenAI(%q) = %+v, %v", configured, provider, err)
		}
	}
	for _, configured := range []string{
		"",
		"resource.endpoint.invalid/openai/v1",
		"https://resource.endpoint.invalid/openai/deployments/some-deployment",
		"https://resource.endpoint.invalid/openai/v1?api-version=2024-01-01",
		"https://resource.endpoint.invalid/openai/v1#fragment",
		"ftp://resource.endpoint.invalid",
		"file://resource.endpoint.invalid/openai/v1",
		"ws://resource.endpoint.invalid/openai/v1",
		"https://proxy.endpoint.invalid/proxy/openai/v1",
		"https://proxy.endpoint.invalid/proxy/openai",
		"https://proxy.endpoint.invalid/azure/resource/openai/v1/",
	} {
		provider, err := NewAzureOpenAI(AzureOpenAIConfig{BaseURL: configured})
		if provider != nil {
			t.Fatalf("NewAzureOpenAI(%q) accepted a base URL that is not a resource endpoint", configured)
		}
		assertAzureFailure(t, err, core.ProviderErrorConfiguration, core.ProviderErrorClassification{FailoverEligible: true})
	}
}

// Ported from the gateway's pagination tests: has_more continues after the
// cursor, a nextLink is never followed, and a walk that cannot continue
// fails rather than truncating the catalog.
func TestAzureCatalogPagination(t *testing.T) {
	t.Parallel()
	page := func(id string, extra map[string]any) map[string]any {
		body := map[string]any{"data": []map[string]any{{"id": id, "status": "succeeded", "model": "gpt-4o"}}}
		for key, value := range extra {
			body[key] = value
		}
		return body
	}
	t.Run("has_more follows the last row's id", func(t *testing.T) {
		t.Parallel()
		backend, models, err := azureCatalog(t, func(w http.ResponseWriter, r *http.Request, _ []byte) {
			if r.URL.Query().Get("after") == "gpt-page-one" {
				writeAzureJSON(w, page("gpt-page-two", nil))
				return
			}
			writeAzureJSON(w, page("gpt-page-one", map[string]any{"has_more": true}))
		})
		calls := backend.take()
		if err != nil || len(models) != 2 || models[0].ID != "gpt-page-one" || models[1].ID != "gpt-page-two" || len(calls) != 2 ||
			calls[1].query != "api-version=2023-03-15-preview&after=gpt-page-one" {
			t.Fatalf("models = %+v, err = %v, calls = %+v", models, err, calls)
		}
	})
	t.Run("a nextLink is ignored", func(t *testing.T) {
		t.Parallel()
		backend, models, err := azureCatalog(t, func(w http.ResponseWriter, r *http.Request, _ []byte) {
			writeAzureJSON(w, page("gpt-page-one", map[string]any{"nextLink": "http://" + r.Host + r.URL.String() + "&page=2"}))
		})
		if calls := backend.take(); err != nil || len(models) != 1 || len(calls) != 1 {
			t.Fatalf("models = %+v, err = %v, calls = %+v", models, err, calls)
		}
	})
	for name, answer := range map[string]map[string]any{
		"has_more without a cursor": {"data": []any{}, "has_more": true},
		"a repeated cursor":         page("gpt-page-one", map[string]any{"has_more": true, "last_id": "same"}),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			_, models, err := azureCatalog(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) { writeAzureJSON(w, answer) })
			if catalog := catalogCode(t, err); models != nil || catalog.Code != CatalogCodeNotDiscoverable {
				t.Fatalf("catalog = %#v, models = %+v", catalog, models)
			}
		})
	}
	t.Run("a walk that never ends", func(t *testing.T) {
		t.Parallel()
		backend, models, err := azureCatalog(t, func(w http.ResponseWriter, r *http.Request, _ []byte) {
			writeAzureJSON(w, page("gpt-"+r.URL.Query().Get("after")+"x", map[string]any{"has_more": true}))
		})
		if catalog := catalogCode(t, err); models != nil || catalog.Code != CatalogCodeNotDiscoverable || len(backend.take()) != azureDeploymentsMaxPages {
			t.Fatalf("catalog = %#v, models = %+v", catalog, models)
		}
	})
}

// Ported from the gateway's TestAzureCatalogCursorCannotRedirectToAnotherHost:
// a cursor shaped like a foreign URL stays a query value on the resource.
func TestAzureCatalogCursorCannotRedirectToAnotherHost(t *testing.T) {
	t.Parallel()
	foreign, foreignServer := newAzureBackend(t, nil)
	backend, models, err := azureCatalog(t, func(w http.ResponseWriter, r *http.Request, _ []byte) {
		if r.URL.Query().Get("after") != "" {
			writeAzureJSON(w, map[string]any{"data": []map[string]any{{"id": "gpt-page-two", "status": "succeeded", "model": "gpt-4o"}}})
			return
		}
		writeAzureJSON(w, map[string]any{
			"data":     []map[string]any{{"id": "gpt-page-one", "status": "succeeded", "model": "gpt-4o"}},
			"last_id":  foreignServer.URL + "/openai/deployments?api-version=" + azureDeploymentsAPIVersion,
			"has_more": true,
		})
	})
	if err != nil || len(models) != 2 || len(foreign.take()) != 0 {
		t.Fatalf("models = %+v, err = %v", models, err)
	}
	if calls := backend.take(); len(calls) != 2 || calls[1].path != "/openai/deployments" {
		t.Fatalf("upstream = %+v", calls)
	}
	if _, ok := azureSameOriginPage("https://resource.invalid", "https://resource.invalid", "https://user:pass@resource.invalid/next"); ok {
		t.Fatal("a continuation with userinfo was accepted")
	}
	if next, ok := azureSameOriginPage("https://resource.invalid", "https://RESOURCE.invalid/openai/deployments", "/openai/deployments?after=x"); !ok ||
		next != "https://RESOURCE.invalid/openai/deployments?after=x" {
		t.Fatalf("a relative continuation on the resource = %q, %v", next, ok)
	}
}

// Ported from the gateway's catalog response checks for Azure: pages are
// decoded as the gateway decodes them, so what is not JSON fails, a null
// page is empty, and a page over the bound fails without its text, a later
// page included.
func TestAzureCatalogReadsPagesAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	for name, check := range map[string]struct {
		pages   []string
		code    string
		status  int
		class   core.ProviderErrorClass
		want    core.ProviderErrorClassification
		entries int
	}{
		"not json":   {pages: []string{"not json"}, code: CatalogCodeNotDiscoverable, status: http.StatusOK, class: core.ProviderErrorUpstream, want: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"a list":     {pages: []string{"[]"}, code: CatalogCodeNotDiscoverable, status: http.StatusOK, class: core.ProviderErrorUpstream, want: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"null":       {pages: []string{"null"}},
		"no data":    {pages: []string{`{"value":[{"id":"gpt-4o"}]}`}},
		"with rest":  {pages: []string{`{"data":[{"id":"gpt-4o"}]} trailing`}, entries: 1},
		"exact size": {pages: []string{`{"data":[{"id":"gpt-4o"}]}` + strings.Repeat(" ", catalogMaxResponseBytes-len(`{"data":[{"id":"gpt-4o"}]}`))}, entries: 1},
		"oversized later page": {
			pages: []string{`{"data":[{"id":"gpt-4o","model":"gpt-4o"}],"has_more":true,"last_id":"next"}`, `{"data":[],"private":"fixture-secret"}` + strings.Repeat(" ", catalogMaxResponseBytes)},
			code:  CatalogCodeNotDiscoverable, status: http.StatusOK, class: core.ProviderErrorUpstream, want: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			_, models, err := azureCatalog(t, func(w http.ResponseWriter, r *http.Request, _ []byte) {
				index := 0
				if r.URL.Query().Get("after") != "" {
					index = 1
				}
				_, _ = io.WriteString(w, check.pages[index])
			})
			if check.code == "" {
				if err != nil || models == nil || len(models) != check.entries {
					t.Fatalf("models = %+v, err = %v", models, err)
				}
				return
			}
			failure := assertAzureFailure(t, err, check.class, check.want)
			if catalog := catalogCode(t, err); catalog.Code != check.code || catalog.Status != check.status || strings.Contains(failure.Error(), "fixture-secret") {
				t.Fatalf("catalog = %#v", catalog)
			}
		})
	}
}

// A listing that gets no complete answer keeps the gateway's code but may
// repeat, unless the caller gave up.
func TestAzureCatalogTransportFailures(t *testing.T) {
	t.Parallel()
	_, server := newAzureBackend(t, nil)
	provider := newTestAzure(t, server)
	server.Close()
	_, err := provider.ListModels(context.Background(), azureKey("fixture-key"))
	assertAzureFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true})
	if catalog := catalogCode(t, err); catalog.Code != CatalogCodeNotDiscoverable {
		t.Fatalf("catalog = %#v", catalog)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = provider.ListModels(ctx, azureKey("fixture-key"))
	assertAzureFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{})
}
