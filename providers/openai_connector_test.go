package providers_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

func TestAnonymousOpenAICompatibleAdapterConnects(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			json.NewEncoder(w).Encode(map[string]any{"data": []any{
				map[string]any{"id": "model-a"}, map[string]any{"id": "model-b"},
			}})
		case "/chat/completions":
			var request map[string]any
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatal(err)
			}
			if request["model"] != "model-a" || request["stream"] != false {
				t.Fatalf("completion request=%+v", request)
			}
			json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "ok"}}}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	orchestrator := core.NewProviderOrchestrator()
	if err := orchestrator.Register("local-compatible", providers.NewAnonymousOpenAICompatibleAdapter(server.URL, server.Client())); err != nil {
		t.Fatal(err)
	}
	result, err := orchestrator.Connect(context.Background(), core.ProviderConnectRequest{
		Connection: core.ProviderConnection{
			ProviderID: "local-compatible", Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous,
		},
		PublicationPolicy: core.PublishVerifiedTargets,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Targets) != 1 || result.Targets[0].Model != "model-a" {
		t.Fatalf("targets=%+v", result.Targets)
	}
}

func TestAnonymousOpenAICompatibleAdapterPreservesRateLimitEvidence(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "model-a"}}})
			return
		}
		w.Header().Set("Retry-After", "12")
		http.Error(w, "limited", http.StatusTooManyRequests)
	}))
	defer server.Close()

	orchestrator := core.NewProviderOrchestrator()
	if err := orchestrator.Register("limited-compatible", providers.NewAnonymousOpenAICompatibleAdapter(server.URL, server.Client())); err != nil {
		t.Fatal(err)
	}
	result, err := orchestrator.Connect(context.Background(), core.ProviderConnectRequest{
		Connection: core.ProviderConnection{
			ProviderID: "limited-compatible", Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous,
		},
		PublicationPolicy: core.PublishVerifiedTargets,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Health.ErrorClass != core.ProviderErrorRateLimited || result.Health.RetryAfter != 12*time.Second {
		t.Fatalf("health=%+v", result.Health)
	}
	if len(result.Targets) != 0 || len(result.Probes) != 1 || result.Probes[0].Status != core.CompletionFailed {
		t.Fatalf("result=%+v", result)
	}
}

func TestAnonymousOpenAICompatibleAdapterClassifiesDiscoveryFailures(t *testing.T) {
	t.Run("unauthorized", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
		}))
		defer server.Close()

		result, err := connectAnonymousAdapter(t, server.URL, server.Client())
		if err == nil {
			t.Fatal("unauthorized discovery succeeded")
		}
		if result.Catalog.Status != core.CatalogFailed || result.Health.ErrorClass != core.ProviderErrorAuth || !result.Health.Permanent {
			t.Fatalf("result=%+v", result)
		}
	})

	t.Run("transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		client := server.Client()
		baseURL := server.URL
		server.Close()

		result, err := connectAnonymousAdapter(t, baseURL, client)
		if err == nil {
			t.Fatal("unreachable discovery succeeded")
		}
		if result.Catalog.Status != core.CatalogFailed || result.Health.ErrorClass != core.ProviderErrorTransport || !result.Health.Retryable {
			t.Fatalf("result=%+v", result)
		}
	})
}

func TestAnonymousOpenAICompatibleAdapterRejectsAuthenticatedConnection(t *testing.T) {
	orchestrator := core.NewProviderOrchestrator()
	if err := orchestrator.Register("compatible", providers.NewAnonymousOpenAICompatibleAdapter("http://127.0.0.1", nil)); err != nil {
		t.Fatal(err)
	}
	result, err := orchestrator.Connect(context.Background(), core.ProviderConnectRequest{
		Connection:        core.ProviderConnection{ProviderID: "compatible", Kind: core.ProviderConnectionSystem, AuthKind: core.ProviderAuthAPIKey},
		PublicationPolicy: core.PublishVerifiedTargets,
	})
	if err == nil {
		t.Fatal("authenticated connection accepted by anonymous adapter")
	}
	if result.Catalog.Status != core.CatalogNotProbed {
		t.Fatalf("catalog=%+v", result.Catalog)
	}
}

func connectAnonymousAdapter(t *testing.T, baseURL string, client *http.Client) (core.ProviderConnectResult, error) {
	t.Helper()
	orchestrator := core.NewProviderOrchestrator()
	if err := orchestrator.Register("compatible", providers.NewAnonymousOpenAICompatibleAdapter(baseURL, client)); err != nil {
		t.Fatal(err)
	}
	return orchestrator.Connect(context.Background(), core.ProviderConnectRequest{
		Connection: core.ProviderConnection{
			ProviderID: "compatible", Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous,
		},
		PublicationPolicy: core.PublishVerifiedTargets,
	})
}
