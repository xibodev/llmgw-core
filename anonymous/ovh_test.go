package anonymous_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/anonymous"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
)

// ovhChatModels are the chat models of OVHcloud AI Endpoints' live catalog
// fixture, in catalog order.
func ovhChatModels() []string {
	return []string{
		"Qwen3.6-27B", "Qwen3.5-9B", "Qwen3.8-27B", "Qwen3-Coder-30B-A3B-Instruct", "Meta-Llama-3_3-70B-Instruct",
		"Mistral-Small-3.2-24B-Instruct-2506", "Qwen3.5-397B-A17B", "gpt-oss-120b", "Mistral-7B-Instruct-v0.3",
		"Qwen2.5-VL-72B-Instruct", "Mistral-Nemo-Instruct-2407", "gpt-oss-20b",
	}
}

// The automation checks OVHcloud AI Endpoints over its live catalog, read
// and probed keyless through a Runtime and OpenAICompatible, as a product
// runs it. The reviewed Qwen3.8-27B verifies the provider, and probing
// every discovered model probes and publishes the chat models only: no
// classifier is ever a chat target.
func TestAutomationVerifiesOVHWithItsReviewedChatModel(t *testing.T) {
	t.Parallel()
	catalog, err := os.ReadFile(filepath.Join("..", "providers", "testdata", "ovh_ai_endpoints", "models-2026-09-28.json"))
	if err != nil {
		t.Fatal(err)
	}
	var profile providers.AnonymousProviderProfile
	for _, candidate := range providers.AnonymousProviderProfiles() {
		if candidate.RegistryID == "ovh_ai_endpoints" {
			profile = candidate
		}
	}
	for name, test := range map[string]struct {
		probe  anonymous.ProbePolicy
		probed []string
	}{
		"verification model": {anonymous.ProbeVerificationModel, []string{"Qwen3.8-27B"}},
		"every model":        {anonymous.ProbeAllDiscovered, ovhChatModels()},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			var mu sync.Mutex
			var probed []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "" {
					t.Errorf("%s carried a key", r.URL.Path)
				}
				w.Header().Set("Content-Type", "application/json")
				switch r.URL.Path {
				case "/v1/models":
					_, _ = w.Write(catalog)
				case "/v1/chat/completions":
					var body struct {
						Model string `json:"model"`
					}
					_ = json.NewDecoder(r.Body).Decode(&body)
					mu.Lock()
					probed = append(probed, body.Model)
					mu.Unlock()
					_, _ = io.WriteString(w, `{"id":"chatcmpl-fixture","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			runtime, err := coreruntime.New(coreruntime.Options[struct{}]{
				Settings: coreruntime.NewMemorySettings(struct{}{}),
				Providers: func(struct{}, string) (core.Provider, error) {
					return providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{
						BaseURL: server.URL + "/v1", RegistryID: profile.RegistryID, Client: server.Client(), CatalogClient: server.Client(),
					})
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			discover := anonymous.CatalogFunc(func(ctx context.Context, caller core.Caller, instance string) ([]core.ModelInfo, error) {
				record, err := runtime.ListModels(ctx, caller, instance)
				return record.Evidence.Models, err
			})
			o, err := anonymous.New(anonymous.Options{
				Profiles: []providers.AnonymousProviderProfile{profile}, Catalog: discover, Invoker: runtime, Hooks: &product{}, Probe: test.probe,
			})
			if err != nil {
				t.Fatal(err)
			}
			results := o.ConnectAll(context.Background())
			if len(results) != 1 {
				t.Fatalf("results = %+v", results)
			}
			result := results[0]
			published := make([]string, 0, len(result.Targets))
			for _, target := range result.Targets {
				if target.Provider != profile.ProviderID {
					t.Fatalf("target = %+v", target)
				}
				published = append(published, target.Model)
			}
			mu.Lock()
			defer mu.Unlock()
			if result.Status != anonymous.StatusPassed || result.FailureCode != "" || result.Verified != len(test.probed) ||
				!slices.Equal(probed, test.probed) || !slices.Equal(published, test.probed) {
				t.Fatalf("probed %v, published %v, result %+v", probed, published, result)
			}
		})
	}
}
