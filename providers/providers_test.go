package providers_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

func TestRegistryIntegrityAndManifest(t *testing.T) {
	entries := providers.ProviderRegistry()
	if len(entries) != 24 {
		t.Fatalf("expected 24 registry entries, got %d", len(entries))
	}

	seen := make(map[string]bool)
	for _, entry := range entries {
		if seen[entry.ID] {
			t.Fatalf("duplicate provider id: %s", entry.ID)
		}
		seen[entry.ID] = true
	}

	// Verify lookup by id and alias
	copilot, ok := providers.RegistryProvider("github_copilot")
	if !ok || copilot.Label != "GitHub Copilot" {
		t.Fatalf("failed to resolve github_copilot: %+v", copilot)
	}

	byAlias, ok := providers.RegistryProvider("copilot")
	if !ok || byAlias.ID != "github_copilot" {
		t.Fatalf("failed to resolve by alias 'copilot': %+v", byAlias)
	}

	canonical := providers.CanonicalRegistryID("zen")
	if canonical != "opencode_zen" {
		t.Fatalf("expected canonical id 'opencode_zen', got %s", canonical)
	}
}

func TestAnonymousProfiles(t *testing.T) {
	profiles := providers.AnonymousProviderProfiles()
	if len(profiles) != 5 {
		t.Fatalf("expected 5 anonymous provider profiles, got %d", len(profiles))
	}

	profileMap := make(map[string]providers.AnonymousProviderProfile)
	for _, p := range profiles {
		profileMap[p.RegistryID] = p
	}

	expected := []string{"kilo_code", "llm7", "opencode_zen", "ovh_ai_endpoints", "pollinations"}
	for _, id := range expected {
		if _, ok := profileMap[id]; !ok {
			t.Fatalf("missing anonymous profile %s", id)
		}
	}

	// Test probe model selection
	models := []core.ModelInfo{
		{ID: "other-model"},
		{ID: "openai-fast"},
	}
	probe := providers.AnonymousVerificationModel("pollinations", models)
	if probe != "openai-fast" {
		t.Fatalf("expected openai-fast for pollinations probe, got %s", probe)
	}
}

func TestErrorClassification(t *testing.T) {
	if !providers.IsThrottle(errors.New("rate limit exceeded")) {
		t.Fatal("expected throttle detection from rate limit string")
	}
	if !providers.IsThrottle(&providers.InvocationError{Status: 429}) {
		t.Fatal("expected throttle detection from 429 status")
	}

	retryableErr := &providers.InvocationError{Status: 503}
	if !providers.InvocationRetryable(retryableErr) {
		t.Fatal("expected 503 to be retryable")
	}
	if !providers.InvocationFailoverEligible(retryableErr) {
		t.Fatal("expected 503 to be failover eligible")
	}

	unrecoverableErr := &providers.InvocationError{Status: 401}
	if providers.InvocationRetryable(unrecoverableErr) {
		t.Fatal("401 should not be retryable")
	}
}

func TestMockAutoConnectAnonymousProviders(t *testing.T) {
	// Mock server that answers /models and /chat/completions
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{
					{
						"id":     "ling-3.0-flash-fin-free",
						"object": "model",
					},
				},
			})
		case "/chat/completions":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "chatcmpl-mock",
				"object": "chat.completion",
				"model":  "ling-3.0-flash-fin-free",
				"choices": []any{
					map[string]any{
						"index": 0,
						"message": map[string]any{
							"role":    "assistant",
							"content": "ok",
						},
						"finish_reason": "stop",
					},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	// Discover models directly with mock server
	profile := providers.AnonymousProviderProfile{
		RegistryID:         "opencode_zen",
		ProviderID:         "mock-zen",
		RuntimeType:        "openai_compatible",
		BaseURL:            server.URL,
		VerificationModels: []string{"ling-3.0-flash-fin-free"},
	}

	models, err := providers.DiscoverAnonymousModels(context.Background(), profile, server.Client())
	if err != nil {
		t.Fatalf("unexpected discovery error: %v", err)
	}
	if len(models) != 1 || models[0].ID != "ling-3.0-flash-fin-free" {
		t.Fatalf("unexpected models: %+v", models)
	}

	// Verify probe completion via OpenAIProvider
	p := providers.NewOpenAIProvider("mock-zen", server.URL, "none", server.Client())
	resp, err := p.Complete(context.Background(), "ling-3.0-flash-fin-free", map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected probe error: %v", err)
	}
	if resp["id"] != "chatcmpl-mock" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}
