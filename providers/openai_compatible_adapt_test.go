package providers

import (
	"context"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// With adaptation on, a model whose row lists reasoning efforts takes
// max_tokens as max_completion_tokens, as the gateway plans it from the
// catalog. Off, or for a model the catalog lacks, nothing is renamed.
func TestOpenAICompatibleRenamesMaxTokensForReasoningModels(t *testing.T) {
	t.Parallel()
	rows := openAIRows(map[string]core.ModelInfo{
		"reasoning-fixture": {SupportedAPIs: []string{"/chat/completions"}, LegacyCapabilities: map[string]any{"reasoning_effort": []string{"low", "high"}}},
		"plain-fixture":     {SupportedAPIs: []string{"/chat/completions"}, LegacyCapabilities: map[string]any{"reasoning": true}},
	})
	backend, server := newOpenAIBackend(t, answerOpenAIChat)
	adapting := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) { config.Models, config.ForceAPISupport = rows, true })
	plain := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) { config.Models = rows })
	for _, test := range []struct {
		provider *OpenAICompatible
		model    string
		body     string
		renamed  bool
	}{
		{adapting, "reasoning-fixture", `{"messages":[],"max_tokens":64,"max_completion_tokens":32}`, true},
		{adapting, "reasoning-fixture", `{"messages":[],"max_tokens":64,"force_api_support":false}`, false},
		{adapting, "plain-fixture", `{"messages":[],"max_tokens":64}`, false},
		{adapting, "unlisted", `{"messages":[],"max_tokens":64}`, false},
		{plain, "reasoning-fixture", `{"messages":[],"max_tokens":64}`, false},
		{plain, "reasoning-fixture", `{"messages":[],"max_tokens":64,"force_api_support":true}`, true},
	} {
		response, err := test.provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, test.model, test.body, nil))
		calls := backend.take()
		if err != nil || len(calls) != 1 || calls[0].path != "/v1/chat/completions" {
			t.Fatalf("%s %s: calls = %+v, err = %v", test.model, test.body, calls, err)
		}
		renamed := strings.Contains(calls[0].body, `"max_completion_tokens":64`) && !strings.Contains(calls[0].body, "max_tokens\":")
		if renamed != test.renamed || slices.Contains(lossPaths(response.Losses), "advisory renamed max_tokens") != test.renamed ||
			strings.Contains(string(response.Body), "forced_support") {
			t.Fatalf("%s %s: body = %s, losses = %v", test.model, test.body, calls[0].body, response.Losses)
		}
	}
}

// A request's force_api_support turns the configured adaptation off, and
// Chat for a Responses-only model then goes to the Chat endpoint, as the
// gateway sends it.
func TestOpenAICompatibleLetsARequestTurnAdaptationOff(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, answerOpenAIChat)
	provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) {
		config.Models, config.ForceAPISupport = openAIResponsesOnly, true
	})
	body := `{"messages":[{"role":"user","content":"hi"}],"force_api_support":false}`
	if _, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "responses-fixture", body, nil)); err != nil {
		t.Fatal(err)
	}
	want := `{"messages":[{"content":"hi","role":"user"}],"model":"responses-fixture","stream":false}`
	if calls := backend.take(); len(calls) != 1 || calls[0].path != "/v1/chat/completions" || calls[0].body != want {
		t.Fatalf("upstream = %+v", calls)
	}
}

// Ported from the gateway's TestV043OfficialOpenAIUsesNativeResponsesWithoutCatalogMetadata.
func TestOpenAICompatibleServesResponsesForTheOfficialOpenAIEntry(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, answerOpenAIResponses)
	provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) { config.RegistryID = "openai" })
	if !core.ServesNatively(provider, "future-model", core.ModelSurfaceResponses) {
		t.Fatal("the official OpenAI entry does not serve Responses natively")
	}
	response, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceResponses, "future-model", `{"input":"hi"}`, nil))
	if err != nil || string(response.Body) != openAIResponsesAnswer {
		t.Fatalf("response = %s, err = %v", response.Body, err)
	}
	if calls := backend.take(); len(calls) != 1 || calls[0].path != "/v1/responses" ||
		calls[0].body != `{"input":"hi","model":"future-model","stream":false}` || calls[0].method != http.MethodPost {
		t.Fatalf("upstream = %+v", calls)
	}
}

// Only adaptation reads the catalog row of a Chat request, so a product's
// lookup, which may load a stored catalog, is not made for every request.
func TestOpenAICompatibleLooksChatUpOnlyToAdaptIt(t *testing.T) {
	t.Parallel()
	_, server := newOpenAIBackend(t, answerOpenAIChat)
	var lookups atomic.Int32
	provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) {
		config.Models = func(model string) (core.ModelInfo, bool) {
			lookups.Add(1)
			return openAIResponsesOnly(model)
		}
	})
	chat := openAIRequest(core.ModelSurfaceChatCompletions, "chat-fixture", `{"messages":[]}`, nil)
	if _, err := provider.Invoke(context.Background(), chat); err != nil || lookups.Load() != 0 {
		t.Fatalf("lookups = %d, err = %v", lookups.Load(), err)
	}
	chat.Body = []byte(`{"messages":[],"force_api_support":true}`)
	if _, err := provider.Invoke(context.Background(), chat); err != nil || lookups.Load() != 1 {
		t.Fatalf("adapted: lookups = %d, err = %v", lookups.Load(), err)
	}
}
