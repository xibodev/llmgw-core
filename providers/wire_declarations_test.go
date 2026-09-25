package providers

import (
	"testing"

	copilotauth "github.com/xibodev/llm-provider-auth/copilot"
	core "github.com/xibodev/llmgw-core"
)

// declared is what a vertical preserves, per model and surface, for a
// request with a key and for one without.
type declared struct {
	model           string
	surface         core.ModelSurface
	keyed, keyless  bool
	credentialBlind bool
}

func assertDeclarations(t *testing.T, provider core.Provider, cases []declared) {
	t.Helper()
	keyed := &core.Credential{APIKey: "fixture-key"}
	for _, check := range cases {
		for credential, want := range map[*core.Credential]bool{keyed: check.keyed, nil: check.keyless, {APIKey: "free"}: check.keyless} {
			if got := core.PreservesWireFor(provider, credential, check.model, check.surface); got != want {
				t.Errorf("%s %s with %v: preserves = %v, want %v", check.model, check.surface, credential, got, want)
			}
		}
		if got := core.PreservesWire(provider, check.model, check.surface); got != check.credentialBlind {
			t.Errorf("%s %s without a credential to go by: preserves = %v, want %v", check.model, check.surface, got, check.credentialBlind)
		}
	}
}

// The gateway labels Codex Responses native and its converted Chat
// translated (codex-chat-completions.golden).
func TestCodexPreservesOnlyResponses(t *testing.T) {
	t.Parallel()
	for _, responsesOnly := range []bool{false, true} {
		codex, err := NewCodex(CodexConfig{Instructions: "fixture", ClientVersion: "fixture/1", ResponsesOnly: responsesOnly})
		if err != nil {
			t.Fatal(err)
		}
		assertDeclarations(t, codex, []declared{
			{"gpt-fixture", core.ModelSurfaceResponses, true, true, true},
			{"gpt-fixture", core.ModelSurfaceChatCompletions, false, false, false},
			{"gpt-fixture", core.ModelSurfaceMessages, false, false, false},
		})
	}
}

// The gateway declares nothing for Antigravity, which maps Chat to Gemini.
func TestAntigravityPreservesNothing(t *testing.T) {
	t.Parallel()
	antigravity, err := NewAntigravity(AntigravityConfig{})
	if err != nil {
		t.Fatal(err)
	}
	assertDeclarations(t, antigravity, []declared{{"gemini-fixture", core.ModelSurfaceChatCompletions, false, false, false}})
}

// Gateway: TestKeyedZenNativeSurfaceUsesPersistedCatalog and
// TestAnonymousZenSurfacesAreNotWireNative.
func TestZenPreservesTheOneNativeSurfaceOnlyWithAKey(t *testing.T) {
	t.Parallel()
	rows := map[string][]string{
		"constellation-free": {"/responses"}, "muse-spark-chat": {"/chat/completions"},
		"both-fixture": {"/chat/completions", "/responses"},
	}
	zen, err := NewZen(ZenConfig{Models: func(model string) (core.ModelInfo, bool) {
		surfaces, ok := rows[model]
		return core.ModelInfo{ID: model, SupportedAPIs: surfaces}, ok
	}})
	if err != nil {
		t.Fatal(err)
	}
	chat, responses := core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses
	assertDeclarations(t, zen, []declared{
		{"constellation-free", responses, true, false, false},
		{"constellation-free", chat, false, false, false},
		{"muse-spark-chat", chat, true, false, false},
		{"both-fixture", chat, false, false, false},
		{"both-fixture", responses, false, false, false},
		{"muse-spark-fixture", responses, true, false, false},
		{"muse-spark-fixture", chat, false, false, false},
		{"cold-fixture", chat, false, false, false},
	})
}

// The gateway's OpenAI transport declares Chat Completions and Responses
// for Copilot. Copilot serves Chat over Responses for a Responses-only
// model by default, which preserves nothing.
func TestCopilotPreservesChatAndListedResponses(t *testing.T) {
	t.Parallel()
	for _, disabled := range []bool{false, true} {
		copilot, err := NewCopilot(CopilotConfig{
			Auth:                copilotauth.New(copilotauth.Config{AllowProxy: true}),
			EditorPluginVersion: "plugin/1", UserAgent: "agent/1", DisableAdaptation: disabled,
		})
		if err != nil {
			t.Fatal(err)
		}
		copilot.learn([]copilotModel{
			{info: core.ModelInfo{ID: "chat-only", SupportedAPIs: []string{"/chat/completions"}}},
			{info: core.ModelInfo{ID: "responses-only", SupportedAPIs: []string{"/responses"}}},
			{info: core.ModelInfo{ID: "both", SupportedAPIs: []string{"/chat/completions", "/responses"}}},
		})
		chat, responses := core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses
		assertDeclarations(t, copilot, []declared{
			{"chat-only", chat, true, true, true},
			{"chat-only", responses, false, false, false},
			{"responses-only", chat, disabled, disabled, disabled},
			{"responses-only", responses, true, true, true},
			{"both", chat, true, true, true},
			{"both", responses, true, true, true},
			{"unlisted", chat, true, true, true},
		})
	}
}
