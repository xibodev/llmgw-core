package core_test

import (
	"slices"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	"github.com/xibodev/llmgw-core/translation"
)

// realVerticals returns core's Codex and Antigravity as a Runtime binds
// them, behind a translation.Adapter.
func realVerticals(t *testing.T) (codex, antigravity core.Provider) {
	t.Helper()
	c, err := providers.NewCodex(providers.CodexConfig{Instructions: "fixture", ClientVersion: "fixture-version"})
	if err != nil {
		t.Fatal(err)
	}
	a, err := providers.NewAntigravity(providers.AntigravityConfig{})
	if err != nil {
		t.Fatal(err)
	}
	return translation.Adapter{Provider: c}, translation.Adapter{Provider: a}
}

func listing(provider core.Provider, row core.ModelInfo, refreshedAt time.Time) (native, emulated, unknown []string) {
	return core.ListingSurfaces(provider, nil, core.TransportEvidence{Row: row, RefreshedAt: refreshedAt}, parityNow)
}

// Gateway: TestModelTransportSurfacesUsesConcreteDeclarationAndTrustedFreshPlan
// and TestModelListTransportMetadataUsesExecutionDeclarationsAndPlanner.
func TestParityListingUsesDeclarationsAndTheTrustedFreshPlan(t *testing.T) {
	codex, antigravity := realVerticals(t)
	all := []string{core.ListingChatCompletions, core.ListingResponses, core.ListingMessages}
	for name, check := range map[string]struct {
		provider         core.Provider
		surface          string
		native, emulated []string
	}{
		"OpenAI Chat declaration":        {openAICompatible(), "/v1/chat/completions", []string{core.ListingChatCompletions}, all[1:]},
		"AI Studio Chat adaptation":      {chatAdapter(), "/v1/chat/completions", nil, all},
		"Vertex Chat adaptation":         {chatAdapter(), "/v1/chat/completions", nil, all},
		"Antigravity Chat adaptation":    {antigravity, "/v1/chat/completions", nil, all},
		"Codex Responses declaration":    {codex, "/v1/responses", []string{core.ListingResponses}, []string{core.ListingChatCompletions, core.ListingMessages}},
		"Anthropic Messages declaration": {anthropicNative(), "/v1/messages", []string{core.ListingMessages}, all[:2]},
	} {
		t.Run(name, func(t *testing.T) {
			row := core.ModelInfo{ID: "fixture-model", LegacyCapabilities: map[string]any{"chat": true}, SupportedAPIs: []string{check.surface}, Capabilities: trusted(check.surface)}
			native, emulated, unknown := listing(check.provider, row, parityNow)
			if !slices.Equal(native, check.native) && !(len(native) == 0 && len(check.native) == 0) {
				t.Fatalf("native=%v, want %v", native, check.native)
			}
			if !slices.Equal(emulated, check.emulated) || len(native)+len(emulated)+len(unknown) != 3 {
				t.Fatalf("native=%v emulated=%v unknown=%v, want emulated %v", native, emulated, unknown, check.emulated)
			}
		})
	}
	row := core.ModelInfo{ID: "chat", LegacyCapabilities: map[string]any{"chat": true}, SupportedAPIs: []string{"/v1/chat/completions"}, Capabilities: trusted("/v1/chat/completions")}
	if native, _, _ := listing(openAICompatible(), row, parityNow.Add(-core.DefaultTransportFreshness)); !slices.Equal(native, []string{core.ListingChatCompletions}) {
		t.Fatalf("a stale projection lost the concrete native declaration: %v", native)
	}
}

// Gateway: TestModelTransportSurfacesKeepsPlannerRejectsUnknown and
// TestModelTransportSurfacesUsesInferredSurfaceArgument.
func TestParityListingKeepsRejectsUnknownAndReadsTheRowsSurfaces(t *testing.T) {
	native, emulated, unknown := listing(nil, core.ModelInfo{ID: "unknown", LegacyCapabilities: map[string]any{"chat": true}}, time.Time{})
	if len(native) != 0 || len(emulated) != 0 || len(unknown) != 3 {
		t.Fatalf("native=%v emulated=%v unknown=%v", native, emulated, unknown)
	}
	if native, emulated, unknown := listing(openAICompatible(), core.ModelInfo{ID: "embedding"}, parityNow); native != nil || emulated != nil || unknown != nil {
		t.Fatal("a row that is not chat was classified")
	}
	capabilities := &core.ModelCapabilities{SchemaVersion: core.ModelCapabilitiesSchemaVersion}
	capabilities.Operations.Chat, capabilities.Surfaces.ChatCompletions = core.SupportSupported, core.SupportSupported
	capabilities.Provenance = core.ModelCapabilityProvenance{Source: core.ModelCapabilitySourceUpstreamReported, Confidence: core.ModelCapabilityConfidenceHigh}
	row := core.ModelInfo{ID: "chat", Capabilities: capabilities, LegacyCapabilities: map[string]any{"chat": true}, SupportedAPIs: []string{"/v1/chat/completions"}}
	if native, _, _ := listing(openAICompatible(), row, parityNow); !slices.Equal(native, []string{core.ListingChatCompletions}) {
		t.Fatalf("the presented surface was not used for interface planning: %v", native)
	}
}

// The native_surfaces and emulated_surfaces codex-models.golden pins, with
// each row as its provider lists it.
func TestParityCodexModelsGoldenSurfaceLists(t *testing.T) {
	codex, _ := realVerticals(t)
	codexRow := core.ModelInfo{ID: "gpt-fixture", SupportedAPIs: []string{"/responses"}}
	codexRow.Capabilities = core.InferCapabilities(codexRow, parityNow, time.Time{})
	codexRow.Capabilities.Provenance = core.ModelCapabilityProvenance{Source: core.ModelCapabilitySourceUpstreamReported, Confidence: core.ModelCapabilityConfidenceHigh}
	codexRow.Capabilities.Surfaces.ChatCompletions = core.SupportUnsupported
	for name, check := range map[string]struct {
		provider         core.Provider
		row              core.ModelInfo
		native, emulated []string
	}{
		"codex/gpt-fixture":       {codex, codexRow, []string{"/v1/responses"}, []string{"/v1/chat/completions", "/v1/messages"}},
		"fixture-a/model-a":       {openAICompatible(), core.ModelInfo{ID: "model-a", SupportedAPIs: []string{"/chat/completions"}}, []string{"/v1/chat/completions"}, []string{"/v1/responses", "/v1/messages"}},
		"fixture/responses-model": {openAICompatible(), core.ModelInfo{ID: "responses-model", SupportedAPIs: []string{"/chat/completions", "/responses"}}, []string{"/v1/chat/completions", "/v1/responses"}, []string{"/v1/messages"}},
		"native/claude-fixture":   {anthropicNative(), core.ModelInfo{ID: "claude-fixture", SupportedAPIs: []string{"/v1/messages"}}, []string{"/v1/messages"}, []string{"/v1/chat/completions", "/v1/responses"}},
	} {
		native, emulated, unknown := listing(check.provider, check.row, parityNow)
		if !slices.Equal(native, check.native) || !slices.Equal(emulated, check.emulated) || len(unknown) != 0 {
			t.Fatalf("%s: native=%v emulated=%v unknown=%v", name, native, emulated, unknown)
		}
	}
}
