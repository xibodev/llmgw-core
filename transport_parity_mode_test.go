package core_test

import (
	"testing"
	"time"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// Gateway: TestPlanTargetTransportFreshnessDoesNotUpgradeCapabilityEvidence.
func TestParityFreshnessNeverUpgradesEvidence(t *testing.T) {
	capabilities := core.AdaptModelCapabilities(map[string]any{"chat": true}, []string{"/v1/responses"}, time.Time{}, time.Time{})
	row := core.ModelInfo{ID: "model", SupportedAPIs: []string{"/v1/responses"}, Capabilities: capabilities}
	plan := planTarget(row, parityNow, nativeResponses, "/v1/responses", true, core.TransportRequirementTransparent, false, translate.Report{})
	if plan.Disposition != core.TransportReject || plan.Reason != core.TransportRejectNativeUnconfirmed {
		t.Fatalf("medium-confidence evidence certified transparent: %+v", plan)
	}
	if capabilities.Provenance.Source != core.ModelCapabilitySourceInferred || capabilities.Provenance.Confidence != core.ModelCapabilityConfidenceMedium ||
		capabilities.Freshness.ExpiresAt != nil {
		t.Fatalf("the row's evidence was mutated: %+v", capabilities)
	}
	evidence := core.TransportEvidence{Row: row, RefreshedAt: parityNow.Add(time.Minute), FreshFor: 10 * time.Minute}.Capabilities()
	if evidence.Freshness.DiscoveredAt == nil || !evidence.Freshness.DiscoveredAt.Equal(parityNow.Add(time.Minute)) ||
		evidence.Freshness.ExpiresAt == nil || !evidence.Freshness.ExpiresAt.Equal(parityNow.Add(11*time.Minute)) {
		t.Fatalf("freshness = %+v", evidence.Freshness)
	}
}

// Gateway: TestPlanTargetTransportRejectsRoutesZenAndLossyAdaptation, and
// the route half of TestTransparentDecisionContractRejectsRoutesAndNonNativeSurfaces.
func TestParityPlanRejectsRoutesZenAndLossyAdaptation(t *testing.T) {
	row := core.ModelInfo{ID: "model", LegacyCapabilities: map[string]any{"chat": true}, SupportedAPIs: []string{"/v1/chat/completions"}}
	nativeChat := []core.TransportInterface{{Surface: core.ModelSurfaceChatCompletions, Native: core.SupportSupported}}
	adaptedChat := []core.TransportInterface{{Surface: core.ModelSurfaceChatCompletions, Native: core.SupportUnsupported}}
	transparent, anything := core.TransportRequirementTransparent, core.TransportRequirementAny
	if plan := planTarget(row, parityNow, nativeChat, "/v1/chat/completions", false, transparent, false, translate.Report{}); plan.Reason != core.TransportRejectExactTargetRequired {
		t.Fatalf("route plan = %+v", plan)
	}
	if plan := planTarget(row, parityNow, adaptedChat, "/v1/chat/completions", true, transparent, false, translate.Report{}); plan.Disposition != core.TransportReject || plan.Reason != core.TransportRejectNativeUnconfirmed {
		t.Fatalf("Zen plan = %+v", plan)
	}
	if plan := planTarget(row, parityNow, nativeChat, "/v1/responses", true, anything, false, translate.Report{}); plan.Disposition != core.TransportReject || plan.Reason != core.TransportRejectTranslationUnevaluated {
		t.Fatalf("unevaluated adaptation plan = %+v", plan)
	}
	loss := translate.NewReport(translate.Loss{Path: "input[0]", Class: translate.LossDropped, Severity: translate.LossMaterial})
	if plan := planTarget(row, parityNow, nativeChat, "/v1/responses", true, anything, true, loss); plan.Disposition != core.TransportReject || plan.Reason != core.TransportRejectTranslationLoss {
		t.Fatalf("lossy adaptation plan = %+v", plan)
	}
}

// Gateway: TestTargetTransportModeReportsExecutedProviderPathDespiteStaleCatalog
// and TestTargetTransportModeUsesModelScopedAdaptation.
func TestParityResponseModeFollowsTheCatalogOrTheDeclaration(t *testing.T) {
	codex := converting()
	if mode := core.ResponseTransportMode(codex, nil, "fixture", core.ModelSurfaceResponses, nil, parityNow); mode != core.TransportModeNative {
		t.Fatalf("codex mode = %q, want native", mode)
	}
	evidence := &core.TransportEvidence{Row: core.ModelInfo{ID: "chat-only", SupportedAPIs: []string{"/v1/chat/completions"}}, RefreshedAt: parityNow}
	for surface, want := range map[core.ModelSurface]string{
		core.ModelSurfaceChatCompletions: core.TransportModeNative, core.ModelSurfaceResponses: core.TransportModeTranslated,
	} {
		if mode := core.ResponseTransportMode(openAICompatible(), nil, "chat-only", surface, evidence, parityNow); mode != want {
			t.Fatalf("%s mode = %q, want %q", surface, mode, want)
		}
	}
	if mode := core.ResponseTransportMode(nil, nil, "fixture", core.ModelSurfaceResponses, nil, parityNow); mode != core.TransportModeTranslated {
		t.Fatalf("a provider that cannot be built labelled %q", mode)
	}
}

// Fakes carrying the gateway's declarations for providers core has no
// vertical for yet, and Codex's.
func openAICompatible() core.Provider {
	both := []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses}
	return preservingProvider{surfaceProvider{both}, both}
}

func anthropicNative() core.Provider {
	messages := []core.ModelSurface{core.ModelSurfaceMessages}
	return preservingProvider{surfaceProvider{messages}, messages}
}

func converting() core.Provider {
	both := []core.ModelSurface{core.ModelSurfaceResponses, core.ModelSurfaceChatCompletions}
	return preservingProvider{surfaceProvider{both}, []core.ModelSurface{core.ModelSurfaceResponses}}
}

func chatAdapter() core.Provider {
	return surfaceProvider{[]core.ModelSurface{core.ModelSurfaceChatCompletions}}
}
