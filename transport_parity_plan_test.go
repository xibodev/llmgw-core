package core_test

import (
	"errors"
	"testing"
	"time"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// These tests port the gateway's api/transport_mode_test.go. planTarget is
// its planTargetTransport, composed from core's pieces.
var parityNow = time.Date(2026, time.September, 21, 12, 0, 0, 0, time.UTC)

func planTarget(row core.ModelInfo, refreshedAt time.Time, interfaces []core.TransportInterface, surface string, exact bool,
	requirement core.TransportRequirement, evaluated bool, translation translate.Report) core.TransportPlan {
	return core.PlanTransport(core.TransportPlanRequest{
		Operation: core.ModelOperationChat, Surface: core.ParseSurfacePath(surface), EvaluatedAt: parityNow,
		ExactTarget: exact, Capabilities: core.TransportEvidence{Row: row, RefreshedAt: refreshedAt}.Capabilities(),
		Interfaces: interfaces, Requirement: requirement, TranslationEvaluated: evaluated, Translation: translation,
	})
}

func trusted(surfaces ...string) *core.ModelCapabilities {
	capabilities := core.AdaptModelCapabilities(map[string]any{"chat": true}, surfaces, time.Time{}, time.Time{})
	capabilities.Provenance = core.ModelCapabilityProvenance{Source: core.ModelCapabilitySourceUpstreamReported, Confidence: core.ModelCapabilityConfidenceHigh}
	return capabilities
}

var nativeResponses = []core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}

// Gateway: TestTransportModeHeaderValidationDoesNotUseUserAgent.
func TestParityTransportRequirementAcceptsOnlyTransparent(t *testing.T) {
	for value, want := range map[string]core.TransportRequirement{"": core.TransportRequirementAny, "  ": core.TransportRequirementAny,
		"transparent": core.TransportRequirementTransparent, " Transparent ": core.TransportRequirementTransparent} {
		if got, err := core.ParseTransportRequirement(value); err != nil || got != want {
			t.Fatalf("%q: requirement=%q err=%v, want %q", value, got, err, want)
		}
	}
	for _, value := range []string{"privileged", "native", "any"} {
		if _, err := core.ParseTransportRequirement(value); !errors.Is(err, core.ErrInvalidTransportRequirement) {
			t.Fatalf("%q was accepted: err=%v", value, err)
		}
	}
	for path, want := range map[string]core.ModelSurface{"/v1/chat/completions": core.ModelSurfaceChatCompletions, " /responses ": core.ModelSurfaceResponses,
		"/v1/messages": core.ModelSurfaceMessages, "/v1/embeddings": "", "ws:/responses": "", "": ""} {
		if got := core.ParseSurfacePath(path); got != want {
			t.Fatalf("%q: surface=%q, want %q", path, got, want)
		}
	}
}

// Gateway: TestPlanTargetTransportUsesFreshCatalogEvidence.
func TestParityPlanUsesFreshCatalogEvidence(t *testing.T) {
	row := core.ModelInfo{ID: "model", LegacyCapabilities: map[string]any{"chat": true}, SupportedAPIs: []string{"/v1/responses"}, Capabilities: trusted("/v1/responses")}
	transparent := core.TransportRequirementTransparent
	if plan := planTarget(row, parityNow.Add(-time.Minute), nativeResponses, "/responses", true, transparent, false, translate.Report{}); plan.Disposition != core.TransportNative || plan.Confidence != core.TransportConfidenceHigh {
		t.Fatalf("fresh transparent plan = %+v", plan)
	}
	if plan := planTarget(row, parityNow, nativeResponses, "/v1/messages", true, transparent, false, translate.Report{}); plan.Disposition != core.TransportReject {
		t.Fatalf("unsupported surface plan = %+v", plan)
	}
	for name, refreshedAt := range map[string]time.Time{"stale": parityNow.Add(-core.DefaultTransportFreshness), "unknown": {}} {
		if plan := planTarget(row, refreshedAt, nativeResponses, "/v1/responses", true, transparent, false, translate.Report{}); plan.Disposition != core.TransportReject || plan.Reason != core.TransportRejectNativeUnconfirmed {
			t.Fatalf("%s plan = %+v", name, plan)
		}
	}
	if plan := planTarget(row, parityNow.Add(-core.DefaultTransportFreshness), nativeResponses, "/v1/responses", true, core.TransportRequirementAny, false, translate.Report{}); plan.Disposition != core.TransportNative || plan.Confidence != core.TransportConfidenceLow {
		t.Fatalf("permissive stale plan = %+v", plan)
	}
}

func TestParityTransparentPlanTrustsCatalogEvidence(t *testing.T) {
	row := core.ModelInfo{
		ID:           "native-model",
		Capabilities: trusted("/responses"),
	}
	plan := planTarget(row, parityNow, nativeResponses, "/v1/responses", true, core.TransportRequirementTransparent, false, translate.Report{})
	if plan.Disposition != core.TransportNative || plan.Confidence != core.TransportConfidenceHigh {
		t.Fatalf("plan=%+v capabilities=%+v", plan, row.Capabilities)
	}
}
