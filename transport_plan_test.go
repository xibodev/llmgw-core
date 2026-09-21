package core_test

import (
	"testing"
	"time"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

func TestPlanTransport(t *testing.T) {
	material := translate.NewReport(translate.Loss{
		Path: "input[0]", Class: translate.LossDropped, Severity: translate.LossMaterial, Detail: "fixture",
	})
	unsupported := translate.NewReport(translate.Loss{
		Path: "tools", Class: translate.LossUnsupported, Severity: translate.LossAdvisory, Detail: "fixture",
	})

	tests := []struct {
		name        string
		request     core.TransportPlanRequest
		want        core.TransportDisposition
		wantSurface core.ModelSurface
		confidence  core.TransportConfidence
		reason      core.TransportRejectReason
		losses      int
	}{
		{
			name: "confirmed native preferred over earlier adaptation",
			request: requestWithCapabilities(core.SupportSupported, core.SupportSupported,
				[]core.TransportInterface{
					{Surface: core.ModelSurfaceMessages, Native: core.SupportSupported},
					{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported},
				}),
			want: core.TransportNative, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceHigh,
		},
		{
			name: "unknown support remains eligible",
			request: requestWithCapabilities(core.SupportUnknown, core.SupportUnknown,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportUnknown}}),
			want: core.TransportNative, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceLow,
		},
		{
			name: "explicit unsupported operation rejects",
			request: requestWithCapabilities(core.SupportUnsupported, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}),
			want: core.TransportReject, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceHigh,
			reason: core.TransportRejectOperationUnsupported,
		},
		{
			name: "unevaluated adaptation rejects",
			request: requestWithCapabilities(core.SupportSupported, core.SupportUnsupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportUnsupported}}),
			want: core.TransportReject, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceHigh,
			reason: core.TransportRejectTranslationUnevaluated,
		},
		{
			name: "evaluated lossless adaptation allowed",
			request: withReport(requestWithCapabilities(core.SupportSupported, core.SupportUnsupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportUnsupported}}), translate.Report{}),
			want: core.TransportAdapted, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceMedium,
		},
		{
			name: "material adaptation loss rejects",
			request: withReport(requestWithCapabilities(core.SupportSupported, core.SupportUnsupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceChatCompletions, Native: core.SupportSupported}}), material),
			want: core.TransportReject, wantSurface: core.ModelSurfaceChatCompletions, confidence: core.TransportConfidenceHigh,
			reason: core.TransportRejectTranslationLoss, losses: 1,
		},
		{
			name: "unsupported advisory rejects",
			request: withReport(requestWithCapabilities(core.SupportSupported, core.SupportUnsupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceChatCompletions, Native: core.SupportSupported}}), unsupported),
			want: core.TransportReject, wantSurface: core.ModelSurfaceChatCompletions, confidence: core.TransportConfidenceHigh,
			reason: core.TransportRejectTranslationLoss, losses: 1,
		},
		{
			name: "transparent exact confirmed native",
			request: withRequirement(requestWithCapabilities(core.SupportSupported, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}),
				core.TransportRequirementTransparent, true),
			want: core.TransportNative, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceHigh,
		},
		{
			name: "transparent requires exact target",
			request: withRequirement(requestWithCapabilities(core.SupportSupported, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}),
				core.TransportRequirementTransparent, false),
			want: core.TransportReject, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceHigh,
			reason: core.TransportRejectExactTargetRequired,
		},
		{
			name: "transparent rejects unknown native evidence",
			request: withRequirement(requestWithCapabilities(core.SupportSupported, core.SupportUnknown,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportUnknown}}),
				core.TransportRequirementTransparent, true),
			want: core.TransportReject, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceLow,
			reason: core.TransportRejectNativeUnconfirmed,
		},
		{
			name: "Zen adapted interface rejects native requirement",
			request: withRequirement(requestWithCapabilities(core.SupportSupported, core.SupportUnsupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportUnsupported}}),
				core.TransportRequirementNative, true),
			want: core.TransportReject, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceHigh,
			reason: core.TransportRejectNativeRequired,
		},
		{
			name: "unknown schema ignores unsupported evidence",
			request: withSchemaVersion(requestWithCapabilities(core.SupportUnsupported, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}), 2),
			want: core.TransportNative, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceLow,
		},
		{
			name: "unknown schema cannot select transparent native",
			request: withRequirement(withSchemaVersion(requestWithCapabilities(core.SupportSupported, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}), 2),
				core.TransportRequirementTransparent, true),
			want: core.TransportReject, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceLow,
			reason: core.TransportRejectNativeUnconfirmed,
		},
		{
			name: "expired evidence remains eligible at low confidence",
			request: withEvaluationTime(requestWithCapabilities(core.SupportSupported, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}), time.Date(2026, 1, 2, 0, 0, 0, 0, time.UTC)),
			want: core.TransportNative, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceLow,
		},
		{
			name: "expired evidence cannot select transparent native",
			request: withRequirement(withEvaluationTime(requestWithCapabilities(core.SupportSupported, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}), time.Date(2026, 1, 2, 0, 0, 0, 0, time.UTC)),
				core.TransportRequirementTransparent, true),
			want: core.TransportReject, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceLow,
			reason: core.TransportRejectNativeUnconfirmed,
		},
		{
			name: "undated evidence remains eligible at low confidence",
			request: withFreshness(requestWithCapabilities(core.SupportSupported, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}), core.ModelCapabilityFreshness{}),
			want: core.TransportNative, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceLow,
		},
		{
			name: "transparent rejects unknown operation support at low confidence",
			request: withRequirement(requestWithCapabilities(core.SupportUnknown, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}),
				core.TransportRequirementTransparent, true),
			want: core.TransportReject, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceLow,
			reason: core.TransportRejectNativeUnconfirmed,
		},
		{
			name: "unknown provenance lowers confidence",
			request: withProvenance(requestWithCapabilities(core.SupportSupported, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}),
				core.ModelCapabilityProvenance{Source: "other", Confidence: core.ModelCapabilityConfidenceHigh}),
			want: core.TransportNative, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceLow,
		},
		{
			name: "unknown provenance cannot select transparent native",
			request: withRequirement(withProvenance(requestWithCapabilities(core.SupportSupported, core.SupportSupported,
				[]core.TransportInterface{{Surface: core.ModelSurfaceResponses, Native: core.SupportSupported}}),
				core.ModelCapabilityProvenance{Source: "other", Confidence: core.ModelCapabilityConfidenceHigh}),
				core.TransportRequirementTransparent, true),
			want: core.TransportReject, wantSurface: core.ModelSurfaceResponses, confidence: core.TransportConfidenceLow,
			reason: core.TransportRejectNativeUnconfirmed,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := core.PlanTransport(test.request)
			if got.Disposition != test.want || got.Surface != test.wantSurface || got.Confidence != test.confidence ||
				got.Reason != test.reason || len(got.Losses) != test.losses {
				t.Fatalf("PlanTransport() = %+v, want disposition=%q surface=%q confidence=%q reason=%q losses=%d",
					got, test.want, test.wantSurface, test.confidence, test.reason, test.losses)
			}
		})
	}
}

func requestWithCapabilities(operation, surface core.Support, interfaces []core.TransportInterface) core.TransportPlanRequest {
	evaluatedAt := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	expiresAt := evaluatedAt.Add(time.Hour)
	return core.TransportPlanRequest{
		Operation:   core.ModelOperationChat,
		Surface:     core.ModelSurfaceResponses,
		EvaluatedAt: evaluatedAt,
		Capabilities: core.ModelCapabilities{
			SchemaVersion: core.ModelCapabilitiesSchemaVersion,
			Operations:    core.ModelOperationCapabilities{Chat: operation},
			Surfaces:      core.ModelSurfaceCapabilities{Responses: surface},
			Provenance: core.ModelCapabilityProvenance{
				Source: core.ModelCapabilitySourceUpstreamReported, Confidence: core.ModelCapabilityConfidenceHigh,
			},
			Freshness: core.ModelCapabilityFreshness{ExpiresAt: &expiresAt},
		},
		Interfaces:  interfaces,
		Requirement: core.TransportRequirementAny,
	}
}

func withSchemaVersion(request core.TransportPlanRequest, version int) core.TransportPlanRequest {
	request.Capabilities.SchemaVersion = version
	return request
}

func withEvaluationTime(request core.TransportPlanRequest, evaluatedAt time.Time) core.TransportPlanRequest {
	request.EvaluatedAt = evaluatedAt
	return request
}

func withProvenance(request core.TransportPlanRequest, provenance core.ModelCapabilityProvenance) core.TransportPlanRequest {
	request.Capabilities.Provenance = provenance
	return request
}

func withFreshness(request core.TransportPlanRequest, freshness core.ModelCapabilityFreshness) core.TransportPlanRequest {
	request.Capabilities.Freshness = freshness
	return request
}

func withRequirement(request core.TransportPlanRequest, requirement core.TransportRequirement, exact bool) core.TransportPlanRequest {
	request.Requirement = requirement
	request.ExactTarget = exact
	return request
}

func withReport(request core.TransportPlanRequest, report translate.Report) core.TransportPlanRequest {
	request.TranslationEvaluated = true
	request.Translation = report
	return request
}
