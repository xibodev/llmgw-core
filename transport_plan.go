package core

import (
	"time"

	translate "github.com/xibodev/llm-translate"
)

// TransportRequirement constrains whether a request may use adaptation.
type TransportRequirement string

const (
	TransportRequirementAny         TransportRequirement = "any"
	TransportRequirementNative      TransportRequirement = "native"
	TransportRequirementTransparent TransportRequirement = "transparent"
)

// TransportInterface is an available provider interface. Native records
// whether using this interface for its named surface preserves native provider
// semantics; SupportUnsupported therefore means available through adaptation,
// not unavailable.
type TransportInterface struct {
	Surface ModelSurface `json:"surface"`
	Native  Support      `json:"native"`
}

// TransportPlanRequest contains only evidence needed to choose a transport.
// Translation describes the conversion that would be used by an adapted plan.
type TransportPlanRequest struct {
	Operation            ModelOperation       `json:"operation"`
	Surface              ModelSurface         `json:"surface"`
	EvaluatedAt          time.Time            `json:"evaluated_at"`
	ExactTarget          bool                 `json:"exact_target"`
	Capabilities         ModelCapabilities    `json:"capabilities"`
	Interfaces           []TransportInterface `json:"interfaces"`
	Requirement          TransportRequirement `json:"requirement"`
	TranslationEvaluated bool                 `json:"translation_evaluated"`
	Translation          translate.Report     `json:"translation"`
}

type TransportDisposition string

const (
	TransportNative  TransportDisposition = "native"
	TransportAdapted TransportDisposition = "adapted"
	TransportReject  TransportDisposition = "reject"
)

type TransportConfidence string

const (
	TransportConfidenceLow    TransportConfidence = "low"
	TransportConfidenceMedium TransportConfidence = "medium"
	TransportConfidenceHigh   TransportConfidence = "high"
)

type TransportRejectReason string

const (
	TransportRejectInvalidRequirement     TransportRejectReason = "invalid_requirement"
	TransportRejectOperationUnsupported   TransportRejectReason = "operation_unsupported"
	TransportRejectNoInterface            TransportRejectReason = "no_interface"
	TransportRejectNativeRequired         TransportRejectReason = "native_required"
	TransportRejectExactTargetRequired    TransportRejectReason = "exact_target_required"
	TransportRejectNativeUnconfirmed      TransportRejectReason = "native_unconfirmed"
	TransportRejectTranslationUnevaluated TransportRejectReason = "translation_not_evaluated"
	TransportRejectTranslationLoss        TransportRejectReason = "translation_loss"
)

// TransportPlan is a deterministic decision over the supplied evidence.
type TransportPlan struct {
	Disposition TransportDisposition  `json:"disposition"`
	Surface     ModelSurface          `json:"surface"`
	Confidence  TransportConfidence   `json:"confidence"`
	Losses      []translate.Loss      `json:"losses,omitempty"`
	Reason      TransportRejectReason `json:"reason,omitempty"`
}

// PlanTransport selects a native interface when one is confirmed, preserves
// unknown evidence as eligible at low confidence, and otherwise considers an
// explicitly reported adaptation. It performs no I/O and applies no gateway or
// provider policy.
func PlanTransport(request TransportPlanRequest) TransportPlan {
	capabilitiesUsable, capabilitiesTrusted := transportCapabilityEvidence(request)
	operationSupport := SupportUnknown
	if capabilitiesUsable {
		operationSupport = request.Capabilities.OperationCompatibility(request.Operation)
	}
	if operationSupport == SupportUnsupported {
		return rejectTransport(request.Surface, transportEvidenceConfidence(capabilitiesTrusted), TransportRejectOperationUnsupported, nil)
	}
	if request.Requirement != "" &&
		request.Requirement != TransportRequirementAny &&
		request.Requirement != TransportRequirementNative &&
		request.Requirement != TransportRequirementTransparent {
		return rejectTransport(request.Surface, TransportConfidenceHigh, TransportRejectInvalidRequirement, nil)
	}

	confirmedNative, unknownNative, adapted, hasInterface := classifyTransportInterfaces(request, capabilitiesUsable)
	if request.Requirement == TransportRequirementTransparent {
		if !request.ExactTarget {
			return rejectTransport(request.Surface, TransportConfidenceHigh, TransportRejectExactTargetRequired, nil)
		}
		if capabilitiesTrusted && confirmedNative != nil && operationSupport == SupportSupported {
			return nativeTransport(confirmedNative.Surface, operationSupport, true, true)
		}
		confidence := transportEvidenceConfidence(capabilitiesTrusted && operationSupport != SupportUnknown && unknownNative == nil)
		return rejectTransport(request.Surface, confidence, TransportRejectNativeUnconfirmed, nil)
	}

	if confirmedNative != nil {
		return nativeTransport(confirmedNative.Surface, operationSupport, true, capabilitiesTrusted)
	}
	if unknownNative != nil {
		return nativeTransport(unknownNative.Surface, operationSupport, false, capabilitiesTrusted)
	}
	if request.Requirement == TransportRequirementNative {
		return rejectTransport(request.Surface, transportEvidenceConfidence(capabilitiesTrusted), TransportRejectNativeRequired, nil)
	}
	if !hasInterface || adapted == nil {
		return rejectTransport(request.Surface, transportEvidenceConfidence(capabilitiesTrusted), TransportRejectNoInterface, nil)
	}
	if !request.TranslationEvaluated {
		return rejectTransport(adapted.Surface, transportEvidenceConfidence(capabilitiesTrusted), TransportRejectTranslationUnevaluated, nil)
	}

	losses := append([]translate.Loss(nil), request.Translation.Losses...)
	if request.Translation.HasMaterialLoss() || hasUnsupportedLoss(losses) {
		return rejectTransport(adapted.Surface, transportEvidenceConfidence(capabilitiesTrusted), TransportRejectTranslationLoss, losses)
	}
	confidence := TransportConfidenceMedium
	if operationSupport == SupportUnknown || !capabilitiesTrusted {
		confidence = TransportConfidenceLow
	}
	return TransportPlan{
		Disposition: TransportAdapted,
		Surface:     adapted.Surface,
		Confidence:  confidence,
		Losses:      losses,
	}
}

func classifyTransportInterfaces(request TransportPlanRequest, capabilitiesUsable bool) (confirmedNative, unknownNative, adapted *TransportInterface, hasInterface bool) {
	requestedSurfaceSupport := SupportUnknown
	if capabilitiesUsable {
		requestedSurfaceSupport = request.Capabilities.SurfaceCompatibility(request.Surface)
	}
	for i := range request.Interfaces {
		candidate := &request.Interfaces[i]
		hasInterface = true
		if candidate.Surface != request.Surface {
			if adapted == nil {
				adapted = candidate
			}
			continue
		}

		switch {
		case candidate.Native == SupportUnsupported || requestedSurfaceSupport == SupportUnsupported:
			if adapted == nil {
				adapted = candidate
			}
		case candidate.Native == SupportSupported || requestedSurfaceSupport == SupportSupported:
			if confirmedNative == nil {
				confirmedNative = candidate
			}
		default:
			if unknownNative == nil {
				unknownNative = candidate
			}
		}
	}
	return confirmedNative, unknownNative, adapted, hasInterface
}

func nativeTransport(surface ModelSurface, operationSupport Support, confirmed, capabilitiesTrusted bool) TransportPlan {
	confidence := TransportConfidenceLow
	if capabilitiesTrusted && confirmed && operationSupport == SupportSupported {
		confidence = TransportConfidenceHigh
	}
	return TransportPlan{Disposition: TransportNative, Surface: surface, Confidence: confidence}
}

func transportCapabilityEvidence(request TransportPlanRequest) (usable, trusted bool) {
	capabilities := request.Capabilities
	if capabilities.SchemaVersion != ModelCapabilitiesSchemaVersion || request.EvaluatedAt.IsZero() ||
		capabilities.Freshness.Status(request.EvaluatedAt) != FreshnessFresh {
		return false, false
	}

	sourceKnown := capabilities.Provenance.Source == ModelCapabilitySourceUpstreamReported ||
		capabilities.Provenance.Source == ModelCapabilitySourceRegistryStatic ||
		capabilities.Provenance.Source == ModelCapabilitySourceModelsDev ||
		capabilities.Provenance.Source == ModelCapabilitySourceInferred ||
		capabilities.Provenance.Source == ModelCapabilitySourceOperatorOverride
	confidenceKnown := capabilities.Provenance.Confidence == ModelCapabilityConfidenceLow ||
		capabilities.Provenance.Confidence == ModelCapabilityConfidenceMedium ||
		capabilities.Provenance.Confidence == ModelCapabilityConfidenceHigh
	return true, sourceKnown && confidenceKnown && capabilities.Provenance.Confidence == ModelCapabilityConfidenceHigh
}

func transportEvidenceConfidence(trusted bool) TransportConfidence {
	if trusted {
		return TransportConfidenceHigh
	}
	return TransportConfidenceLow
}

func rejectTransport(surface ModelSurface, confidence TransportConfidence, reason TransportRejectReason, losses []translate.Loss) TransportPlan {
	return TransportPlan{
		Disposition: TransportReject,
		Surface:     surface,
		Confidence:  confidence,
		Losses:      losses,
		Reason:      reason,
	}
}

func hasUnsupportedLoss(losses []translate.Loss) bool {
	for _, loss := range losses {
		if loss.Class == translate.LossUnsupported {
			return true
		}
	}
	return false
}
