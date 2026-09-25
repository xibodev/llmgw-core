package core

import (
	"fmt"
	"time"
)

// The chat surfaces a model list classifies, labelled as the gateway's
// /v1/models labels them.
const (
	ListingChatCompletions = "/v1/chat/completions"
	ListingResponses       = "/v1/responses"
	ListingMessages        = "/v1/messages"
)

// ListingSurfaces classifies the chat surfaces of a model list's row, as
// the gateway's native_surfaces, emulated_surfaces and unknown_surfaces do:
// a surface is native when a plan under TransportRequirementAny carries it
// natively, emulated when the plan adapts it, and unknown when it refuses.
// A row that neither claims chat in LegacyCapabilities nor lists a chat
// surface classifies nothing. The row is as the list presents it, so a
// product that infers capabilities or surfaces for presentation passes them
// in its LegacyCapabilities and SupportedAPIs. Credential is the one the
// caller's requests carry.
func ListingSurfaces(provider Provider, credential *Credential, evidence TransportEvidence, evaluatedAt time.Time) (native, emulated, unknown []string) {
	row := evidence.Row
	chat, _ := row.LegacyCapabilities["chat"].(bool)
	for _, surface := range row.SupportedAPIs {
		if ParseSurfacePath(surface) != "" {
			chat = true
			break
		}
	}
	if !chat {
		return nil, nil, nil
	}
	capabilities := evidence.Capabilities()
	interfaces := TransportInterfaces(provider, credential, row.ID, row.SupportedAPIs)
	native, emulated, unknown = []string{}, []string{}, []string{}
	for _, label := range []string{ListingChatCompletions, ListingResponses, ListingMessages} {
		plan := PlanTransport(TransportPlanRequest{
			Operation: ModelOperationChat, Surface: ParseSurfacePath(label), EvaluatedAt: evaluatedAt,
			ExactTarget: true, Capabilities: capabilities, Interfaces: interfaces,
			Requirement: TransportRequirementAny, TranslationEvaluated: true,
		})
		switch plan.Disposition {
		case TransportNative:
			native = append(native, label)
		case TransportAdapted:
			emulated = append(emulated, label)
		default:
			unknown = append(unknown, label)
		}
	}
	return native, emulated, unknown
}

// The transport labels of a response.
const (
	TransportModeNative      = "native"
	TransportModeTranslated  = "translated"
	TransportModeTransparent = "transparent"
)

// ResponseTransportMode labels the transport of a response an exact target
// served a request that carried credential, as the gateway's
// X-LLMGW-Transport-Mode does. With evidence from the target's stored
// catalog row, it is native when a plan under TransportRequirementAny
// carries the surface natively. Without it, it reports the path the
// provider takes: native when the provider preserves the surface's wire.
// Anything else is translated.
func ResponseTransportMode(provider Provider, credential *Credential, model string, surface ModelSurface, evidence *TransportEvidence, evaluatedAt time.Time) string {
	if evidence != nil && provider != nil {
		plan := PlanTransport(TransportPlanRequest{
			Operation: ModelOperationChat, Surface: surface, EvaluatedAt: evaluatedAt, ExactTarget: true,
			Capabilities: evidence.Capabilities(),
			Interfaces:   TransportInterfaces(provider, credential, model, evidence.Row.SupportedAPIs),
			Requirement:  TransportRequirementAny, TranslationEvaluated: true,
		})
		if plan.Disposition == TransportNative {
			return TransportModeNative
		}
		return TransportModeTranslated
	}
	if provider != nil && PreservesWireFor(provider, credential, model, surface) {
		return TransportModeNative
	}
	return TransportModeTranslated
}

// Reasons a transparent request is refused besides a plan's.
const (
	// TransportRejectNotCataloged is a model its stored catalog does not
	// list, so no evidence can confirm a native surface.
	TransportRejectNotCataloged TransportRejectReason = "not_cataloged"
	// TransportRejectStreaming is a transparent request that streams,
	// which transparency does not support yet.
	TransportRejectStreaming TransportRejectReason = "streaming_unsupported"
)

// TransportRejectError reports a request refused for the transport it
// requires, before anything reached the provider. A product words the
// refusal from Reason, as the gateway does, and NativeInterface tells a
// surface the provider does not preserve from evidence that is stale or
// untrusted.
type TransportRejectError struct {
	Reason  TransportRejectReason
	Surface ModelSurface
	Model   string
	// NativeInterface reports that the stored catalog lists Surface for
	// Model and the provider preserves its wire.
	NativeInterface bool
}

func (e *TransportRejectError) Error() string {
	return fmt.Sprintf("the %s request for model %q cannot be carried transparently: %s", e.Surface, e.Model, e.Reason)
}
