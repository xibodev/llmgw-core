package core

import (
	"encoding/json"
	"fmt"
	"time"
)

const ModelCapabilitiesSchemaVersion = 1

// Support is an explicitly tri-state capability value. Its zero value is
// SupportUnknown and marshals as "unknown" rather than false.
type Support uint8

const (
	SupportUnknown Support = iota
	SupportSupported
	SupportUnsupported
)

func (s Support) MarshalJSON() ([]byte, error) {
	if !s.valid() {
		return nil, fmt.Errorf("invalid capability support %d", s)
	}
	return json.Marshal(s.String())
}

func (s *Support) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	var parsed Support
	switch value {
	case "unknown":
		parsed = SupportUnknown
	case "supported":
		parsed = SupportSupported
	case "unsupported":
		parsed = SupportUnsupported
	default:
		return fmt.Errorf("invalid capability support %q", value)
	}
	*s = parsed
	return nil
}

func (s Support) valid() bool {
	return s == SupportUnknown || s == SupportSupported || s == SupportUnsupported
}

func (s Support) String() string {
	switch s {
	case SupportUnknown:
		return "unknown"
	case SupportSupported:
		return "supported"
	case SupportUnsupported:
		return "unsupported"
	default:
		return fmt.Sprintf("Support(%d)", s)
	}
}

func (s Support) IsKnown() bool     { return s == SupportSupported || s == SupportUnsupported }
func (s Support) IsSupported() bool { return s == SupportSupported }

type ModelOperation string

const (
	ModelOperationChat       ModelOperation = "chat"
	ModelOperationEmbeddings ModelOperation = "embeddings"
	ModelOperationImage      ModelOperation = "image"
	ModelOperationAudioIn    ModelOperation = "audio_in"
	ModelOperationAudioOut   ModelOperation = "audio_out"
	ModelOperationVideo      ModelOperation = "video"
	ModelOperationTokenCount ModelOperation = "token_count"
)

type ModelSurface string

const (
	ModelSurfaceChatCompletions ModelSurface = "chat_completions"
	ModelSurfaceResponses       ModelSurface = "responses"
	ModelSurfaceMessages        ModelSurface = "messages"
)

type ModelOperationCapabilities struct {
	Chat       Support `json:"chat"`
	Embeddings Support `json:"embeddings"`
	Image      Support `json:"image"`
	AudioIn    Support `json:"audio_in"`
	AudioOut   Support `json:"audio_out"`
	Video      Support `json:"video"`
	TokenCount Support `json:"token_count"`
}

type ModelSurfaceCapabilities struct {
	ChatCompletions Support `json:"chat_completions"`
	Responses       Support `json:"responses"`
	Messages        Support `json:"messages"`
}

type ModelInputCapabilities struct {
	Text  Support `json:"text"`
	Image Support `json:"image"`
}

// ModelCapabilityLimits uses pointers so an unknown limit is distinct from a
// provider-reported numeric value.
type ModelCapabilityLimits struct {
	ContextTokens   *int64 `json:"context_tokens,omitempty"`
	MaxOutputTokens *int64 `json:"max_output_tokens,omitempty"`
}

type ModelCapabilitySource string

const (
	ModelCapabilitySourceUpstreamReported ModelCapabilitySource = "upstream_reported"
	ModelCapabilitySourceRegistryStatic   ModelCapabilitySource = "registry_static"
	ModelCapabilitySourceModelsDev        ModelCapabilitySource = "models_dev"
	ModelCapabilitySourceInferred         ModelCapabilitySource = "inferred"
	ModelCapabilitySourceOperatorOverride ModelCapabilitySource = "operator_override"
)

type ModelCapabilityConfidence string

const (
	ModelCapabilityConfidenceLow    ModelCapabilityConfidence = "low"
	ModelCapabilityConfidenceMedium ModelCapabilityConfidence = "medium"
	ModelCapabilityConfidenceHigh   ModelCapabilityConfidence = "high"
)

type ModelCapabilityProvenance struct {
	Source     ModelCapabilitySource     `json:"source"`
	Confidence ModelCapabilityConfidence `json:"confidence"`
}

type ModelCapabilityFreshness struct {
	DiscoveredAt *time.Time `json:"discovered_at,omitempty"`
	VerifiedAt   *time.Time `json:"verified_at,omitempty"`
	ExpiresAt    *time.Time `json:"expires_at,omitempty"`
}

type FreshnessStatus string

const (
	FreshnessUnknown FreshnessStatus = "unknown"
	FreshnessFresh   FreshnessStatus = "fresh"
	FreshnessExpired FreshnessStatus = "expired"
)

// Status reports unknown when no expiry was supplied. Expiry is inclusive: a
// capability is expired at its expires_at instant.
func (f ModelCapabilityFreshness) Status(at time.Time) FreshnessStatus {
	if f.ExpiresAt == nil || f.ExpiresAt.IsZero() {
		return FreshnessUnknown
	}
	if !at.Before(*f.ExpiresAt) {
		return FreshnessExpired
	}
	return FreshnessFresh
}

func (f ModelCapabilityFreshness) IsFresh(at time.Time) bool {
	return f.Status(at) == FreshnessFresh
}

// ModelCapabilities is the bounded version 1 shared model capability contract.
type ModelCapabilities struct {
	SchemaVersion     int                        `json:"schema_version"`
	Operations        ModelOperationCapabilities `json:"operations"`
	Surfaces          ModelSurfaceCapabilities   `json:"surfaces"`
	Inputs            ModelInputCapabilities     `json:"inputs"`
	Tools             Support                    `json:"tools"`
	Reasoning         Support                    `json:"reasoning"`
	StructuredOutput  Support                    `json:"structured_output"`
	Streaming         Support                    `json:"streaming"`
	StatefulResponses Support                    `json:"stateful_responses"`
	Limits            ModelCapabilityLimits      `json:"limits"`
	Provenance        ModelCapabilityProvenance  `json:"provenance"`
	Freshness         ModelCapabilityFreshness   `json:"freshness"`
}

// OperationCompatibility returns unknown for both unreported capabilities and
// operation values outside this schema version.
func (c ModelCapabilities) OperationCompatibility(operation ModelOperation) Support {
	var support Support
	switch operation {
	case ModelOperationChat:
		support = c.Operations.Chat
	case ModelOperationEmbeddings:
		support = c.Operations.Embeddings
	case ModelOperationImage:
		support = c.Operations.Image
	case ModelOperationAudioIn:
		support = c.Operations.AudioIn
	case ModelOperationAudioOut:
		support = c.Operations.AudioOut
	case ModelOperationVideo:
		support = c.Operations.Video
	case ModelOperationTokenCount:
		support = c.Operations.TokenCount
	default:
		return SupportUnknown
	}
	return support
}

// SurfaceCompatibility returns unknown for both unreported capabilities and
// surface values outside this schema version.
func (c ModelCapabilities) SurfaceCompatibility(surface ModelSurface) Support {
	var support Support
	switch surface {
	case ModelSurfaceChatCompletions:
		support = c.Surfaces.ChatCompletions
	case ModelSurfaceResponses:
		support = c.Surfaces.Responses
	case ModelSurfaceMessages:
		support = c.Surfaces.Messages
	default:
		return SupportUnknown
	}
	return support
}
