package core

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"
)

// AdaptModelCapabilities converts the gateway's compatibility metadata into
// the bounded capability contract: legacy is a catalog row's untyped
// capability map, and surfaces are the endpoints it lists. Missing keys
// remain unknown; only explicit false values become unsupported.
//
// It is the gateway's function, ported verbatim, so a product and the
// gateway infer the same capabilities from the same row. The result is
// inferred with medium confidence. A listed chat surface supports the chat
// operation, and chat implies text input unless the map says otherwise.
// Freshness records the times given, in UTC, and never an expiry.
func AdaptModelCapabilities(
	legacy map[string]any, surfaces []string, discoveredAt, verifiedAt time.Time,
) *ModelCapabilities {
	capabilities := &ModelCapabilities{
		SchemaVersion: ModelCapabilitiesSchemaVersion,
		Provenance: ModelCapabilityProvenance{
			Source: ModelCapabilitySourceInferred, Confidence: ModelCapabilityConfidenceMedium,
		},
	}
	capabilities.Operations.Chat = legacySupport(legacy, "chat")
	capabilities.Operations.Embeddings = legacySupport(legacy, "embedding", "embeddings")
	capabilities.Operations.Image = legacySupport(legacy, "image")
	capabilities.Operations.AudioIn = legacySupport(legacy, "transcription", "stt", "asr", "audio_in")
	capabilities.Operations.AudioOut = legacySupport(legacy, "tts", "speech", "audio_out")
	capabilities.Operations.Video = legacySupport(legacy, "video")
	capabilities.Operations.TokenCount = legacySupport(legacy, "token_count")

	capabilities.Inputs.Text = legacySupport(legacy, "text")
	capabilities.Inputs.Image = legacySupport(legacy, "vision")
	capabilities.Tools = legacySupport(legacy, "tool_calls", "tools")
	capabilities.Reasoning = legacySupport(legacy, "reasoning", "reasoning_effort")
	capabilities.StructuredOutput = legacySupport(legacy, "structured_outputs", "structured_output")
	capabilities.Streaming = legacySupport(legacy, "streaming")
	capabilities.StatefulResponses = legacySupport(legacy, "stateful_responses")

	for _, raw := range surfaces {
		surface := strings.ToLower(strings.TrimSpace(raw))
		switch {
		case surface == "/chat/completions" || surface == "/v1/chat/completions":
			capabilities.Surfaces.ChatCompletions = SupportSupported
			markSupported(&capabilities.Operations.Chat)
		case surface == "/responses" || surface == "/v1/responses" || surface == "ws:/responses":
			capabilities.Surfaces.Responses = SupportSupported
			markSupported(&capabilities.Operations.Chat)
		case surface == "/messages" || surface == "/v1/messages":
			capabilities.Surfaces.Messages = SupportSupported
			markSupported(&capabilities.Operations.Chat)
		case strings.Contains(surface, "/embeddings"):
			markSupported(&capabilities.Operations.Embeddings)
		case strings.Contains(surface, "/images/"):
			markSupported(&capabilities.Operations.Image)
		case strings.Contains(surface, "/audio/transcriptions"):
			markSupported(&capabilities.Operations.AudioIn)
		case strings.Contains(surface, "/audio/speech"):
			markSupported(&capabilities.Operations.AudioOut)
		case strings.Contains(surface, "/videos/"):
			markSupported(&capabilities.Operations.Video)
		case strings.Contains(surface, "count_tokens"):
			markSupported(&capabilities.Operations.TokenCount)
		}
	}
	if capabilities.Operations.Chat == SupportSupported && capabilities.Inputs.Text == SupportUnknown {
		capabilities.Inputs.Text = SupportSupported
	}
	capabilities.Limits.ContextTokens = legacyInt64(legacy, "context_window", "context_tokens")
	capabilities.Limits.MaxOutputTokens = legacyInt64(legacy, "max_output_tokens")
	if !discoveredAt.IsZero() {
		discovered := discoveredAt.UTC()
		capabilities.Freshness.DiscoveredAt = &discovered
	}
	if !verifiedAt.IsZero() {
		verified := verifiedAt.UTC()
		capabilities.Freshness.VerifiedAt = &verified
	}
	return capabilities
}

// InferCapabilities infers a row's capabilities from what the row lists,
// its LegacyCapabilities and its SupportedAPIs, through
// AdaptModelCapabilities. It ignores Capabilities, so the caller decides
// whether capabilities a catalog reported take precedence over inferred
// ones, as the gateway's catalog lets them when it stores a row.
func InferCapabilities(model ModelInfo, discoveredAt, verifiedAt time.Time) *ModelCapabilities {
	return AdaptModelCapabilities(model.LegacyCapabilities, model.SupportedAPIs, discoveredAt, verifiedAt)
}

// legacySupport reads the first of keys that claims support: true, a
// nonblank string or a nonempty list. Otherwise an explicit false is
// unsupported, and anything else unknown.
func legacySupport(values map[string]any, keys ...string) Support {
	foundUnsupported := false
	for _, key := range keys {
		value, found := values[key]
		if !found {
			continue
		}
		if enabled, ok := value.(bool); ok {
			if enabled {
				return SupportSupported
			}
			foundUnsupported = true
			continue
		}
		switch current := value.(type) {
		case string:
			if strings.TrimSpace(current) != "" {
				return SupportSupported
			}
		case []string:
			if len(current) > 0 {
				return SupportSupported
			}
		case []any:
			if len(current) > 0 {
				return SupportSupported
			}
		}
	}
	if foundUnsupported {
		return SupportUnsupported
	}
	return SupportUnknown
}

// legacyInt64 reads the first of keys that holds a positive integer, in any
// form a decoded map may hold it. A float is truncated, and a string or a
// json.Number must spell an integer.
func legacyInt64(values map[string]any, keys ...string) *int64 {
	for _, key := range keys {
		value, found := values[key]
		if !found {
			continue
		}
		var parsed int64
		switch current := value.(type) {
		case int:
			parsed = int64(current)
		case int64:
			parsed = current
		case float64:
			parsed = int64(current)
		case json.Number:
			parsed, _ = current.Int64()
		case string:
			parsed, _ = strconv.ParseInt(current, 10, 64)
		}
		if parsed > 0 {
			return &parsed
		}
	}
	return nil
}

func markSupported(value *Support) {
	*value = SupportSupported
}
