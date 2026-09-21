package core_test

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

func TestModelCapabilitiesJSONRoundTrip(t *testing.T) {
	discovered := time.Date(2026, time.September, 20, 10, 0, 0, 0, time.UTC)
	verified := discovered.Add(time.Hour)
	expires := verified.Add(24 * time.Hour)
	contextTokens := int64(128000)
	maxOutputTokens := int64(16384)
	want := core.ModelInfo{
		ID: "model-a",
		Capabilities: &core.ModelCapabilities{
			SchemaVersion: core.ModelCapabilitiesSchemaVersion,
			Operations: core.ModelOperationCapabilities{
				Chat: core.SupportSupported, Embeddings: core.SupportUnsupported,
			},
			Surfaces: core.ModelSurfaceCapabilities{
				ChatCompletions: core.SupportSupported, Responses: core.SupportSupported,
			},
			Inputs: core.ModelInputCapabilities{Text: core.SupportSupported, Image: core.SupportUnknown},
			Tools:  core.SupportSupported, Reasoning: core.SupportSupported,
			StructuredOutput: core.SupportSupported, Streaming: core.SupportSupported,
			StatefulResponses: core.SupportUnsupported,
			Limits:            core.ModelCapabilityLimits{ContextTokens: &contextTokens, MaxOutputTokens: &maxOutputTokens},
			Provenance: core.ModelCapabilityProvenance{
				Source: core.ModelCapabilitySourceUpstreamReported, Confidence: core.ModelCapabilityConfidenceHigh,
			},
			Freshness: core.ModelCapabilityFreshness{
				DiscoveredAt: &discovered, VerifiedAt: &verified, ExpiresAt: &expires,
			},
		},
	}

	encoded, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got core.ModelInfo
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	reencoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(reencoded) != string(encoded) {
		t.Fatalf("round trip JSON mismatch:\n got: %s\nwant: %s", reencoded, encoded)
	}
	if got.Capabilities == nil || got.Capabilities.SchemaVersion != core.ModelCapabilitiesSchemaVersion ||
		got.Capabilities.Operations.Chat != core.SupportSupported ||
		got.Capabilities.Operations.Image != core.SupportUnknown ||
		got.Capabilities.Freshness.VerifiedAt == nil || !got.Capabilities.Freshness.VerifiedAt.Equal(verified) {
		t.Fatalf("round trip values mismatch: %#v", got.Capabilities)
	}
}

func TestModelCapabilitiesUnknownIsNotUnsupported(t *testing.T) {
	var support core.Support
	if support != core.SupportUnknown || support.IsKnown() || support.IsSupported() {
		t.Fatalf("zero support must be unknown: %v", support)
	}
	if core.SupportUnknown == core.SupportUnsupported {
		t.Fatal("unknown collapsed into unsupported")
	}

	encoded, err := json.Marshal(core.ModelInputCapabilities{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), `false`) || !strings.Contains(string(encoded), `"text":"unknown"`) {
		t.Fatalf("unknown capability encoded incorrectly: %s", encoded)
	}
}

func TestModelCapabilityFreshness(t *testing.T) {
	now := time.Date(2026, time.September, 20, 12, 0, 0, 0, time.UTC)
	if status := (core.ModelCapabilityFreshness{}).Status(now); status != core.FreshnessUnknown {
		t.Fatalf("freshness without expiry = %q", status)
	}

	expires := now.Add(time.Hour)
	freshness := core.ModelCapabilityFreshness{ExpiresAt: &expires}
	if !freshness.IsFresh(now) || freshness.Status(now) != core.FreshnessFresh {
		t.Fatalf("future expiry is not fresh: %q", freshness.Status(now))
	}
	if freshness.IsFresh(expires) || freshness.Status(expires) != core.FreshnessExpired {
		t.Fatalf("expiry instant is not expired: %q", freshness.Status(expires))
	}
}

func TestModelCapabilityOperationAndSurfaceCompatibility(t *testing.T) {
	capabilities := core.ModelCapabilities{
		Operations: core.ModelOperationCapabilities{
			Chat: core.SupportSupported, Embeddings: core.SupportUnsupported,
		},
		Surfaces: core.ModelSurfaceCapabilities{
			Responses: core.SupportSupported, Messages: core.SupportUnsupported,
		},
	}

	checks := []struct {
		name string
		got  core.Support
		want core.Support
	}{
		{name: "supported operation", got: capabilities.OperationCompatibility(core.ModelOperationChat), want: core.SupportSupported},
		{name: "unsupported operation", got: capabilities.OperationCompatibility(core.ModelOperationEmbeddings), want: core.SupportUnsupported},
		{name: "unknown operation", got: capabilities.OperationCompatibility(core.ModelOperationImage), want: core.SupportUnknown},
		{name: "invalid operation", got: capabilities.OperationCompatibility(core.ModelOperation("future")), want: core.SupportUnknown},
		{name: "supported surface", got: capabilities.SurfaceCompatibility(core.ModelSurfaceResponses), want: core.SupportSupported},
		{name: "unsupported surface", got: capabilities.SurfaceCompatibility(core.ModelSurfaceMessages), want: core.SupportUnsupported},
		{name: "unknown surface", got: capabilities.SurfaceCompatibility(core.ModelSurfaceChatCompletions), want: core.SupportUnknown},
	}
	for _, check := range checks {
		t.Run(check.name, func(t *testing.T) {
			if check.got != check.want {
				t.Fatalf("compatibility = %v, want %v", check.got, check.want)
			}
		})
	}
}

func TestModelInfoWithoutCapabilitiesRetainsLegacyJSONShape(t *testing.T) {
	encoded, err := json.Marshal(core.ModelInfo{ID: "legacy-model", Object: "model"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "capabilities") {
		t.Fatalf("optional capabilities appeared in legacy JSON: %s", encoded)
	}
}
