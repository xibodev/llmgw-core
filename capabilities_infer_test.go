package core_test

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// Ported from the gateway's model_capabilities_test.go.
func TestAdaptModelCapabilitiesMapsLegacyMetadataWithoutInventingSupport(t *testing.T) {
	discovered := time.Date(2026, time.September, 20, 10, 0, 0, 0, time.UTC)
	verified := discovered.Add(time.Hour)
	capabilities := core.AdaptModelCapabilities(map[string]any{
		"embedding": false, "vision": true, "tool_calls": true,
		"reasoning_effort": []string{"low", "high"}, "structured_outputs": true,
		"context_window": 128000, "max_output_tokens": float64(16384),
	}, []string{"/v1/chat/completions", "/responses", "/v1/audio/speech"}, discovered, verified)

	if capabilities.Operations.Chat != core.SupportSupported ||
		capabilities.Operations.Embeddings != core.SupportUnsupported ||
		capabilities.Operations.AudioOut != core.SupportSupported ||
		capabilities.Operations.Image != core.SupportUnknown {
		t.Fatalf("operations = %+v", capabilities.Operations)
	}
	if capabilities.Surfaces.ChatCompletions != core.SupportSupported ||
		capabilities.Surfaces.Responses != core.SupportSupported ||
		capabilities.Surfaces.Messages != core.SupportUnknown {
		t.Fatalf("surfaces = %+v", capabilities.Surfaces)
	}
	if capabilities.Inputs.Text != core.SupportSupported || capabilities.Inputs.Image != core.SupportSupported ||
		capabilities.Tools != core.SupportSupported || capabilities.Reasoning != core.SupportSupported ||
		capabilities.StructuredOutput != core.SupportSupported {
		t.Fatalf("feature mapping = %+v", capabilities)
	}
	if capabilities.Limits.ContextTokens == nil || *capabilities.Limits.ContextTokens != 128000 ||
		capabilities.Limits.MaxOutputTokens == nil || *capabilities.Limits.MaxOutputTokens != 16384 {
		t.Fatalf("limits = %+v", capabilities.Limits)
	}
	if capabilities.Provenance.Source != core.ModelCapabilitySourceInferred ||
		capabilities.Freshness.DiscoveredAt == nil || !capabilities.Freshness.DiscoveredAt.Equal(discovered) ||
		capabilities.Freshness.VerifiedAt == nil || !capabilities.Freshness.VerifiedAt.Equal(verified) {
		t.Fatalf("evidence = %+v %+v", capabilities.Provenance, capabilities.Freshness)
	}
}

// Ported from the gateway's TestCatalogModelsWithTypedCapabilitiesAddsDiscoverySnapshot,
// through a core row.
func TestInferCapabilitiesAddsTheDiscoveryWithoutChangingTheRow(t *testing.T) {
	discovered := time.Date(2026, time.September, 20, 12, 0, 0, 0, time.FixedZone("fixture", 2*60*60))
	row := core.ModelInfo{
		ID: "speech", LegacyCapabilities: map[string]any{"audio": true, "tts": true},
		SupportedAPIs: []string{"/v1/audio/speech"},
	}
	capabilities := core.InferCapabilities(row, discovered, time.Time{})
	if row.Capabilities != nil || len(row.LegacyCapabilities) != 2 || len(row.SupportedAPIs) != 1 {
		t.Fatalf("inference changed the row: %+v", row)
	}
	if capabilities == nil || capabilities.Operations.AudioOut != core.SupportSupported ||
		capabilities.Operations.AudioIn != core.SupportUnknown ||
		capabilities.Freshness.DiscoveredAt == nil || !capabilities.Freshness.DiscoveredAt.Equal(discovered) ||
		capabilities.Freshness.DiscoveredAt.Location() != time.UTC || capabilities.Freshness.VerifiedAt != nil ||
		capabilities.Freshness.ExpiresAt != nil {
		t.Fatalf("inferred capabilities = %+v", capabilities)
	}
}

// Every typed_capabilities block the gateway's model list pins is what the
// port infers from the row's capabilities and surfaces, byte for byte once
// the discovery time is normalized as the gateway's characterization
// normalizes it. The golden is the gateway's codex-models.golden, copied
// unchanged.
func TestAdaptModelCapabilitiesMatchesTheGatewayModelList(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "gateway", "codex-models.golden"))
	if err != nil {
		t.Fatal(err)
	}
	_, body, found := bytes.Cut(bytes.ReplaceAll(raw, []byte("\r\n"), []byte("\n")), []byte("\nbody:\n"))
	if !found {
		t.Fatal("the golden has no body")
	}
	var list struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(body, &list); err != nil {
		t.Fatal(err)
	}
	discovered := time.Date(2026, time.September, 21, 12, 0, 0, 0, time.UTC)
	compared := 0
	for _, row := range list.Data {
		want, ok := row["typed_capabilities"]
		if !ok {
			continue
		}
		legacy, _ := row["capabilities"].(map[string]any)
		var surfaces []string
		for _, surface := range row["supported_surfaces"].([]any) {
			surfaces = append(surfaces, surface.(string))
		}
		adapted := core.AdaptModelCapabilities(legacy, surfaces, discovered, time.Time{})
		inferred := core.InferCapabilities(core.ModelInfo{LegacyCapabilities: legacy, SupportedAPIs: surfaces}, discovered, time.Time{})
		for name, capabilities := range map[string]*core.ModelCapabilities{"adapted": adapted, "inferred": inferred} {
			if got := normalizedCapabilities(t, capabilities); !reflect.DeepEqual(got, want) {
				t.Errorf("%s %s = %v, the gateway lists %v", row["id"], name, got, want)
			}
		}
		compared++
	}
	if compared != 6 {
		t.Fatalf("compared %d typed_capabilities blocks, want the golden's 6", compared)
	}
}

// normalizedCapabilities is capabilities as the gateway's characterization
// test records them: JSON with each timestamp replaced by "<time>".
func normalizedCapabilities(t *testing.T, capabilities *core.ModelCapabilities) any {
	t.Helper()
	encoded, err := json.Marshal(capabilities)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	freshness := decoded["freshness"].(map[string]any)
	for key := range freshness {
		freshness[key] = "<time>"
	}
	return decoded
}

func TestAdaptModelCapabilitiesOfNothingIsUnknown(t *testing.T) {
	encoded, err := json.Marshal(core.AdaptModelCapabilities(nil, nil, time.Time{}, time.Time{}))
	if err != nil {
		t.Fatal(err)
	}
	unknown := func(names ...string) string {
		fields := make([]string, len(names))
		for index, name := range names {
			fields[index] = `"` + name + `":"unknown"`
		}
		return strings.Join(fields, ",")
	}
	want := `{"schema_version":1,` +
		`"operations":{` + unknown("chat", "embeddings", "image", "audio_in", "audio_out", "video", "token_count") + `},` +
		`"surfaces":{` + unknown("chat_completions", "responses", "messages") + `},` +
		`"inputs":{` + unknown("text", "image") + `},` +
		unknown("tools", "reasoning", "structured_output", "streaming", "stateful_responses") + `,` +
		`"limits":{},"provenance":{"source":"inferred","confidence":"medium"},"freshness":{}}`
	if string(encoded) != want {
		t.Fatalf("capabilities = %s\nwant %s", encoded, want)
	}
}

func TestAdaptModelCapabilitiesReadsEveryLegacyValueShape(t *testing.T) {
	chat := func(c *core.ModelCapabilities) core.Support { return c.Operations.Chat }
	embeddings := func(c *core.ModelCapabilities) core.Support { return c.Operations.Embeddings }
	reasoning := func(c *core.ModelCapabilities) core.Support { return c.Reasoning }
	tools := func(c *core.ModelCapabilities) core.Support { return c.Tools }
	text := func(c *core.ModelCapabilities) core.Support { return c.Inputs.Text }
	for name, check := range map[string]struct {
		legacy map[string]any
		read   func(*core.ModelCapabilities) core.Support
		want   core.Support
	}{
		"true":                 {map[string]any{"chat": true}, chat, core.SupportSupported},
		"false":                {map[string]any{"chat": false}, chat, core.SupportUnsupported},
		"absent":               {map[string]any{"vision": true}, chat, core.SupportUnknown},
		"nonblank string":      {map[string]any{"reasoning": "high"}, reasoning, core.SupportSupported},
		"blank string":         {map[string]any{"reasoning": " "}, reasoning, core.SupportUnknown},
		"string list":          {map[string]any{"reasoning_effort": []string{"low"}}, reasoning, core.SupportSupported},
		"decoded list":         {map[string]any{"reasoning_effort": []any{"low"}}, reasoning, core.SupportSupported},
		"empty list":           {map[string]any{"reasoning_effort": []any{}}, reasoning, core.SupportUnknown},
		"number":               {map[string]any{"tools": float64(1)}, tools, core.SupportUnknown},
		"false beside true":    {map[string]any{"embedding": false, "embeddings": true}, embeddings, core.SupportSupported},
		"chat implies text":    {map[string]any{"chat": true}, text, core.SupportSupported},
		"stated text survives": {map[string]any{"chat": true, "text": false}, text, core.SupportUnsupported},
	} {
		t.Run(name, func(t *testing.T) {
			if got := check.read(core.AdaptModelCapabilities(check.legacy, nil, time.Time{}, time.Time{})); got != check.want {
				t.Fatalf("support = %v, want %v", got, check.want)
			}
		})
	}
}

func TestAdaptModelCapabilitiesReadsLimitsInEveryDecodedForm(t *testing.T) {
	for _, check := range []struct {
		value any
		want  int64
	}{
		{int(5), 5}, {int64(6), 6}, {float64(7.9), 7}, {json.Number("8"), 8}, {"9", 9},
		{json.Number("1.5"), 0}, {"many", 0}, {0, 0}, {-3, 0}, {true, 0},
	} {
		limits := core.AdaptModelCapabilities(map[string]any{"context_tokens": check.value}, nil, time.Time{}, time.Time{}).Limits
		if got := limits.ContextTokens; (got == nil) != (check.want == 0) || (got != nil && *got != check.want) {
			t.Errorf("context_tokens %#v = %v, want %d", check.value, got, check.want)
		}
	}
	limits := core.AdaptModelCapabilities(map[string]any{"context_window": 0, "context_tokens": 4096}, nil, time.Time{}, time.Time{}).Limits
	if limits.ContextTokens == nil || *limits.ContextTokens != 4096 {
		t.Fatalf("a zero context_window hid context_tokens: %v", limits.ContextTokens)
	}
}

func TestAdaptModelCapabilitiesReadsEverySurface(t *testing.T) {
	capabilities := core.AdaptModelCapabilities(nil, []string{
		" /V1/Messages ", "ws:/responses", "/v1/embeddings", "/v1/images/generations",
		"/v1/audio/transcriptions", "/v1/videos/generations", "/v1/messages/count_tokens", "/v1/unknown",
	}, time.Time{}, time.Time{})
	operations := capabilities.Operations
	if capabilities.Surfaces != (core.ModelSurfaceCapabilities{Messages: core.SupportSupported, Responses: core.SupportSupported}) ||
		operations != (core.ModelOperationCapabilities{
			Chat: core.SupportSupported, Embeddings: core.SupportSupported, Image: core.SupportSupported,
			AudioIn: core.SupportSupported, Video: core.SupportSupported, TokenCount: core.SupportSupported,
		}) || capabilities.Inputs.Text != core.SupportSupported || capabilities.Freshness != (core.ModelCapabilityFreshness{}) {
		t.Fatalf("capabilities = %+v", capabilities)
	}
}
