package zen

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// metadataFixture is the opencode provider of the current models.dev
// catalog: provider fields the catalog ignores, rows carrying every field
// the catalog reads, and rows omitting each optional one.
const metadataFixture = `{
"opencode":{"id":"opencode","env":["OPENCODE_API_KEY"],"npm":"@ai-sdk/openai-compatible","api":"https://zen.example/v1","name":"OpenCode Zen","doc":"https://zen.example/docs","models":{
"chat-fixture-free":{"id":"chat-fixture-free","name":"Chat Fixture Free","family":"fixture","attachment":false,"reasoning":true,"tool_call":true,"temperature":true,"knowledge":"2025-01","release_date":"2025-06-01","last_updated":"2025-06-01","modalities":{"input":["text"],"output":["text"]},"open_weights":true,"cost":{"input":0,"output":0,"cache_read":0},"limit":{"context":262144,"output":65536}},
"responses-fixture-free":{"id":"responses-fixture-free","name":"Responses Fixture Free","attachment":true,"reasoning":true,"tool_call":true,"structured_output":true,"modalities":{"input":["text","image"],"output":["text"]},"cost":{"input":0,"output":0,"cache_read":0,"cache_write":0,"context_over_200k":{"input":0,"output":0}},"limit":{"context":400000,"output":128000},"status":"beta","provider":{"npm":"@ai-sdk/openai"}},
"sparse-fixture-free":{"id":"sparse-fixture-free","cost":{"input":0,"output":0}},
"no-cost-fixture":{"id":"no-cost-fixture","name":"No Cost"},
"paid-output-fixture":{"id":"paid-output-fixture","cost":{"input":0,"output":2}},
"paid-fixture":{"id":"paid-fixture","cost":{"input":1.25,"output":10,"cache_read":0.125}},
"deprecated-fixture-free":{"id":"deprecated-fixture-free","status":"deprecated","cost":{"input":0,"output":0}},
"messages-fixture-free":{"id":"messages-fixture-free","provider":{"npm":"@ai-sdk/anthropic"},"cost":{"input":0,"output":0}}
}},
"other":{"id":"other","models":{"other-free":{"id":"other-free","cost":{"input":0,"output":0}}}}}`

// liveFixture is Zen's current /models envelope. Rows may omit everything
// but the id.
const liveFixture = `{"object":"list","data":[
{"id":"chat-fixture-free","object":"model","created":1750000000,"owned_by":"opencode"},
{"id":"responses-fixture-free","object":"model","created":1750000000,"owned_by":"opencode"},
{"id":"sparse-fixture-free"},{"id":"no-cost-fixture","object":"model"},
{"id":"paid-output-fixture"},{"id":"paid-fixture"},{"id":"messages-fixture-free"}]}`

func modelIDs(models []core.ModelInfo) []string {
	ids := make([]string, 0, len(models))
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	return ids
}

func TestNormalizeReadsCurrentUpstreamShapes(t *testing.T) {
	observedAt := time.Date(2026, time.September, 24, 12, 0, 0, 0, time.FixedZone("fixture", 3600))
	options := NormalizeOptions{ProviderID: "zen-fixture", ObservedAt: observedAt, CapabilityTTL: 2 * time.Hour}

	snapshot, err := Normalize([]byte(metadataFixture), nil, options)
	if err != nil || snapshot.Status != core.CatalogDiscovered || !snapshot.ObservedAt.Equal(observedAt) {
		t.Fatalf("snapshot = %+v, err = %v", snapshot, err)
	}
	if want := []string{"chat-fixture-free", "no-cost-fixture", "paid-output-fixture", "responses-fixture-free", "sparse-fixture-free"}; !slices.Equal(modelIDs(snapshot.Models), want) {
		t.Fatalf("snapshot = %v, want %v", modelIDs(snapshot.Models), want)
	}

	verified, err := Normalize([]byte(metadataFixture), []byte(liveFixture), options)
	if err != nil || verified.Status != core.CatalogDiscovered {
		t.Fatalf("verified = %+v, err = %v", verified, err)
	}
	if want := []string{"chat-fixture-free", "responses-fixture-free", "sparse-fixture-free"}; !slices.Equal(modelIDs(verified.Models), want) {
		t.Fatalf("verified = %v, want %v", modelIDs(verified.Models), want)
	}
	for _, model := range verified.Models {
		freshness := model.Capabilities.Freshness
		if model.OwnedBy != "zen-fixture" || model.Object != "model" || freshness.DiscoveredAt.Location() != time.UTC ||
			!freshness.DiscoveredAt.Equal(observedAt) || !freshness.ExpiresAt.Equal(observedAt.Add(2*time.Hour)) {
			t.Fatalf("%s = %+v", model.ID, model)
		}
	}

	chat, responses, sparse := verified.Models[0], verified.Models[1], verified.Models[2]
	if surface, _ := NativeSurface(chat); surface != core.ModelSurfaceChatCompletions || chat.Description != "Chat Fixture Free" {
		t.Fatalf("chat = %+v", chat)
	}
	if caps := chat.Capabilities; caps.Tools != core.SupportSupported || caps.Reasoning != core.SupportSupported ||
		caps.StructuredOutput != core.SupportUnknown || caps.Inputs.Text != core.SupportSupported ||
		caps.Inputs.Image != core.SupportUnsupported || *caps.Limits.ContextTokens != 262144 || *caps.Limits.MaxOutputTokens != 65536 {
		t.Fatalf("chat capabilities = %+v", caps)
	}
	if surface, _ := NativeSurface(responses); surface != core.ModelSurfaceResponses ||
		responses.Capabilities.Inputs.Image != core.SupportSupported || responses.Capabilities.StructuredOutput != core.SupportSupported {
		t.Fatalf("responses = %+v", responses)
	}
	// A row omitting every optional field reports what it omits as unknown.
	if caps := sparse.Capabilities; sparse.Description != "" || caps.Tools != core.SupportUnknown || caps.Reasoning != core.SupportUnknown ||
		caps.Inputs.Text != core.SupportUnknown || caps.Inputs.Image != core.SupportUnknown ||
		caps.Limits.ContextTokens != nil || caps.Limits.MaxOutputTokens != nil {
		t.Fatalf("sparse = %+v", sparse)
	}
	if surface, _ := NativeSurface(sparse); surface != core.ModelSurfaceChatCompletions {
		t.Fatalf("sparse surface = %q", surface)
	}
}

func TestNormalizeDefaultsAndEmptyCatalogs(t *testing.T) {
	evidence, err := Normalize([]byte(metadataFixture), []byte(`{"data":[{"id":"paid-fixture"}]}`), NormalizeOptions{})
	if err != nil || evidence.Status != core.CatalogEmpty || len(evidence.Models) != 0 {
		t.Fatalf("evidence = %+v, err = %v", evidence, err)
	}
	evidence, err = Normalize([]byte(metadataFixture), nil, NormalizeOptions{ObservedAt: time.Unix(0, 0)})
	if err != nil || evidence.Models[0].OwnedBy != DefaultProviderID ||
		!evidence.Models[0].Capabilities.Freshness.ExpiresAt.Equal(time.Unix(0, 0).Add(time.Hour)) {
		t.Fatalf("evidence = %+v, err = %v", evidence, err)
	}
}

func TestDiscoveryNormalizesAsNormalizeDoes(t *testing.T) {
	observedAt := time.Date(2026, time.September, 24, 12, 0, 0, 0, time.UTC)
	var live atomic.Value
	live.Store(liveFixture)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/zen/models" {
			_, _ = fmt.Fprint(w, live.Load())
			return
		}
		_, _ = fmt.Fprint(w, metadataFixture)
	}))
	defer server.Close()
	client := testClient(t, server.URL+"/zen", server.URL+"/metadata", observedAt)
	options := NormalizeOptions{ObservedAt: observedAt}

	discovered, err := client.Discover(context.Background())
	want, _ := Normalize([]byte(metadataFixture), nil, options)
	if err != nil || !reflect.DeepEqual(discovered, want) {
		t.Fatalf("Discover = %+v, err = %v, want %+v", discovered, err, want)
	}
	verified, err := client.DiscoverVerified(context.Background())
	want, _ = Normalize([]byte(metadataFixture), []byte(liveFixture), options)
	if err != nil || !reflect.DeepEqual(verified, want) {
		t.Fatalf("DiscoverVerified = %+v, err = %v, want %+v", verified, err, want)
	}
	live.Store(`{"object":"list"}`)
	if _, err := client.DiscoverVerified(context.Background()); err == nil || err.Error() != "decode Zen catalog" {
		t.Fatalf("err = %v", err)
	}
}

func TestNormalizeReportsTheDocumentItCannotRead(t *testing.T) {
	observedAt := time.Unix(1750000000, 0).UTC()
	for _, test := range []struct{ metadata, live, document, code, message string }{
		{`not json`, ``, DocumentMetadata, "invalid_json", "decode models.dev catalog"},
		{`{}`, ``, DocumentMetadata, "invalid_shape", "models.dev catalog has no opencode provider"},
		{`{"opencode":[]}`, ``, DocumentMetadata, "invalid_json", "decode models.dev opencode provider"},
		{`{"opencode":{"npm":"@ai-sdk/openai-compatible"}}`, ``, DocumentMetadata, "invalid_shape", "decode models.dev opencode provider"},
		{metadataFixture, `not json`, DocumentCatalog, "invalid_json", "decode Zen catalog"},
		{metadataFixture, `{"object":"list"}`, DocumentCatalog, "invalid_shape", "decode Zen catalog"},
		{metadataFixture, `{"data":[{"id":"chat-fixture-free","created":"yesterday"}]}`, DocumentCatalog, "invalid_json", "decode Zen catalog"},
	} {
		var live []byte
		if test.live != "" {
			live = []byte(test.live)
		}
		evidence, err := Normalize([]byte(test.metadata), live, NormalizeOptions{ObservedAt: observedAt})
		var document *DocumentError
		if !errors.As(err, &document) || document.Document != test.document || document.Code != test.code || err.Error() != test.message {
			t.Fatalf("%s/%s: err = %v", test.metadata, test.live, err)
		}
		if evidence.Status != core.CatalogFailed || len(evidence.Models) != 0 || !evidence.ObservedAt.Equal(observedAt) {
			t.Fatalf("%s/%s: evidence = %+v", test.metadata, test.live, evidence)
		}
	}
}
