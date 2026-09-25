package providers

import (
	"errors"
	"slices"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers/zen"
)

// zenMetadataFixture is the opencode provider of the current models.dev
// catalog, with rows carrying every field the catalog reads and rows
// omitting each optional one.
const zenMetadataFixture = `{"opencode":{"id":"opencode","npm":"@ai-sdk/openai-compatible","api":"https://zen.example/v1","name":"OpenCode Zen","models":{
"chat-fixture-free":{"id":"chat-fixture-free","name":"Chat Fixture Free","reasoning":true,"tool_call":true,"modalities":{"input":["text"],"output":["text"]},"cost":{"input":0,"output":0,"cache_read":0},"limit":{"context":262144,"output":65536}},
"muse-fixture-free":{"id":"muse-fixture-free","name":"Muse Fixture Free","status":"beta","provider":{"npm":"@ai-sdk/openai"},"cost":{"input":0,"output":0,"context_over_200k":{"input":0,"output":0}}},
"sparse-fixture-free":{"id":"sparse-fixture-free","cost":{"input":0,"output":0}},
"paid-fixture":{"id":"paid-fixture","cost":{"input":1,"output":8}},
"unpriced-fixture":{"id":"unpriced-fixture"}}}}`

// zenLiveFixture is Zen's current /models envelope; rows may omit all but
// their id.
const zenLiveFixture = `{"object":"list","data":[{"id":"chat-fixture-free","object":"model","created":1750000000,"owned_by":"opencode"},` +
	`{"id":"muse-fixture-free"},{"id":"sparse-fixture-free","object":"model"},{"id":"paid-fixture"},{"id":"unpriced-fixture"}]}`

func TestParseZenAnonymousModelsAdmitsFreeModelsOnTheirSurface(t *testing.T) {
	observedAt := time.Date(2026, time.September, 24, 12, 0, 0, 0, time.UTC)
	models, err := parseZenAnonymousModels([]byte(zenMetadataFixture), []byte(zenLiveFixture), zen.NormalizeOptions{ObservedAt: observedAt})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"chat-fixture-free": "/chat/completions", "muse-fixture-free": "/responses", "sparse-fixture-free": "/chat/completions"}
	if len(models) != len(want) {
		t.Fatalf("models = %+v", models)
	}
	for _, model := range models {
		if !slices.Equal(model.SupportedAPIs, []string{want[model.ID]}) || !slices.Equal(model.Tags, []string{ModelTagFree}) ||
			model.OwnedBy != zen.DefaultProviderID || model.Capabilities == nil ||
			model.Capabilities.Provenance.Source != core.ModelCapabilitySourceModelsDev {
			t.Fatalf("%s = %+v", model.ID, model)
		}
		chat, responses := zenRowSurfaces(model)
		if chat == responses || responses != (want[model.ID] == "/responses") {
			t.Fatalf("%s surfaces = chat %v responses %v", model.ID, chat, responses)
		}
	}
}

func TestParseZenAnonymousModelsReportsTheUnreadableDocument(t *testing.T) {
	for _, test := range []struct{ metadata, live, code string }{
		{`[]`, zenLiveFixture, "metadata_invalid_json"},
		{`{"models":{}}`, zenLiveFixture, "metadata_invalid_shape"},
		{zenMetadataFixture, `{"data":{}}`, "invalid_json"},
		{zenMetadataFixture, `{}`, "invalid_shape"},
	} {
		_, err := parseZenAnonymousModels([]byte(test.metadata), []byte(test.live), zen.NormalizeOptions{})
		var failure *core.ProviderError
		var catalog *CatalogError
		if !errors.As(err, &failure) || !errors.As(err, &catalog) || catalog.Code != test.code ||
			failure.Class != core.ProviderErrorUpstream || !failure.Classification.FailoverEligible {
			t.Fatalf("%s: err = %v", test.code, err)
		}
	}
}

func TestParseZenModelsReadsTheKeyedCatalogAsTheGatewayDoes(t *testing.T) {
	discoveredAt := time.Date(2026, time.September, 24, 12, 0, 0, 0, time.UTC)
	raw := `{"object":"list","data":[
{"id":"big-fixture","object":"model","created":1750000000,"owned_by":"opencode"},
{"id":"sparse-fixture"},
{"id":" ","name":"named-fixture","vendor":"fixture-labs","owned_by":"opencode"},
{"id":"labelled-fixture","display_name":"Labelled Fixture","name":"ignored"},
{"id":"listed-fixture","supported_endpoints":["/responses"]},
{"id":"null-endpoints-fixture","supported_endpoints":null},
{"id":"both-fixture","supported_endpoints":["/v1/chat/completions","/responses",7]}]}`
	models, err := parseZenModels([]byte(raw), discoveredAt)
	if err != nil {
		t.Fatal(err)
	}
	type row struct {
		id, object, owner, description string
		created                        int64
		endpoints                      []string
	}
	want := []row{
		{"big-fixture", "model", "opencode", "", 1750000000, nil},
		{"sparse-fixture", "model", "", "", 0, nil},
		{"named-fixture", "model", "fixture-labs", "", 0, nil},
		{"labelled-fixture", "model", "", "Labelled Fixture", 0, nil},
		{"listed-fixture", "model", "", "", 0, []string{"/responses"}},
		{"null-endpoints-fixture", "model", "", "", 0, nil},
		{"both-fixture", "model", "", "", 0, []string{"/v1/chat/completions", "/responses"}},
	}
	if len(models) != len(want) {
		t.Fatalf("models = %+v", models)
	}
	for i, model := range models {
		got := row{model.ID, model.Object, model.OwnedBy, model.Description, model.Created, model.SupportedAPIs}
		if got.id != want[i].id || got.object != want[i].object || got.owner != want[i].owner ||
			got.description != want[i].description || got.created != want[i].created || !slices.Equal(got.endpoints, want[i].endpoints) {
			t.Fatalf("row %d = %+v, want %+v", i, got, want[i])
		}
		if len(model.Tags) != 0 || model.Capabilities.Provenance.Source != core.ModelCapabilitySourceInferred ||
			!model.Capabilities.Freshness.DiscoveredAt.Equal(discoveredAt) {
			t.Fatalf("row %d = %+v", i, model)
		}
	}
	if caps := models[0].Capabilities; caps.Operations.Chat != core.SupportUnknown || caps.Surfaces.ChatCompletions != core.SupportUnknown {
		t.Fatalf("a row without endpoints claims a surface: %+v", caps)
	}
	if caps := models[6].Capabilities; caps.Surfaces.ChatCompletions != core.SupportSupported || caps.Surfaces.Responses != core.SupportSupported ||
		caps.Operations.Chat != core.SupportSupported || caps.Inputs.Text != core.SupportSupported {
		t.Fatalf("both = %+v", caps)
	}
	if chat, responses := zenRowSurfaces(models[4]); chat || !responses {
		t.Fatalf("listed surfaces = %v %v", chat, responses)
	}
}

func TestParseZenModelsRejectsMalformedCatalogs(t *testing.T) {
	for raw, code := range map[string]string{
		`not json`:                        "invalid_json",
		`[]`:                              "invalid_shape",
		`{"object":"list"}`:               "invalid_shape",
		`{"data":["big-fixture"]}`:        "invalid_shape",
		`{"data":[{"id":7}]}`:             "invalid_shape",
		`{"data":[{"id":"a","name":7}]}`:  "invalid_shape",
		`{"data":[{"id":" ","name":""}]}`: "invalid_shape",
		`{"data":[{"object":"model"}]}`:   "invalid_shape",
	} {
		_, err := parseZenModels([]byte(raw), time.Now())
		var catalog *CatalogError
		if !errors.As(err, &catalog) || catalog.Code != code {
			t.Fatalf("%s: err = %v", raw, err)
		}
	}
	if models, err := parseZenModels([]byte(`{"data":[]}`), time.Now()); err != nil || len(models) != 0 {
		t.Fatalf("empty catalog = %+v, err = %v", models, err)
	}
}
