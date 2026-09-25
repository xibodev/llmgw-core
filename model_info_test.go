package core_test

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// modelInfoM4 is ModelInfo as the M4 verticals shipped it: the same fields,
// types and tags, in the same order.
type modelInfoM4 struct {
	ID            string                  `json:"id"`
	Object        string                  `json:"object"`
	Created       int64                   `json:"created"`
	OwnedBy       string                  `json:"owned_by"`
	Description   string                  `json:"description,omitempty"`
	Tags          []string                `json:"tags,omitempty"`
	APIEligible   *bool                   `json:"api_eligible,omitempty"`
	APIVisibility string                  `json:"api_visibility,omitempty"`
	SupportedAPIs []string                `json:"supported_apis,omitempty"`
	Capabilities  *core.ModelCapabilities `json:"capabilities,omitempty"`
}

func (m modelInfoM4) current() core.ModelInfo {
	return core.ModelInfo{
		ID: m.ID, Object: m.Object, Created: m.Created, OwnedBy: m.OwnedBy, Description: m.Description,
		Tags: m.Tags, APIEligible: m.APIEligible, APIVisibility: m.APIVisibility,
		SupportedAPIs: m.SupportedAPIs, Capabilities: m.Capabilities,
	}
}

// Every field keeps its M4 place, type and tag, and every field after them
// is omitted when empty, so no row an M4 vertical builds changes its JSON.
func TestModelInfoKeepsTheM4FieldsAndOmitsEveryLaterOneWhenEmpty(t *testing.T) {
	m4, current := reflect.TypeFor[modelInfoM4](), reflect.TypeFor[core.ModelInfo]()
	if current.NumField() <= m4.NumField() {
		t.Fatalf("ModelInfo has %d fields, want more than M4's %d", current.NumField(), m4.NumField())
	}
	for index := range current.NumField() {
		field := current.Field(index)
		if index < m4.NumField() {
			if want := m4.Field(index); field.Name != want.Name || field.Type != want.Type || field.Tag != want.Tag {
				t.Errorf("field %d is %s %v %q, M4 had %s %v %q", index, field.Name, field.Type, field.Tag, want.Name, want.Type, want.Tag)
			}
			continue
		}
		name, options, _ := strings.Cut(field.Tag.Get("json"), ",")
		if name == "" || name == "-" || !slices.Contains(strings.Split(options, ","), "omitempty") {
			t.Errorf("field %s has tag %q; a field after M4's must be omitted when empty", field.Name, field.Tag)
		}
	}
}

func TestModelInfoWithoutTheM5FieldsEncodesAsM4Did(t *testing.T) {
	eligible := false
	discovered := time.Date(2026, time.September, 20, 10, 0, 0, 0, time.UTC)
	for _, row := range []modelInfoM4{
		{},
		{ID: "fixture-model", Object: "model"},
		{
			ID: "fixture-model", Object: "model", Created: 1_800_000_000, OwnedBy: "fixture",
			Description: "Fixture Model", Tags: []string{"free"}, APIEligible: &eligible, APIVisibility: "list",
			SupportedAPIs: []string{"/chat/completions", "/responses"},
			Capabilities: &core.ModelCapabilities{
				SchemaVersion: core.ModelCapabilitiesSchemaVersion,
				Freshness:     core.ModelCapabilityFreshness{DiscoveredAt: &discovered},
			},
		},
	} {
		want, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		got, err := json.Marshal(row.current())
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != string(want) {
			t.Errorf("ModelInfo JSON changed:\n got: %s\nwant: %s", got, want)
		}
	}

	// Pinned as bytes too, so a change to the mirror cannot hide one.
	got, err := json.Marshal(core.ModelInfo{ID: "fixture-model", Object: "model", OwnedBy: "fixture", Tags: []string{"free"}})
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"id":"fixture-model","object":"model","created":0,"owned_by":"fixture","tags":["free"]}`; string(got) != want {
		t.Fatalf("ModelInfo JSON = %s, want %s", got, want)
	}
}

func TestModelInfoM5FieldsRoundTrip(t *testing.T) {
	row := core.ModelInfo{
		ID: "fixture-model", Object: "model", OwnedBy: "fixture",
		DisplayName: "Fixture Model", Vendor: "fixture-lab", Free: true,
		LegacyCapabilities: map[string]any{"chat": true, "context_window": float64(128000), "reasoning_effort": []any{"low", "high"}},
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"id":"fixture-model","object":"model","created":0,"owned_by":"fixture",` +
		`"display_name":"Fixture Model","vendor":"fixture-lab","free":true,` +
		`"legacy_capabilities":{"chat":true,"context_window":128000,"reasoning_effort":["low","high"]}}`
	if string(encoded) != want {
		t.Fatalf("ModelInfo JSON = %s, want %s", encoded, want)
	}
	var decoded core.ModelInfo
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, row) {
		t.Fatalf("round trip = %#v, want %#v", decoded, row)
	}
}
