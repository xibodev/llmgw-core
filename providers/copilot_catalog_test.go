package providers

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

var copilotDiscoveredAt = time.Date(2026, time.September, 21, 12, 0, 0, 0, time.UTC)

func readCopilotFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "copilot", name))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// The fixtures are synthetic rows shaped like the catalogs Copilot serves
// now: newer rows list supported_endpoints and reasoning efforts, older and
// embedding rows omit them, and every optional field may be missing.
func TestCopilotCatalogFixturesNormalizeAsTheGatewayDerives(t *testing.T) {
	t.Parallel()
	for fixture, golden := range map[string]string{
		"catalog-current.json": "catalog-current.golden.json",
		"catalog-sparse.json":  "catalog-sparse.golden.json",
	} {
		t.Run(fixture, func(t *testing.T) {
			t.Parallel()
			models, err := parseCopilotCatalog(readCopilotFixture(t, fixture), copilotDiscoveredAt)
			if err != nil {
				t.Fatal(err)
			}
			infos := make([]core.ModelInfo, len(models))
			for index, model := range models {
				infos[index] = model.info
			}
			got, err := json.Marshal(infos)
			if err != nil {
				t.Fatal(err)
			}
			assertSameJSON(t, got, readCopilotFixture(t, golden))
		})
	}
}

// A nonempty reasoning_effort list, whatever it holds, is what makes the
// gateway rename max_tokens, while only string levels report reasoning.
func TestCopilotCatalogRecordsReasoningEffortListsForTheTransport(t *testing.T) {
	t.Parallel()
	models, err := parseCopilotCatalog([]byte(`{"data":[
		{"id":"levels","capabilities":{"supports":{"reasoning_effort":["low"]}}},
		{"id":"opaque","capabilities":{"supports":{"reasoning_effort":[1]}}},
		{"id":"empty","capabilities":{"supports":{"reasoning_effort":[]}}},
		{"id":"absent","capabilities":{"supports":{}}}
	]}`), copilotDiscoveredAt)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]struct {
		rename    bool
		reasoning core.Support
	}{
		"levels": {true, core.SupportSupported},
		"opaque": {true, core.SupportUnknown},
		"empty":  {false, core.SupportUnknown},
		"absent": {false, core.SupportUnknown},
	}
	for _, model := range models {
		expected := want[model.info.ID]
		if model.reasoningEffort != expected.rename || model.info.Capabilities.Reasoning != expected.reasoning {
			t.Errorf("%s: rename %v reasoning %v, want %v %v", model.info.ID, model.reasoningEffort,
				model.info.Capabilities.Reasoning, expected.rename, expected.reasoning)
		}
	}
}

func TestCopilotCatalogWithoutRowsIsEmptyNotAnError(t *testing.T) {
	t.Parallel()
	models, err := parseCopilotCatalog([]byte(`{"data":[],"object":"list"}`), copilotDiscoveredAt)
	if err != nil || models == nil || len(models) != 0 {
		t.Fatalf("models = %#v err = %v, want an empty catalog", models, err)
	}
}

func TestCopilotCatalogNormalizationKeepsTheGatewaysStrictness(t *testing.T) {
	t.Parallel()
	for name, fixture := range map[string]string{
		"not JSON":               `not-json`,
		"not an object":          `[{"id":"model"}]`,
		"no data":                `{"object":"list"}`,
		"data not an array":      `{"data":{"id":"model"}}`,
		"row not an object":      `{"data":["model"]}`,
		"id not a string":        `{"data":[{"id":7}]}`,
		"null id beside name":    `{"data":[{"id":null,"name":"model"}]}`,
		"name not a string":      `{"data":[{"id":"model","name":false}]}`,
		"no identity":            `{"data":[{"object":"model"}]}`,
		"blank identity":         `{"data":[{"id":" ","name":""}]}`,
		"later row malformed":    `{"data":[{"id":"model"},{"id":["model"]}]}`,
		"later row identityless": `{"data":[{"id":"model"},{}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if models, err := parseCopilotCatalog([]byte(fixture), copilotDiscoveredAt); err == nil {
				t.Fatalf("malformed catalog accepted: %+v", models)
			}
		})
	}
}
