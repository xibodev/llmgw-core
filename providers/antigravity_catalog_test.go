package providers

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

func readAntigravityFixture(t *testing.T, path ...string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(append([]string{"testdata"}, path...)...))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// The fixtures are synthetic catalogs shaped like fetchAvailableModels
// responses: the current one, whose rows carry limits, quotas and
// placeholders beside the capability fields and whose image roster names a
// model the root roster no longer lists, and one whose rows and rosters
// omit every optional field.
func TestAntigravityCatalogFixturesNormalizeToTheGatewaysRows(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"catalog-current", "catalog-omitted-fields"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			models, err := parseAntigravityModels(readAntigravityFixture(t, "antigravity", name+".json"))
			if err != nil {
				t.Fatal(err)
			}
			got, err := json.Marshal(models)
			if err != nil {
				t.Fatal(err)
			}
			assertSameJSON(t, got, readAntigravityFixture(t, "antigravity", name+".golden.json"))
		})
	}
}

// gatewayAntigravityRows derives rows from the legacy adapter's as the
// gateway's ListModelsWithError does: no model streams, and a model whose
// image operation is supported also serves image generation.
func gatewayAntigravityRows(models []core.ModelInfo) []core.ModelInfo {
	for index := range models {
		surfaces := []string{"/v1/chat/completions"}
		if capabilities := models[index].Capabilities; capabilities != nil {
			capabilities.Streaming = core.SupportUnsupported
			if capabilities.Operations.Image == core.SupportSupported {
				surfaces = append(surfaces, "/v1/images/generations")
			}
		}
		models[index].SupportedAPIs = surfaces
	}
	return models
}

// The normalizer moves the gateway's derivation into core, so a gateway
// that adopts it lists the model set and capabilities it lists today.
func TestAntigravityCatalogMatchesWhatTheGatewayDerives(t *testing.T) {
	t.Parallel()
	for _, fixture := range [][]string{
		{"antigravity", "catalog-current.json"}, {"antigravity", "catalog-omitted-fields.json"},
		{"antigravity_catalog_metadata.json"}, {"antigravity_catalog_absent_rosters.json"},
	} {
		raw := readAntigravityFixture(t, fixture...)
		catalog, err := decodeAntigravityCatalog(raw)
		if err != nil {
			t.Fatal(err)
		}
		models, err := parseAntigravityModels(raw)
		if err != nil {
			t.Fatal(err)
		}
		if want := gatewayAntigravityRows(catalog.models()); len(models) == 0 || !reflect.DeepEqual(models, want) {
			t.Fatalf("%s: rows = %+v, want the gateway's %+v", filepath.Join(fixture...), models, want)
		}
	}
}

func TestAntigravityCatalogWithoutModelsIsEmptyNotAnError(t *testing.T) {
	t.Parallel()
	for _, raw := range []string{`{}`, `null`, `{"models":null}`, `{"models":{}}`, `{"models":{" ":{}},"imageGenerationModelIds":["orphan"]}`} {
		if models, err := parseAntigravityModels([]byte(raw)); err != nil || models == nil || len(models) != 0 {
			t.Fatalf("%s: models = %#v, err = %v, want an empty catalog", raw, models, err)
		}
	}
}

// The catalog decodes as the legacy adapter always decoded it, so a shape
// it refused is still refused rather than listed as something else.
func TestAntigravityCatalogRejectsMalformedShapes(t *testing.T) {
	t.Parallel()
	for name, raw := range map[string]string{
		"not JSON":             `models`,
		"empty body":           ``,
		"models as a list":     `{"models":[{"id":"fixture"}]}`,
		"row as a string":      `{"models":{"fixture":"row"}}`,
		"roster as a string":   `{"models":{"fixture":{}},"imageGenerationModelIds":"fixture"}`,
		"thinking as a string": `{"models":{"fixture":{"supportsThinking":"yes"}}}`,
	} {
		if models, err := parseAntigravityModels([]byte(raw)); err == nil {
			t.Fatalf("%s: malformed catalog accepted: %+v", name, models)
		}
	}
}
