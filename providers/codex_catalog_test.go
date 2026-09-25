package providers

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

// The fixtures are synthetic rows shaped like the catalogs Codex serves:
// the current models envelope keyed by slug, which omits
// supported_endpoints, and the older data envelope keyed by id.
func TestCodexCatalogFixturesNormalizeToUsableModels(t *testing.T) {
	t.Parallel()
	discoveredAt := time.Date(2026, time.September, 21, 12, 0, 0, 0, time.UTC)
	for fixture, golden := range map[string]string{
		"catalog-current.json":                "catalog-current.golden.json",
		"catalog-data-without-endpoints.json": "catalog-data-without-endpoints.golden.json",
		"catalog-data.json":                   "",
	} {
		t.Run(fixture, func(t *testing.T) {
			t.Parallel()
			models, err := parseCodexModels(readCodexFixture(t, fixture), discoveredAt)
			if err != nil {
				t.Fatal(err)
			}
			if golden == "" {
				if len(models) != 1 || !reflect.DeepEqual(models[0].SupportedAPIs, []string{"/responses"}) {
					t.Fatalf("models = %+v, want the listed endpoint kept", models)
				}
				return
			}
			got, err := json.Marshal(models)
			if err != nil {
				t.Fatal(err)
			}
			assertSameJSON(t, got, readCodexFixture(t, golden))
		})
	}
}

func TestCodexCatalogWithoutUsableRowsIsEmptyNotAnError(t *testing.T) {
	t.Parallel()
	models, err := parseCodexModels(readCodexFixture(t, "catalog-models.json"), time.Now())
	if err != nil || models == nil || len(models) != 0 {
		t.Fatalf("models = %#v err = %v, want an empty catalog", models, err)
	}
}

func TestCodexCatalogNormalizationKeepsTheParsersStrictness(t *testing.T) {
	t.Parallel()
	for name, fixture := range map[string]string{
		"ambiguous envelope": `{"data":[],"models":[]}`,
		"inexact identity":   `{"models":[{"slug":" padded ","supported_in_api":true,"visibility":"list"}]}`,
		"wrong eligibility":  `{"models":[{"slug":"exact","supported_in_api":"yes","visibility":"list"}]}`,
		"duplicate identity": `{"models":[{"slug":"twin","supported_in_api":true,"visibility":"list"},{"slug":"twin"}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if models, err := parseCodexModels([]byte(fixture), time.Now()); err == nil {
				t.Fatalf("malformed catalog accepted: %+v", models)
			}
		})
	}
}

// CodexProvider.ListModels returns every row as reported, so products that
// filter and default rows themselves see what they always saw.
func TestCodexLegacyCatalogStillReturnsEveryRowAsReported(t *testing.T) {
	t.Parallel()
	models, err := parseCodexCatalog(readCodexFixture(t, "catalog-current.json"), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 6 {
		t.Fatalf("legacy catalog = %d rows, want all 6", len(models))
	}
	for _, model := range models {
		if model.SupportedAPIs != nil {
			t.Fatalf("legacy row %q gained endpoints %v", model.ID, model.SupportedAPIs)
		}
	}
}

func assertSameJSON(t *testing.T, got, want []byte) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		var indented bytes.Buffer
		_ = json.Indent(&indented, got, "", "  ")
		t.Fatalf("JSON drifted from the fixture:\n got: %s\nwant: %s", indented.String(), want)
	}
}
