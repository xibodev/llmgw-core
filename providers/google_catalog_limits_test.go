package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// Ported from the gateway's TestGoogleCatalogPaginationLimits. The walk is
// bounded as a whole, duplicates and filtered rows included, and each
// answer on its own; neither a partial catalog nor upstream data escapes a
// limit. Not parallel: some cases send the whole size bound.
func TestGoogleCatalogPaginationLimits(t *testing.T) {
	for _, surface := range []struct {
		name, basePath, path, field, publisher, pageSize string
	}{
		{"studio-v1", "/v1", "/v1/models", "models", "", "1000"},
		{"studio-v1beta", "/v1beta", "/v1beta/models", "models", "", "1000"},
		{"vertex-google", "/v1", "/v1beta1/publishers/google/models", "publisherModels", "google", "200"},
		{"vertex-publisher-proxy", "/proxy/v1", "/proxy/v1beta1/publishers/fixture/models", "publisherModels", "fixture", "200"},
	} {
		t.Run(surface.name, func(t *testing.T) {
			for _, tc := range []struct {
				name                          string
				pages, rows, lastRows, count  int
				endless, duplicates, filtered bool
				bodySize                      int
				failure                       string
			}{
				{name: "unique-token-loop", pages: googleCatalogMaxPages, rows: 1, lastRows: 1, endless: true, failure: "page"},
				{name: "empty-unique-token-loop", pages: googleCatalogMaxPages, endless: true, failure: "page"},
				{name: "complete-at-page-limit", pages: googleCatalogMaxPages, rows: 1, lastRows: 1, count: googleCatalogMaxPages},
				{name: "complete-at-model-limit", pages: 2, rows: googleCatalogMaxModels / 2, lastRows: googleCatalogMaxModels / 2, count: googleCatalogMaxModels},
				{name: "accumulated-model-overflow", pages: 3, rows: googleCatalogMaxModels / 2, lastRows: 1, failure: "model"},
				{name: "duplicate-model-overflow", pages: 3, rows: googleCatalogMaxModels / 2, lastRows: 1, duplicates: true, failure: "model"},
				{name: "filtered-model-overflow", pages: 3, rows: googleCatalogMaxModels / 2, lastRows: 1, filtered: true, failure: "model"},
				{name: "single-page-model-overflow", pages: 1, lastRows: googleCatalogMaxModels + 1, failure: "model"},
				{name: "complete-with-duplicates", pages: 2, rows: 1, lastRows: 1, duplicates: true, count: 2},
				{name: "complete-at-body-limit", pages: 1, lastRows: 1, bodySize: catalogMaxResponseBytes, count: 1},
				{name: "oversized-later-body", pages: 2, rows: 1, lastRows: 1, bodySize: catalogMaxResponseBytes + 1, failure: "size"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					var calls atomic.Int32
					token := func(page int) string { return fmt.Sprintf("fixture-private-token +/&=%d", page) }
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						page := int(calls.Add(1))
						if r.URL.Path != surface.path || page > tc.pages {
							t.Errorf("unexpected request: path=%s page=%d", r.URL.Path, page)
							http.Error(w, "fixture-private-body", http.StatusInternalServerError)
							return
						}
						wantToken := ""
						if page > 1 {
							wantToken = token(page - 1)
						}
						if r.URL.Query().Get("pageToken") != wantToken || r.URL.Query().Get("pageSize") != surface.pageSize {
							t.Error("pagination query changed")
						}
						if surface.publisher == "" {
							if r.Header.Get("x-goog-api-key") != "fixture-private-key" || r.Header.Get("Authorization") != "" {
								t.Error("AI Studio authentication changed")
							}
						} else if r.Header.Get("Authorization") != "Bearer fixture-private-bearer" || r.Header.Get("x-goog-api-key") != "" {
							t.Error("Vertex authentication changed")
						}
						n := tc.rows
						if page == tc.pages {
							n = tc.lastRows
						}
						rows := make([]map[string]any, n)
						for i := range rows {
							id := fmt.Sprintf("gemini-fixture-%d-%d", page, i)
							if tc.duplicates {
								id = "gemini-fixture-duplicate"
							}
							row := map[string]any{"name": "models/" + id}
							if surface.publisher == "" {
								if !tc.filtered {
									row["supportedGenerationMethods"] = []string{"generateContent"}
								}
							} else {
								row["name"] = "publishers/" + surface.publisher + "/models/" + id
								if !tc.filtered {
									row["supportedActions"] = map[string]any{"requestAccess": map[string]any{}}
								}
							}
							rows[i] = row
						}
						body := map[string]any{surface.field: rows, "private": "fixture-private-body"}
						if page < tc.pages || tc.endless {
							body["nextPageToken"] = token(page)
						}
						raw, err := json.Marshal(body)
						if err != nil {
							t.Error(err)
							return
						}
						w.Header().Set("Content-Type", "application/json")
						_, _ = w.Write(raw)
						if page == tc.pages && tc.bodySize > len(raw) {
							_, _ = fmt.Fprint(w, strings.Repeat(" ", tc.bodySize-len(raw)))
						}
					}))
					defer server.Close()
					models, err := googleLimitWalk(t, server.URL+surface.basePath, surface.publisher)
					if int(calls.Load()) != tc.pages {
						t.Fatalf("requests=%d, want %d", calls.Load(), tc.pages)
					}
					if tc.failure == "" {
						if err != nil || len(models) != tc.count {
							t.Fatalf("complete catalog: models=%d err=%v, want %d", len(models), err, tc.count)
						}
						return
					}
					wantStatus := 0 // Local walk ceilings are not upstream HTTP failures.
					if tc.failure == "size" {
						wantStatus = http.StatusOK
					}
					code, detail, status := googleCatalogCode(t, err)
					if models != nil || code != CatalogCodeNotDiscoverable || status != wantStatus ||
						detail != "Provider catalog "+map[string]string{"page": "listing", "model": "listing", "size": "response"}[tc.failure]+" exceeded the "+tc.failure+" limit." {
						t.Fatalf("limit failure: models=%d error=%v code=%s status=%d", len(models), err, code, status)
					}
					if classification := core.ClassifyError(err); classification.Retryable || !classification.FailoverEligible {
						t.Fatalf("classification = %+v, want failover without a retry", classification)
					}
					if strings.Contains(err.Error(), "fixture-private") || strings.Contains(detail, "fixture-private") {
						t.Fatal("limit failure disclosed upstream data")
					}
				})
			}
		})
	}
}

// googleLimitWalk lists a catalog: AI Studio's with an API key, and Vertex
// AI's with a bearer, through its own publisher when it names one.
func googleLimitWalk(t *testing.T, base, publisher string) ([]core.ModelInfo, error) {
	t.Helper()
	if publisher == "" {
		provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
		return provider.ListModels(context.Background(), googleKey("fixture-private-key"))
	}
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base, Project: "fixture-project", Location: "global"})
	if publisher == "google" {
		return provider.ListModels(context.Background(), googleBearer("fixture-private-bearer"))
	}
	return provider.vertexPublisherModels(context.Background(), googleAuthorization{name: "Authorization", value: "Bearer fixture-private-bearer"}, publisher)
}
