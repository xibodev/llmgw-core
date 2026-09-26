package providers

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

func TestOllamaListsTheModelsTheDaemonHolds(t *testing.T) {
	t.Parallel()
	provider, calls := ollamaDaemon(t, 200, `{"models":[`+
		`{"name":"llama3.2:latest","model":"llama3.2:latest","details":{"family":"llama","parameter_size":"3.2B"}},`+
		`{"name":" ","model":"qwen3:8b","details":{"family":""}},`+
		`{"model":"fixture-model","details":"unexpected"}]}`)
	models, err := provider.ListModels(t.Context(), &core.Credential{APIKey: "ignored"})
	want := []core.ModelInfo{
		{ID: "llama3.2:latest", Object: "model", OwnedBy: "llama", Vendor: "llama"},
		{ID: "qwen3:8b", Object: "model", OwnedBy: "ollama", Vendor: "ollama"},
		{ID: "fixture-model", Object: "model", OwnedBy: "ollama", Vendor: "ollama"},
	}
	if err != nil || !reflect.DeepEqual(models, want) {
		t.Fatalf("models = %+v, err = %v", models, err)
	}
	if got := calls(); len(got) != 1 || got[0].method != "GET" || got[0].path != "/api/tags" || got[0].body != "" {
		t.Fatalf("upstream = %+v", got)
	}
}

// Ported from the gateway's catalog payload cases for Ollama: a row is named
// by a nonblank name or model, and a name or model of another type rejects
// the whole catalog.
func TestOllamaCatalogIsCheckedAsTheGatewayChecksIt(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, body, code string
		status, count    int
	}{
		{name: "empty", body: `{"models":[]}`, status: 200},
		{name: "name", body: `{"models":[{"name":"fixture:latest"}]}`, status: 200, count: 1},
		{name: "both", body: `{"models":[{"name":"fixture:latest","model":"fixture:latest","future":null}]}`, status: 200, count: 1},
		{name: "empty name", body: `{"models":[{"name":"","model":"fixture:latest"}]}`, status: 200, count: 1},
		{name: "wrong name", body: `{"models":[{"name":false,"model":"fixture:latest"}]}`, code: CatalogCodeInvalidShape, status: 200},
		{name: "wrong model", body: `{"models":[{"name":"fixture:latest","model":17}]}`, code: CatalogCodeInvalidShape, status: 200},
		{name: "unnamed", body: `{"models":[{"details":{}}]}`, code: CatalogCodeInvalidShape, status: 200},
		{name: "no models", body: `{}`, code: CatalogCodeInvalidShape, status: 200},
		{name: "malformed", body: `{"models":[`, code: CatalogCodeInvalidJSON, status: 200},
		{name: "forbidden", body: `{"error":"fixture-secret"}`, code: CatalogCodeAuthenticationFailed, status: 403},
		{name: "unavailable", body: `fixture-secret`, code: CatalogCodeHTTPError, status: 503},
	} {
		provider, _ := ollamaDaemon(t, tc.status, tc.body)
		models, err := provider.ListModels(t.Context(), nil)
		if tc.code == "" {
			if err != nil || models == nil || len(models) != tc.count {
				t.Fatalf("%s: models = %+v, err = %v", tc.name, models, err)
			}
			continue
		}
		if failure := catalogCode(t, err); models != nil || failure.Code != tc.code || failure.Status != tc.status {
			t.Fatalf("%s: models = %+v, failure = %+v", tc.name, models, failure)
		}
	}
}

func TestOllamaCatalogTransportFailure(t *testing.T) {
	t.Parallel()
	closed := httptest.NewServer(http.NotFoundHandler())
	closed.Close()
	provider, err := NewOllama(OllamaConfig{BaseURL: closed.URL})
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	for ctx, want := range map[context.Context]core.ProviderErrorClassification{
		t.Context(): {Retryable: true, FailoverEligible: true, CircuitFailure: true},
		canceled:    {},
	} {
		_, err = provider.ListModels(ctx, nil)
		var failure *core.ProviderError
		if !errors.As(err, &failure) || catalogCode(t, err).Code != CatalogCodeTransportError || failure.Classification != want {
			t.Fatalf("err = %#v, want a transport error classified %+v", err, want)
		}
	}
}
