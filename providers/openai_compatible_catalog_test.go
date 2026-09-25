package providers

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// openAICatalogProvider lists the catalog body answers with status.
func openAICatalogProvider(t *testing.T, status int, body string, adjust func(*OpenAICompatibleConfig)) (*OpenAICompatible, *openAIBackend) {
	t.Helper()
	backend, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	})
	return newTestOpenAICompatible(t, server, adjust), backend
}

// Ported from the gateway's TestOpenAIListModelsPrefersDisplayName.
func TestOpenAICompatibleCatalogPrefersTheDisplayName(t *testing.T) {
	t.Parallel()
	provider, backend := openAICatalogProvider(t, http.StatusOK, `{"data":[{"id":"atlas-small","owned_by":"demo","display_name":"Atlas Small"},`+
		`{"id":"","name":"named","vendor":"vendor","owned_by":"owner","created":1700000000},{"id":"same","name":"same"}]}`, nil)
	models, err := provider.ListModels(context.Background(), &core.Credential{APIKey: "fixture-key"})
	if err != nil || len(models) != 3 {
		t.Fatalf("models = %+v, err = %v", models, err)
	}
	if m := models[0]; m.ID != "atlas-small" || m.DisplayName != "Atlas Small" || m.Vendor != "demo" || m.OwnedBy != "demo" || m.Object != "model" {
		t.Fatalf("display name row = %+v", m)
	}
	if m := models[1]; m.ID != "named" || m.DisplayName != "" || m.Vendor != "vendor" || m.Created != 1700000000 {
		t.Fatalf("named row = %+v", m)
	}
	if m := models[2]; m.DisplayName != "" || m.SupportedAPIs != nil || m.LegacyCapabilities != nil {
		t.Fatalf("bare row = %+v", m)
	}
	if calls := backend.take(); len(calls) != 1 || calls[0].method != http.MethodGet || calls[0].path != "/v1/models" ||
		calls[0].authorization != "Bearer fixture-key" || calls[0].body != "" {
		t.Fatalf("upstream = %+v", calls)
	}
}

// A capabilities block, as Copilot's catalog reports one, is distilled as
// the gateway distills it, and the typed capabilities are what the gateway
// infers from the row when it stores it.
func TestOpenAICompatibleCatalogDistillsCapabilities(t *testing.T) {
	t.Parallel()
	discoveredAt := time.Date(2026, 9, 1, 12, 0, 0, 0, time.FixedZone("fixture", 3600))
	provider, _ := openAICatalogProvider(t, http.StatusOK, `{"data":[{"id":"gpt-fixture","supported_endpoints":["/chat/completions","/responses",17],`+
		`"capabilities":{"family":"gpt-fixture","type":"chat","limits":{"max_prompt_tokens":64000,"max_output_tokens":4096.9},`+
		`"supports":{"streaming":true,"tool_calls":true,"vision":false,"structured_outputs":"yes","reasoning_effort":["low",3,"high"]}}},`+
		`{"id":"empty-block","capabilities":{"supports":{"reasoning_effort":[]}}}]}`,
		func(config *OpenAICompatibleConfig) { config.Now = func() time.Time { return discoveredAt } })
	models, err := provider.ListModels(context.Background(), nil)
	if err != nil || len(models) != 2 {
		t.Fatalf("models = %+v, err = %v", models, err)
	}
	want := map[string]any{
		"family": "gpt-fixture", "context_window": 64000, "max_output_tokens": 4096,
		"streaming": true, "tool_calls": true, "vision": false, "reasoning_effort": []string{"low", "high"},
	}
	row := models[0]
	if !reflect.DeepEqual(row.LegacyCapabilities, want) || !slices.Equal(row.SupportedAPIs, []string{"/chat/completions", "/responses"}) {
		t.Fatalf("row = %+v", row)
	}
	typed := row.Capabilities
	if typed == nil || !reflect.DeepEqual(typed, core.InferCapabilities(row, discoveredAt, time.Time{})) ||
		typed.Surfaces.ChatCompletions != core.SupportSupported || typed.Surfaces.Responses != core.SupportSupported ||
		typed.Inputs.Image != core.SupportUnsupported || typed.Tools != core.SupportSupported || typed.Reasoning != core.SupportSupported ||
		*typed.Limits.ContextTokens != 64000 || *typed.Limits.MaxOutputTokens != 4096 || typed.Freshness.ExpiresAt != nil ||
		!typed.Freshness.DiscoveredAt.Equal(discoveredAt) || typed.Provenance.Source != core.ModelCapabilitySourceInferred {
		t.Fatalf("typed capabilities = %+v", typed)
	}
	if empty := models[1]; empty.LegacyCapabilities != nil || empty.Capabilities.Operations.Chat != core.SupportUnknown {
		t.Fatalf("empty block = %+v", empty)
	}
}

// Ported from the gateway's TestOpenAIListModelsReportsSafeFailures: this
// catalog reports every refusal, 401 and 403 included, as http_error with
// its status, so a Runtime still refreshes a credential rejected with 401.
func TestOpenAICompatibleCatalogReportsSafeFailures(t *testing.T) {
	t.Parallel()
	for status, class := range map[int]core.ProviderErrorClass{
		http.StatusUnauthorized: core.ProviderErrorAuth, http.StatusForbidden: core.ProviderErrorForbidden,
		http.StatusTooManyRequests: core.ProviderErrorRateLimited, http.StatusBadGateway: core.ProviderErrorUpstream,
	} {
		provider, _ := openAICatalogProvider(t, status, `{"error":{"message":"fixture-secret catalog unavailable"}}`, nil)
		_, err := provider.ListModels(context.Background(), &core.Credential{APIKey: "fixture-key"})
		failure := catalogCode(t, err)
		if failure.Code != CatalogCodeHTTPError || failure.Status != status || failure.Detail != fmt.Sprintf("Provider catalog returned HTTP %d.", status) ||
			core.ClassifyError(err).StatusCode != status || errors.Unwrap(err) != failure {
			t.Fatalf("%d: failure = %+v", status, failure)
		}
		var providerErr *core.ProviderError
		if errors.As(err, &providerErr); providerErr.Class != class || strings.Contains(err.Error(), "fixture") {
			t.Fatalf("%d: err = %#v", status, err)
		}
	}
	provider, _ := openAICatalogProvider(t, http.StatusOK, "not-json", nil)
	if _, err := provider.ListModels(context.Background(), nil); catalogCode(t, err).Code != CatalogCodeInvalidJSON {
		t.Fatalf("invalid JSON: err = %v", err)
	}
	provider, _ = openAICatalogProvider(t, http.StatusOK, `{"data":[]}`, nil)
	if models, err := provider.ListModels(context.Background(), nil); err != nil || models == nil || len(models) != 0 {
		t.Fatalf("empty catalog: models = %#v, err = %v", models, err)
	}
	_, server := newOpenAIBackend(t, answerOpenAIChat)
	unreachable := newTestOpenAICompatible(t, server, nil)
	server.Close()
	_, err := unreachable.ListModels(context.Background(), nil)
	if catalogCode(t, err).Code != CatalogCodeTransportError || !core.ClassifyError(err).Retryable {
		t.Fatalf("unreachable: err = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := unreachable.ListModels(ctx, nil); core.ClassifyError(err) != (core.ProviderErrorClassification{}) {
		t.Fatalf("canceled: err = %v", err)
	}
}

// Ported from the gateway's TestCatalogPayloadValidationAndCache and
// TestCatalogIdentityVariantsAndFiltering, for its OpenAI-compatible
// reader: every row is checked before any is used.
func TestOpenAICompatibleCatalogValidatesEveryRow(t *testing.T) {
	t.Parallel()
	const row, variant = `{"id":"fixture-model","owned_by":"fixture"}`, `{"name":"fixture-model","future":{"nested":[null,17]}}`
	for name, test := range map[string]struct {
		body  string
		code  string
		count int
	}{
		"empty":                {`{"data":[]}`, "", 0},
		"populated":            {`{"data":[` + row + `]}`, "", 1},
		"variant":              {`{"data":[` + variant + `]}`, "", 1},
		"valid-mixed-variants": {`{"data":[` + row + `,` + variant + `]}`, "", 2},
		"empty-id":             {`{"data":[{"id":"","name":"fixture"}]}`, "", 1},
		"both":                 {`{"data":[{"id":"fixture","name":"Fixture","future":{"nested":null}}]}`, "", 1},
		"null-row":             {`{"data":[null]}`, CatalogCodeInvalidShape, 0},
		"string-row":           {`{"data":["fixture-secret"]}`, CatalogCodeInvalidShape, 0},
		"array-row":            {`{"data":[[]]}`, CatalogCodeInvalidShape, 0},
		"missing-identity":     {`{"data":[{"future":"fixture-secret"}]}`, CatalogCodeInvalidShape, 0},
		"null-identity":        {`{"data":[{"id":null}]}`, CatalogCodeInvalidShape, 0},
		"number-identity":      {`{"data":[{"id":17}]}`, CatalogCodeInvalidShape, 0},
		"wrong-id":             {`{"data":[{"id":false,"name":"fixture"}]}`, CatalogCodeInvalidShape, 0},
		"wrong-name":           {`{"data":[{"name":17}]}`, CatalogCodeInvalidShape, 0},
		"blank-identity":       {`{"data":[{"id":"  "}]}`, CatalogCodeInvalidShape, 0},
		"mixed-valid-first":    {`{"data":[` + row + `,null]}`, CatalogCodeInvalidShape, 0},
		"mixed-invalid-first":  {`{"data":[{"id":false},` + row + `]}`, CatalogCodeInvalidShape, 0},
		"malformed":            {`{"fixture-private":"fixture-secret",`, CatalogCodeInvalidJSON, 0},
		"trailing-json":        {`{"data":[]} {"extra":true}`, CatalogCodeInvalidJSON, 0},
		"missing":              {`{}`, CatalogCodeInvalidShape, 0},
		"array-body":           {`["fixture-secret"]`, CatalogCodeInvalidShape, 0},
		"null-array":           {`{"data":null}`, CatalogCodeInvalidShape, 0},
		"object-array":         {`{"data":{}}`, CatalogCodeInvalidShape, 0},
	} {
		provider, _ := openAICatalogProvider(t, http.StatusOK, test.body, nil)
		models, err := provider.ListModels(context.Background(), &core.Credential{APIKey: "fixture-key"})
		if test.code == "" {
			if err != nil || models == nil || len(models) != test.count {
				t.Fatalf("%s: models = %+v, err = %v", name, models, err)
			}
			continue
		}
		if failure := catalogCode(t, err); models != nil || failure.Code != test.code || failure.Status != http.StatusOK ||
			strings.Contains(err.Error(), "fixture-secret") || strings.Contains(err.Error(), "fixture-key") {
			t.Fatalf("%s: models = %+v, err = %v", name, models, err)
		}
	}
}
