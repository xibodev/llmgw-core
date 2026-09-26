package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// Ported from the gateway's TestAnthropicListModelsDeclaresMessagesSurface.
func TestAnthropicListModelsDeclaresMessagesSurface(t *testing.T) {
	t.Parallel()
	discoveredAt := time.Date(2026, time.September, 21, 12, 0, 0, 0, time.FixedZone("fixture", 3600))
	backend, server := newAnthropicBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
		_, _ = io.WriteString(w, `{"data":[{"id":"claude-test","type":"model","display_name":"Claude Test","created_at":"2025-01-01T00:00:00Z"},{"id":"claude-plain"}],"has_more":false}`)
	})
	provider := newTestAnthropic(t, server, func(config *AnthropicConfig) {
		config.Now = func() time.Time { return discoveredAt }
	})
	models, err := provider.ListModels(context.Background(), anthropicAPIKey("fixture-key"))
	if err != nil {
		t.Fatal(err)
	}
	want := []core.ModelInfo{
		{
			ID: "claude-test", Object: "model", OwnedBy: "anthropic", Vendor: "anthropic", DisplayName: "Claude Test",
			SupportedAPIs: []string{"/v1/messages"}, Capabilities: anthropicModelCapabilities(discoveredAt.UTC()),
		},
		{
			ID: "claude-plain", Object: "model", OwnedBy: "anthropic", Vendor: "anthropic",
			SupportedAPIs: []string{"/v1/messages"}, Capabilities: anthropicModelCapabilities(discoveredAt.UTC()),
		},
	}
	if !reflect.DeepEqual(models, want) {
		t.Fatalf("models = %+v", models)
	}
	capabilities := models[0].Capabilities
	if capabilities.Operations.Chat != core.SupportSupported || capabilities.Surfaces.Messages != core.SupportSupported ||
		capabilities.Surfaces.ChatCompletions != core.SupportUnsupported || capabilities.Surfaces.Responses != core.SupportUnsupported ||
		capabilities.Streaming != core.SupportSupported || capabilities.Provenance.Source != core.ModelCapabilitySourceRegistryStatic ||
		capabilities.Provenance.Confidence != core.ModelCapabilityConfidenceHigh ||
		!capabilities.Freshness.DiscoveredAt.Equal(discoveredAt) || capabilities.Freshness.DiscoveredAt.Location() != time.UTC ||
		!capabilities.Freshness.ExpiresAt.Equal(discoveredAt.Add(time.Hour)) {
		t.Fatalf("capability evidence = %+v", capabilities)
	}
	if models[0].Capabilities == models[1].Capabilities || models[0].Capabilities.Freshness.DiscoveredAt == models[1].Capabilities.Freshness.DiscoveredAt {
		t.Fatal("rows share their capabilities")
	}
	if calls := backend.take(); len(calls) != 1 || calls[0].method != http.MethodGet || calls[0].path != "/v1/models" || calls[0].body != "" {
		t.Fatalf("upstream = %+v", calls)
	}
}

// A catalog failure keeps the gateway's code, and one the upstream refused
// keeps its status and Retry-After, so a Runtime refreshes or backs off.
func TestAnthropicCatalogFailuresKeepTheGatewayCodes(t *testing.T) {
	t.Parallel()
	for name, check := range map[string]struct {
		status  int
		answer  string
		code    string
		class   core.ProviderErrorClass
		classes core.ProviderErrorClassification
	}{
		"rejected key": {status: http.StatusUnauthorized, code: CatalogCodeAuthenticationFailed, class: core.ProviderErrorAuth,
			classes: core.ProviderErrorClassification{StatusCode: http.StatusUnauthorized}},
		"unavailable": {status: http.StatusServiceUnavailable, code: CatalogCodeHTTPError, class: core.ProviderErrorUpstream,
			classes: core.ProviderErrorClassification{StatusCode: http.StatusServiceUnavailable, Retryable: true, FailoverEligible: true, CircuitFailure: true, RetryAfter: 3 * time.Second}},
		"not json":     {answer: "not json", code: CatalogCodeInvalidJSON, class: core.ProviderErrorUpstream, classes: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"no data":      {answer: `{"models":[]}`, code: CatalogCodeInvalidShape, class: core.ProviderErrorUpstream, classes: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"unnamed row":  {answer: `{"data":[{"id":"claude-fixture"},{"display_name":"No id"}]}`, code: CatalogCodeInvalidShape, class: core.ProviderErrorUpstream, classes: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"oversized":    {answer: `{"data":[{"id":"claude-fixture"}],"private":"fixture-secret"}` + strings.Repeat(" ", catalogMaxResponseBytes), code: CatalogCodeNotDiscoverable, class: core.ProviderErrorUpstream, classes: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"exact bound":  {answer: `{"data":[{"id":"claude-fixture"}]}` + strings.Repeat(" ", catalogMaxResponseBytes-len(`{"data":[{"id":"claude-fixture"}]}`))},
		"unreachable":  {status: -1, code: CatalogCodeTransportError, class: core.ProviderErrorTransport, classes: core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}},
		"empty answer": {answer: `{"data":[]}`},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			_, server := newAnthropicBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
				if check.status > 0 {
					if check.status == http.StatusServiceUnavailable {
						w.Header().Set("Retry-After", "3")
					}
					w.WriteHeader(check.status)
				}
				_, _ = io.WriteString(w, check.answer)
			})
			provider := newTestAnthropic(t, server)
			if check.status < 0 {
				server.Close()
			}
			models, err := provider.ListModels(context.Background(), nil)
			if check.code == "" {
				if err != nil || models == nil {
					t.Fatalf("models = %v, err = %v", models, err)
				}
				return
			}
			failure := assertAnthropicFailure(t, err, check.class, check.classes)
			var catalog *CatalogError
			if !errors.As(err, &catalog) || catalog.Code != check.code || strings.Contains(failure.Error(), "fixture-secret") {
				t.Fatalf("catalog error = %#v", catalog)
			}
		})
	}
}
