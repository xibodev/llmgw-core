package providers_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"testing"

	"github.com/xibodev/llmgw-core/providers"
)

func TestAdmitAnonymousModelAppliesReviewedRules(t *testing.T) {
	t.Parallel()
	text := []any{"text"}
	free := map[string]any{"prompt": "0", "completion": 0.0}
	cases := []struct {
		name     string
		registry string
		row      map[string]any
		want     providers.AnonymousAdmission
	}{
		{"kilo free text model", "kilo_code", map[string]any{"isFree": true, "pricing": free, "architecture": map[string]any{"output_modalities": text}, "context_length": 131072.0},
			providers.AnonymousAdmission{Free: true, ContextWindow: 131072}},
		{"kilo priced model", "kilo_code", map[string]any{"isFree": true, "pricing": map[string]any{"prompt": "0.1", "completion": "0"}, "architecture": map[string]any{"output_modalities": text}},
			providers.AnonymousAdmission{}},
		{"kilo image-only model", "kilo_code", map[string]any{"isFree": true, "pricing": free, "architecture": map[string]any{"output_modalities": []any{"image"}}},
			providers.AnonymousAdmission{}},
		{"kilo without the free flag", "kilo_code", map[string]any{"pricing": free, "architecture": map[string]any{"output_modalities": text}},
			providers.AnonymousAdmission{}},
		{"llm7 turbo chat model", "llm7", map[string]any{"tier": "Turbo", "model_type": "chat", "schema_endpoints": []any{"openai"}, "usage_based_only": false, "context_length": 32000.0},
			providers.AnonymousAdmission{Free: true, ContextWindow: 32000}},
		{"llm7 usage-based model", "llm7", map[string]any{"tier": "turbo", "model_type": "chat", "schema_endpoints": []any{"openai"}, "usage_based_only": true},
			providers.AnonymousAdmission{}},
		{"llm7 unknown usage policy", "llm7", map[string]any{"tier": "turbo", "model_type": "chat", "schema_endpoints": []any{"openai"}},
			providers.AnonymousAdmission{}},
		{"llm7 without the OpenAI schema", "llm7", map[string]any{"tier": "turbo", "model_type": "chat", "schema_endpoints": []any{"anthropic"}, "usage_based_only": false},
			providers.AnonymousAdmission{}},
		{"ovh bounded free model", "ovh_ai_endpoints", map[string]any{"context_length": 8192.0, "max_completion_tokens": 4096.0, "pricing": free},
			providers.AnonymousAdmission{Free: true, ContextWindow: 8192}},
		{"ovh model without limits", "ovh_ai_endpoints", map[string]any{"context_length": 8192.0, "pricing": free},
			providers.AnonymousAdmission{ContextWindow: 8192}},
		{"pollinations anonymous text model", "pollinations", map[string]any{"tier": "anonymous", "output_modalities": text, "reasoning": true, "tools": true},
			providers.AnonymousAdmission{Free: true, Reasoning: true, ToolCalls: true}},
		{"pollinations seed tier", "pollinations", map[string]any{"tier": "seed", "output_modalities": text},
			providers.AnonymousAdmission{}},
		{"zen raw row", "opencode_zen", map[string]any{"id": "ling-3.0-flash-fin-free"},
			providers.AnonymousAdmission{}},
		{"unknown provider", "custom_openai", map[string]any{"isFree": true},
			providers.AnonymousAdmission{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := providers.AdmitAnonymousModel(tc.registry, tc.row); got != tc.want {
				t.Fatalf("admission=%+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestAdmitAnonymousModelRejectsMalformedPrices(t *testing.T) {
	t.Parallel()
	row := func(price any) map[string]any {
		return map[string]any{"context_length": 8192.0, "max_completion_tokens": 4096.0, "pricing": map[string]any{"prompt": price, "completion": price}}
	}
	for _, price := range []any{"0", "0.000", 0.0} {
		if !providers.AdmitAnonymousModel("ovh_ai_endpoints", row(price)).Free {
			t.Fatalf("valid zero price %#v rejected", price)
		}
	}
	for _, price := range []any{"", ".", "...", "0..0", "1", 1.0, nil, true} {
		if providers.AdmitAnonymousModel("ovh_ai_endpoints", row(price)).Free {
			t.Fatalf("malformed or nonzero price %#v admitted", price)
		}
	}
}

func TestSelectVerificationModelPrefersReviewedDefaults(t *testing.T) {
	t.Parallel()
	if got := providers.SelectVerificationModel("kilo_code", []string{"zeta:free", "COHERE/NORTH-MINI-CODE:FREE", "alpha:free"}); got != "COHERE/NORTH-MINI-CODE:FREE" {
		t.Fatalf("preferred default=%q", got)
	}
	if got := providers.SelectVerificationModel("kilo_code", []string{"zeta:free", "alpha:free"}); got != "alpha:free" {
		t.Fatalf("fallback=%q, want the lexically first free id", got)
	}
	if got := providers.SelectVerificationModel("kilo_code", nil); got != "" {
		t.Fatalf("empty=%q", got)
	}
	for _, profile := range providers.AnonymousProviderProfiles() {
		if len(profile.VerificationModels) == 0 {
			t.Fatalf("%s has no reviewed verification models", profile.RegistryID)
		}
	}
}

func TestDiscoverAnonymousPollinationsAdmitsOnlyAnonymousTextModels(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			http.NotFound(w, r)
			return
		}
		_, _ = fmt.Fprint(w, `[
			{"name":"openai-fast","tier":"anonymous","output_modalities":["text"],"description":"Fast"},
			{"name":"flux","tier":"anonymous","output_modalities":["image"]},
			{"name":"openai-large","tier":"seed","output_modalities":["text"]}
		]`)
	}))
	defer server.Close()
	models, err := providers.DiscoverAnonymousModels(context.Background(), providers.AnonymousProviderProfile{
		RegistryID: "pollinations", ProviderID: "pollinations", BaseURL: server.URL,
	}, server.Client())
	if err != nil || len(models) != 1 || models[0].ID != "openai-fast" || models[0].Description != "Fast" {
		t.Fatalf("models=%+v err=%v", models, err)
	}
}

func TestDiscoverAnonymousModelsFailsClosed(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/unknown/models":
			_, _ = fmt.Fprint(w, `{"data":[{"id":"anything","isFree":true}]}`)
		default:
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
		}
	}))
	defer server.Close()
	models, err := providers.DiscoverAnonymousModels(context.Background(), providers.AnonymousProviderProfile{
		RegistryID: "custom_openai", ProviderID: "custom", BaseURL: server.URL + "/unknown",
	}, server.Client())
	if err != nil || len(models) != 0 {
		t.Fatalf("an unreviewed provider admitted models=%+v err=%v", models, err)
	}
	if _, err := providers.DiscoverAnonymousModels(context.Background(), providers.AnonymousProviderProfile{
		RegistryID: "kilo_code", ProviderID: "kilo", BaseURL: server.URL + "/down",
	}, server.Client()); err == nil {
		t.Fatal("a failing catalog was not reported")
	}
}

// redirectTransport sends every request to one test server, keeping its path,
// so discovery that also reads a fixed metadata URL stays hermetic.
type redirectTransport struct{ target *url.URL }

func (rt redirectTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	redirected := request.Clone(request.Context())
	redirected.URL.Scheme, redirected.URL.Host, redirected.Host = rt.target.Scheme, rt.target.Host, rt.target.Host
	return http.DefaultTransport.RoundTrip(redirected)
}

func TestDiscoverAnonymousZenUsesVerifiedMetadata(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/zen/models":
			_, _ = fmt.Fprint(w, `{"data":[{"id":"ling-3.0-flash-fin-free"},{"id":"paid-model"},{"id":"unlisted-model"}]}`)
		case "/api.json":
			_, _ = fmt.Fprint(w, `{"opencode":{"npm":"@ai-sdk/openai-compatible","models":{
				"ling-3.0-flash-fin-free":{"id":"ling-3.0-flash-fin-free","status":"active","cost":{"input":0,"output":0}},
				"paid-model":{"id":"paid-model","status":"active","cost":{"input":0,"output":1}}
			}}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	target, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: redirectTransport{target: target}}
	models, err := providers.DiscoverAnonymousModels(context.Background(), providers.AnonymousProviderProfile{
		RegistryID: "opencode_zen", ProviderID: "opencode_zen", BaseURL: server.URL + "/zen",
	}, client)
	if err != nil {
		t.Fatal(err)
	}
	ids := []string{}
	for _, model := range models {
		ids = append(ids, model.ID)
		if model.OwnedBy != "opencode_zen" {
			t.Fatalf("model %s owned by %q", model.ID, model.OwnedBy)
		}
	}
	if !slices.Equal(ids, []string{"ling-3.0-flash-fin-free"}) {
		t.Fatalf("models=%v, want only the verified zero-cost model", ids)
	}
}
