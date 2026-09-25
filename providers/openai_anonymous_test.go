package providers

import (
	"context"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// Ported from the gateway's TestAnonymousCatalogProfilesFilterClaimedModels:
// without a key, each anonymous entry lists only the models its reviewed
// rules admit, free and serving Chat Completions; with a key, every row.
func TestOpenAICompatibleAnonymousCatalogsAdmitFreeModels(t *testing.T) {
	t.Parallel()
	for registry, rows := range map[string]string{
		"kilo_code": `{"id":"free","isFree":true,"context_length":8192,"pricing":{"prompt":"0","completion":"0"},"architecture":{"output_modalities":["text"]}},` +
			`{"id":"paid","isFree":false,"pricing":{"prompt":"1","completion":"1"},"architecture":{"output_modalities":["text"]}},` +
			`{"id":"media","isFree":true,"pricing":{"prompt":"0","completion":"0"},"architecture":{"output_modalities":["image"]}}`,
		"llm7": `{"id":"free","tier":"turbo","context_length":8192,"usage_based_only":false,"model_type":"chat","schema_endpoints":["openai"]},` +
			`{"id":"paid","tier":"pro","usage_based_only":true,"model_type":"chat","schema_endpoints":["openai"]},` +
			`{"id":"media","tier":"turbo","usage_based_only":false,"model_type":"image","schema_endpoints":["openai"]}`,
		"ovh_ai_endpoints": `{"id":"free","context_length":8192,"max_completion_tokens":128,"pricing":{"prompt":"0","completion":"0"}},` +
			`{"id":"paid","context_length":1024,"max_completion_tokens":128,"pricing":{"prompt":"1","completion":"1"}},` +
			`{"id":"media","context_length":0,"max_completion_tokens":0,"pricing":{"prompt":"0","completion":"0"}}`,
	} {
		provider, backend := openAICatalogProvider(t, http.StatusOK, `{"data":[`+rows+`]}`, func(config *OpenAICompatibleConfig) { config.RegistryID = registry })
		for _, credential := range []*core.Credential{nil, {APIKey: "free"}, {APIKey: "public"}} {
			models, err := provider.ListModels(context.Background(), credential)
			if err != nil || len(models) != 1 {
				t.Fatalf("%s %v: models = %+v, err = %v", registry, credential, models, err)
			}
			free := models[0]
			if free.ID != "free" || !free.Free || !slices.Equal(free.SupportedAPIs, []string{"/chat/completions"}) ||
				!reflect.DeepEqual(free.LegacyCapabilities, map[string]any{"chat": true, "context_window": 8192}) ||
				free.Capabilities.Operations.Chat != core.SupportSupported || *free.Capabilities.Limits.ContextTokens != 8192 {
				t.Fatalf("%s: admitted row = %+v", registry, free)
			}
		}
		models, err := provider.ListModels(context.Background(), &core.Credential{APIKey: "fixture-key"})
		if err != nil || !slices.Equal(modelIDsOf(models), []string{"free", "paid", "media"}) || models[0].Free || models[0].SupportedAPIs != nil {
			t.Fatalf("%s keyed: models = %+v, err = %v", registry, models, err)
		}
		calls := backend.take()
		if len(calls) != 4 || calls[0].authorization != "" || calls[2].authorization != "Bearer public" || calls[3].authorization != "Bearer fixture-key" {
			t.Fatalf("%s: upstream = %+v", registry, calls)
		}
	}
	// Admission applies only to the anonymous entries.
	provider, _ := openAICatalogProvider(t, http.StatusOK, `{"data":[{"id":"paid","isFree":false}]}`, nil)
	if models, err := provider.ListModels(context.Background(), nil); err != nil || len(models) != 1 || models[0].Free {
		t.Fatalf("generic keyless catalog: models = %+v, err = %v", models, err)
	}
}

// Ported from the gateway's TestPollinationsProfileUsesDistinctPathsAndArrayCatalog
// and TestPollinationsCatalogRejectsNull.
func TestOpenAICompatibleReadsThePollinationsProfile(t *testing.T) {
	t.Parallel()
	const catalog = `[{"name":"openai-fast","tier":"anonymous","output_modalities":["text"],"tools":true,"reasoning":true},` +
		`{"name":"paid","tier":"paid","output_modalities":["TEXT"]},{"name":"image","tier":"anonymous","output_modalities":["image"]}]`
	backend, server := newOpenAIBackend(t, func(w http.ResponseWriter, r *http.Request, _ int) {
		if r.URL.Path == "/models" {
			_, _ = io.WriteString(w, catalog)
			return
		}
		answerOpenAIChat(w, r, 0)
	})
	provider, err := NewOpenAICompatible(OpenAICompatibleConfig{
		BaseURL: server.URL, RegistryID: "pollinations", Client: server.Client(), CatalogClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	models, err := provider.ListModels(context.Background(), nil)
	want := core.ModelInfo{
		ID: "openai-fast", Object: "model", DisplayName: "openai-fast", Free: true, SupportedAPIs: []string{"/chat/completions"},
		LegacyCapabilities: map[string]any{"chat": true, "reasoning": true, "tool_calls": true},
	}
	if err != nil || len(models) != 1 || models[0].Capabilities == nil || models[0].Capabilities.Tools != core.SupportSupported {
		t.Fatalf("anonymous catalog = %+v, err = %v", models, err)
	}
	if models[0].Capabilities = nil; !reflect.DeepEqual(models[0], want) {
		t.Fatalf("row = %+v", models[0])
	}
	keyed, err := provider.ListModels(context.Background(), &core.Credential{APIKey: "fixture-key"})
	if err != nil || !slices.Equal(modelIDsOf(keyed), []string{"openai-fast", "paid"}) || keyed[1].Free {
		t.Fatalf("keyed catalog = %+v, err = %v", keyed, err)
	}
	if _, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "openai-fast", `{"messages":[]}`, nil)); err != nil {
		t.Fatal(err)
	}
	calls := backend.take()
	if len(calls) != 3 || calls[0].path != "/models" || calls[1].path != "/models" || calls[2].path != "/v1/chat/completions" {
		t.Fatalf("upstream = %+v", calls)
	}
}

func TestOpenAICompatiblePollinationsCatalogIsAStrictArray(t *testing.T) {
	t.Parallel()
	for body, code := range map[string]string{
		`null`:        CatalogCodeInvalidJSON,
		`{"data":[]}`: CatalogCodeInvalidJSON,
		`[17]`:        CatalogCodeInvalidJSON,
		`[{"name":"ok","output_modalities":["text"]},null]`: CatalogCodeInvalidShape,
		`[{"name":"  "}]`:                  CatalogCodeInvalidShape,
		`[{"name":17,"tier":"anonymous"}]`: CatalogCodeInvalidShape,
		`[{"name":"fixture-secret"`:        CatalogCodeInvalidJSON,
	} {
		provider, _ := openAICatalogProvider(t, http.StatusOK, body, func(config *OpenAICompatibleConfig) { config.RegistryID = "pollinations" })
		models, err := provider.ListModels(context.Background(), nil)
		if failure := catalogCode(t, err); models != nil || failure.Code != code || strings.Contains(err.Error(), "fixture-secret") {
			t.Fatalf("%s: models = %+v, err = %v", body, models, err)
		}
	}
	provider, _ := openAICatalogProvider(t, http.StatusOK, `[]`, func(config *OpenAICompatibleConfig) { config.RegistryID = "pollinations" })
	if models, err := provider.ListModels(context.Background(), nil); err != nil || models == nil || len(models) != 0 {
		t.Fatalf("empty array: models = %#v, err = %v", models, err)
	}
}

// The anonymous entries are the registry's anonymous profiles but OpenCode
// Zen, which Zen serves.
func TestOpenAICompatibleAnonymousEntriesMatchTheRegistry(t *testing.T) {
	t.Parallel()
	var profiles []string
	for _, profile := range DefaultRegistry().AnonymousProfiles() {
		if profile.RegistryID != "opencode_zen" {
			profiles = append(profiles, profile.RegistryID)
			if profile.RuntimeType != "openai_compatible" || !openAIAnonymousEntry(profile.RegistryID) {
				t.Fatalf("profile = %+v", profile)
			}
		}
	}
	if !slices.Equal(profiles, []string{"kilo_code", "llm7", "ovh_ai_endpoints", "pollinations"}) || openAIAnonymousEntry("opencode_zen") {
		t.Fatalf("profiles = %v", profiles)
	}
}
