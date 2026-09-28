package providers_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

// ovhCatalogFixture is OVHcloud AI Endpoints' catalog as GET /v1/models
// answered it without a key on 2026-09-28, byte for byte.
const ovhCatalogFixture = "testdata/ovh_ai_endpoints/models-2026-09-28.json"

// ovhCatalogRows sorts the fixture's rows: the chat models anonymous access
// admits, in catalog order, and every other row by kind.
func ovhCatalogRows() (chat []string, rejected map[string][]string) {
	return []string{
			"Qwen3.6-27B", "Qwen3.5-9B", "Qwen3.8-27B", "Qwen3-Coder-30B-A3B-Instruct", "Meta-Llama-3_3-70B-Instruct",
			"Mistral-Small-3.2-24B-Instruct-2506", "Qwen3.5-397B-A17B", "gpt-oss-120b", "Mistral-7B-Instruct-v0.3",
			"Qwen2.5-VL-72B-Instruct", "Mistral-Nemo-Instruct-2407", "gpt-oss-20b",
		}, map[string][]string{
			"classifier":     {"Qwen3Guard-Gen-0.6B", "Qwen3Guard-Gen-8B"},
			"embedding":      {"Qwen3-Embedding-8B", "bge-m3", "bge-multilingual-gemma2"},
			"speech-to-text": {"whisper-large-v3-turbo", "whisper-large-v3"},
			"text-to-speech": {"nvr-tts-it-it", "nvr-tts-en-us", "nvr-tts-de-de", "nvr-tts-es-es"},
			"image":          {"stable-diffusion-xl-base-v10"},
		}
}

func ovhProfile(t *testing.T) providers.AnonymousProviderProfile {
	t.Helper()
	for _, profile := range providers.AnonymousProviderProfiles() {
		if profile.RegistryID == "ovh_ai_endpoints" {
			return profile
		}
	}
	t.Fatal("the default registry has no OVH AI Endpoints profile")
	return providers.AnonymousProviderProfile{}
}

// OVHcloud AI Endpoints serves its chat models without a key whatever the
// paid price its catalog lists. Against the live catalog, anonymous access
// admits every chat model and rejects every classifier, embedding,
// speech-to-text, text-to-speech and image row, on every path that admits:
// the rule, catalog discovery, and OpenAICompatible's keyless catalog. The
// reviewed Qwen3.8-27B is the verification model.
func TestOVHAnonymousAccessAdmitsTheLiveCatalogsChatModels(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(ovhCatalogFixture)
	if err != nil {
		t.Fatal(err)
	}
	var catalog struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(raw, &catalog); err != nil {
		t.Fatal(err)
	}
	chat, rejected := ovhCatalogRows()
	listed := make([]string, 0, len(catalog.Data))
	for _, row := range catalog.Data {
		id, _ := row["id"].(string)
		listed = append(listed, id)
	}
	sorted := slices.Clone(chat)
	for _, ids := range rejected {
		sorted = append(sorted, ids...)
	}
	slices.Sort(sorted)
	if all := slices.Sorted(slices.Values(listed)); !slices.Equal(all, sorted) {
		t.Fatalf("the fixture lists %v, the test sorts %v", all, sorted)
	}

	for _, row := range catalog.Data {
		id, _ := row["id"].(string)
		admission := providers.AdmitAnonymousModel("ovh_ai_endpoints", row)
		if admission.Free != slices.Contains(chat, id) || admission.ContextWindow != int(row["context_length"].(float64)) {
			t.Errorf("%s: admission = %+v", id, admission)
		}
		// None of the chat models is free by the paid tier's price, which
		// is why a price rule admitted only the classifiers.
		if pricing, _ := row["pricing"].(map[string]any); slices.Contains(chat, id) && pricing["prompt"] == "0" && pricing["completion"] == "0" {
			t.Errorf("%s: the fixture lists a zero price", id)
		}
	}
	for kind, ids := range rejected {
		for _, id := range ids {
			if slices.Contains(chat, id) {
				t.Errorf("the %s row %s is sorted as chat", kind, id)
			}
		}
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" || r.Header.Get("Authorization") != "" {
			t.Errorf("request %s authorization=%q", r.URL.Path, r.Header.Get("Authorization"))
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(raw)
	}))
	defer server.Close()
	profile := ovhProfile(t)
	if !slices.Equal(profile.VerificationModels, []string{"Qwen3.8-27B", "Mistral-Nemo-Instruct-2407", "gpt-oss-20b"}) {
		t.Fatalf("reviewed verification models = %v", profile.VerificationModels)
	}
	for _, model := range profile.VerificationModels {
		if !slices.Contains(chat, model) {
			t.Fatalf("the reviewed %s is not admitted", model)
		}
	}
	profile.BaseURL = server.URL + "/v1"
	discovered, err := providers.DiscoverAnonymousModels(context.Background(), profile, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{
		BaseURL: profile.BaseURL, RegistryID: profile.RegistryID, CatalogClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	keyless, err := provider.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	for path, models := range map[string][]core.ModelInfo{"discovery": discovered, "keyless catalog": keyless} {
		ids := make([]string, len(models))
		for index, model := range models {
			ids[index] = model.ID
		}
		if !slices.Equal(ids, chat) {
			t.Fatalf("%s admits %v, want %v", path, ids, chat)
		}
		if model := providers.AnonymousVerificationModel(profile.RegistryID, models); model != "Qwen3.8-27B" {
			t.Fatalf("%s verifies with %q", path, model)
		}
	}
	for _, model := range keyless {
		if !model.Free || !slices.Equal(model.SupportedAPIs, []string{"/chat/completions"}) || model.LegacyCapabilities["chat"] != true ||
			model.LegacyCapabilities["context_window"] == nil {
			t.Fatalf("admitted row = %+v", model)
		}
	}
}
