package providers_test

import (
	"encoding/json"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/xibodev/llmgw-core/providers"
)

// gatewayRuntimeTypes are the runtime types llm-gateway executes; the default
// manifest must be fully runnable by it.
var gatewayRuntimeTypes = []string{
	"openai_compatible", "anthropic", "bedrock", "github_copilot", "ollama", "litellm", "edge_tts",
	"ai_studio", "vertex_ai", "azure_openai", "google_antigravity",
}

func validEntry(id string) providers.RegistryEntry {
	return providers.RegistryEntry{
		ID: id, Label: id, Description: "test provider", RuntimeType: "openai_compatible",
		Protocol: "openai", Availability: providers.ProviderAvailable, AuthMethods: []string{"api_key"},
		ConnectionScope: providers.ConnectionScopeSystemOrPersonal, DefaultProviderID: id,
		RiskLevel: providers.RiskGreen,
	}
}

func mustRegistry(t *testing.T, entries ...providers.RegistryEntry) *providers.Registry {
	t.Helper()
	registry, err := providers.NewRegistry(entries, providers.ValidationOptions{})
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	return registry
}

func ptr[T any](value T) *T { return &value }

func TestDefaultRegistryIsValidAndRunnable(t *testing.T) {
	t.Parallel()
	registry := providers.DefaultRegistry()
	if registry.Len() != 25 {
		t.Fatalf("default registry has %d entries, want 25", registry.Len())
	}
	if providers.DefaultRegistry() != registry {
		t.Fatal("DefaultRegistry must return one shared immutable registry")
	}
	if _, err := providers.NewRegistry(registry.Entries(), providers.ValidationOptions{RuntimeTypes: gatewayRuntimeTypes}); err != nil {
		t.Fatalf("default manifest is not runnable by the gateway runtime types: %v", err)
	}
	for _, entry := range registry.Entries() {
		if entry.Priority != 0 || entry.Icon != "" || entry.Domain != "" || len(entry.CommonModels) != 0 || len(entry.Extensions) != 0 {
			t.Fatalf("default manifest sets overlay-only fields on %q", entry.ID)
		}
	}
}

func TestDefaultManifestContracts(t *testing.T) {
	t.Parallel()
	anthropic, ok := providers.RegistryProviderByID("anthropic")
	if !ok || !slices.Contains(anthropic.AuthMethods, providers.AuthAPIKey) || !slices.Contains(anthropic.AuthMethods, providers.AuthSetupToken) {
		t.Fatalf("anthropic=%+v", anthropic)
	}
	antigravity, ok := providers.RegistryProvider("google-antigravity")
	if !ok || antigravity.ID != "google_antigravity" || !slices.Equal(antigravity.AuthMethods, []string{providers.AuthOAuthBrowser}) ||
		antigravity.ConnectionScope != providers.ConnectionScopePersonal || antigravity.RiskLevel != providers.RiskRed || antigravity.RiskNotice == "" {
		t.Fatalf("antigravity=%+v", antigravity)
	}
	codex, ok := providers.RegistryProvider("openai_codex")
	if !ok || !slices.Equal(codex.AuthMethods, []string{providers.AuthOAuthBrowser, providers.AuthOAuthDevice}) {
		t.Fatalf("codex=%+v", codex)
	}
	anonymous := map[string]bool{"opencode_zen": true, "kilo_code": true, "llm7": true, "ovh_ai_endpoints": true, "pollinations": true}
	for _, entry := range providers.ProviderRegistry() {
		if entry.AnonymousAutomation != anonymous[entry.ID] {
			t.Fatalf("anonymous automation for %q=%v", entry.ID, entry.AnonymousAutomation)
		}
	}
	zen, ok := providers.RegistryProvider("opencode_zen")
	if !ok || zen.RequiresAPIKey || !slices.Contains(zen.AuthMethods, providers.AuthNone) || !slices.Contains(zen.OnboardingFields, "api_key") {
		t.Fatalf("opencode_zen=%+v", zen)
	}
	gemini, ok := providers.RegistryProvider(" GEMINI ")
	if !ok || gemini.RuntimeType != "openai_compatible" || gemini.DefaultBaseURL != "https://generativelanguage.googleapis.com/v1beta/openai" || !gemini.RequiresAPIKey {
		t.Fatalf("gemini=%+v", gemini)
	}
	if localai, ok := providers.RegistryProvider("localai"); !ok || !localai.InferAudioCapabilities {
		t.Fatalf("localai=%+v", localai)
	}
	vertex, ok := providers.RegistryProvider("vertex_ai")
	if !ok || vertex.DefaultLocation != "global" || !slices.Contains(vertex.OnboardingFields, "vertex_request_type") {
		t.Fatalf("vertex_ai=%+v", vertex)
	}
	for _, id := range []string{"github_copilot", "openai_codex"} {
		if entry, ok := providers.RegistryProvider(id); !ok || entry.ConnectionScope != providers.ConnectionScopePersonal {
			t.Fatalf("%s=%+v, want personal scope", id, entry)
		}
	}
	claude, ok := providers.RegistryProvider("claude_code")
	if !ok || !claude.ClientOnly || claude.Availability != providers.ProviderClientOnly || claude.ConnectionScope != providers.ConnectionScopeGatewayClient {
		t.Fatalf("claude_code=%+v", claude)
	}
}

func TestLookupResolvesAliasesAndReturnsCopies(t *testing.T) {
	t.Parallel()
	entry, ok := providers.RegistryProvider(" COPILOT ")
	if !ok || entry.ID != "github_copilot" || entry.RiskLevel != providers.RiskYellow || entry.QuotaAdapter != "github_copilot" {
		t.Fatalf("alias lookup=%+v ok=%v", entry, ok)
	}
	entry.AuthMethods[0] = "mutated"
	entry.Aliases[0] = "mutated"
	fresh, _ := providers.RegistryProvider("github_copilot")
	if fresh.AuthMethods[0] != providers.AuthOAuthDevice || fresh.Aliases[0] != "copilot" {
		t.Fatalf("registry mutated through a caller copy: %+v", fresh)
	}
	if got := providers.CanonicalRegistryID(" CoDeX "); got != "openai_codex" {
		t.Fatalf("canonical codex id=%q", got)
	}
	if got := providers.CanonicalRegistryID(" custom-provider "); got != "custom-provider" {
		t.Fatalf("unknown id=%q", got)
	}
	if _, ok := providers.RegistryProviderByID("copilot"); ok {
		t.Fatal("ByID must not resolve aliases")
	}
}

func TestValidationRejectsUnsafeEntries(t *testing.T) {
	t.Parallel()
	cases := map[string]func(*providers.RegistryEntry){
		"requires api key without api_key auth": func(e *providers.RegistryEntry) { e.RequiresAPIKey, e.AuthMethods = true, []string{providers.AuthNone} },
		"non-green without notice":              func(e *providers.RegistryEntry) { e.RiskLevel = providers.RiskYellow },
		"missing risk":                          func(e *providers.RegistryEntry) { e.RiskLevel = "" },
		"oauth without adapter": func(e *providers.RegistryEntry) {
			e.AuthMethods, e.ConnectionScope = []string{providers.AuthOAuthDevice}, providers.ConnectionScopePersonal
		},
		"personal no-auth": func(e *providers.RegistryEntry) {
			e.AuthMethods, e.ConnectionScope = []string{providers.AuthNone}, providers.ConnectionScopePersonal
		},
		"unknown auth":        func(e *providers.RegistryEntry) { e.AuthMethods = []string{"password"} },
		"invalid id":          func(e *providers.RegistryEntry) { e.ID = "Not Valid" },
		"missing label":       func(e *providers.RegistryEntry) { e.Label = "" },
		"invalid docs url":    func(e *providers.RegistryEntry) { e.DocsURL = "ftp://example.test" },
		"unsafe anonymous":    func(e *providers.RegistryEntry) { e.AnonymousAutomation = true },
		"non-finite priority": func(e *providers.RegistryEntry) { e.Priority = math.NaN() },
		"invalid domain":      func(e *providers.RegistryEntry) { e.Domain = "https://example.test/path" },
		"invalid icon":        func(e *providers.RegistryEntry) { e.Icon = "not an icon" },
		"empty common model":  func(e *providers.RegistryEntry) { e.CommonModels = []string{" "} },
		"invalid extension key": func(e *providers.RegistryEntry) {
			e.Extensions = map[string]json.RawMessage{"Bad Key": json.RawMessage(`true`)}
		},
		"invalid extension": func(e *providers.RegistryEntry) {
			e.Extensions = map[string]json.RawMessage{"facet": json.RawMessage(`{`)}
		},
		"inconsistent client-only": func(e *providers.RegistryEntry) {
			e.ClientOnly = true
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			entry := validEntry("candidate")
			mutate(&entry)
			if _, err := providers.NewRegistry([]providers.RegistryEntry{entry}, providers.ValidationOptions{}); err == nil {
				t.Fatalf("%s was accepted", name)
			}
		})
	}
}

func TestValidationRejectsAliasCollisionsAndUnrunnableEntries(t *testing.T) {
	t.Parallel()
	first, second := validEntry("first"), validEntry("second")
	first.Aliases, second.Aliases = []string{"shared"}, []string{"SHARED"}
	if _, err := providers.NewRegistry([]providers.RegistryEntry{first, second}, providers.ValidationOptions{}); err == nil {
		t.Fatal("case-insensitive alias collision was accepted")
	}
	custom := validEntry("custom")
	custom.RuntimeType = "exotic"
	if _, err := providers.NewRegistry([]providers.RegistryEntry{custom}, providers.ValidationOptions{RuntimeTypes: gatewayRuntimeTypes}); err == nil {
		t.Fatal("an available entry with an unsupported runtime was accepted")
	}
	if _, err := providers.NewRegistry([]providers.RegistryEntry{custom}, providers.ValidationOptions{}); err != nil {
		t.Fatalf("nil runtime types must skip the runtime check: %v", err)
	}
	if _, err := providers.NewRegistry(nil, providers.ValidationOptions{}); err == nil {
		t.Fatal("an empty manifest was accepted")
	}
}

func TestNewRegistryNormalizesWithoutMutatingInput(t *testing.T) {
	t.Parallel()
	entry := validEntry("normalize")
	entry.ID, entry.Aliases, entry.AuthMethods = " Normalize ", []string{" ALIAS "}, []string{" API_KEY "}
	registry := mustRegistry(t, entry)
	got, ok := registry.Lookup("alias")
	if !ok || got.ID != "normalize" || got.AuthMethods[0] != providers.AuthAPIKey {
		t.Fatalf("normalized=%+v ok=%v", got, ok)
	}
	if entry.ID != " Normalize " || entry.Aliases[0] != " ALIAS " || entry.AuthMethods[0] != " API_KEY " {
		t.Fatalf("input mutated: %+v", entry)
	}
}

func TestDecodeRegistryEntriesIsStrict(t *testing.T) {
	t.Parallel()
	raw, _ := json.Marshal(validEntry("strict"))
	var object map[string]any
	_ = json.Unmarshal(raw, &object)
	object["risk_levle"] = "yellow"
	payload, _ := json.Marshal([]map[string]any{object})
	if _, err := providers.DecodeRegistryEntries(payload); err == nil {
		t.Fatal("an unknown manifest field was accepted")
	}
	if _, err := providers.DecodeRegistryEntries([]byte(`[] []`)); err == nil {
		t.Fatal("trailing content was accepted")
	}
}

func TestEmptyOverlayReproducesBaseExactly(t *testing.T) {
	t.Parallel()
	base := providers.DefaultRegistry()
	overlaid, err := base.WithOverlay(providers.Overlay{}, providers.ValidationOptions{RuntimeTypes: gatewayRuntimeTypes})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(overlaid.Entries(), base.Entries()) {
		t.Fatal("an empty overlay changed the registry")
	}
	encodedBase, _ := json.Marshal(base.Entries())
	encodedOverlay, _ := json.Marshal(overlaid.Entries())
	if string(encodedBase) != string(encodedOverlay) {
		t.Fatal("an empty overlay changed the registry's JSON")
	}
}

func TestOverlayChangesOnlyCurationAndKeepsBaseIntact(t *testing.T) {
	t.Parallel()
	base := mustRegistry(t, validEntry("alpha"), validEntry("beta"), validEntry("gamma"))
	withCategories := validEntry("delta")
	withCategories.Categories = []string{"chat"}
	base, err := base.WithOverlay(providers.Overlay{Add: []providers.RegistryEntry{withCategories}}, providers.ValidationOptions{})
	if err != nil {
		t.Fatal(err)
	}

	overlaid, err := base.WithOverlay(providers.Overlay{
		Override: []providers.EntryOverride{
			{
				ID: "GAMMA", Label: ptr("Gamma Product"), Priority: ptr(10.0), Icon: ptr("gamma"),
				Domain: ptr("gamma.example.test"), Aliases: []string{"g"}, CommonModels: []string{"gamma-large"},
				Extensions: map[string]json.RawMessage{"product": json.RawMessage(`{"create_allowed":true}`)},
			},
			{ID: "delta", Categories: []string{}},
		},
		Remove: []string{"beta"},
	}, providers.ValidationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ids := []string{}
	for _, entry := range overlaid.Entries() {
		ids = append(ids, entry.ID)
	}
	if !slices.Equal(ids, []string{"gamma", "alpha", "delta"}) {
		t.Fatalf("order=%v, want priority first then manifest order", ids)
	}
	gamma, ok := overlaid.Lookup("g")
	if !ok || gamma.Label != "Gamma Product" || gamma.Icon != "gamma" || gamma.Domain != "gamma.example.test" ||
		!slices.Equal(gamma.CommonModels, []string{"gamma-large"}) || string(gamma.Extensions["product"]) != `{"create_allowed":true}` {
		t.Fatalf("gamma=%+v ok=%v", gamma, ok)
	}
	if gamma.RuntimeType != "openai_compatible" || gamma.RiskLevel != providers.RiskGreen {
		t.Fatalf("an override changed wire or safety facts: %+v", gamma)
	}
	if delta, _ := overlaid.ByID("delta"); len(delta.Categories) != 0 {
		t.Fatalf("an empty slice must clear categories: %+v", delta.Categories)
	}
	if _, ok := overlaid.ByID("beta"); ok {
		t.Fatal("removed entry still resolves")
	}
	if original, ok := base.ByID("gamma"); !ok || original.Label != "gamma" || original.Priority != 0 {
		t.Fatalf("WithOverlay changed its receiver: %+v", original)
	}
	if _, ok := base.ByID("beta"); !ok {
		t.Fatal("WithOverlay removed an entry from its receiver")
	}
}

func TestOverlayKeepsCoreAliases(t *testing.T) {
	t.Parallel()
	overlaid, err := providers.DefaultRegistry().WithOverlay(providers.Overlay{
		Override: []providers.EntryOverride{{ID: "github_copilot", Aliases: []string{"github-copilot", "COPILOT", "github_copilot"}}},
	}, providers.ValidationOptions{})
	if err != nil {
		t.Fatalf("re-declaring an entry's own alias or id must be accepted: %v", err)
	}
	for _, name := range []string{"copilot", "github-copilot"} {
		if entry, ok := overlaid.Lookup(name); !ok || entry.ID != "github_copilot" {
			t.Fatalf("%q resolved to %+v ok=%v", name, entry, ok)
		}
	}
}

func TestOverlayRejectsAmbiguousOrUnsafeChanges(t *testing.T) {
	t.Parallel()
	base := mustRegistry(t, validEntry("alpha"), validEntry("beta"))
	invalid := validEntry("gamma")
	invalid.AuthMethods = nil
	collision := validEntry("delta")
	collision.Aliases = []string{"alpha"}
	cases := map[string]providers.Overlay{
		"unknown removal":       {Remove: []string{"missing"}},
		"duplicate removal":     {Remove: []string{"alpha", "ALPHA"}},
		"unknown override":      {Override: []providers.EntryOverride{{ID: "missing"}}},
		"duplicate override":    {Override: []providers.EntryOverride{{ID: "alpha"}, {ID: "alpha"}}},
		"override removed":      {Remove: []string{"alpha"}, Override: []providers.EntryOverride{{ID: "alpha"}}},
		"re-add base id":        {Remove: []string{"alpha"}, Add: []providers.RegistryEntry{validEntry("alpha")}},
		"invalid added entry":   {Add: []providers.RegistryEntry{invalid}},
		"alias collision":       {Add: []providers.RegistryEntry{collision}},
		"invalid availability":  {Override: []providers.EntryOverride{{ID: "alpha", Availability: ptr("everywhere")}}},
		"invalid overlay field": {Override: []providers.EntryOverride{{ID: "alpha", Domain: ptr("not a domain")}}},
		"remove everything":     {Remove: []string{"alpha", "beta"}},
	}
	for name, overlay := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if _, err := base.WithOverlay(overlay, providers.ValidationOptions{}); err == nil {
				t.Fatalf("%s was accepted", name)
			}
		})
	}
	planned := validEntry("planned")
	planned.Availability, planned.RuntimeType = providers.ProviderPlanned, "exotic"
	runtimeBase := mustRegistry(t, planned)
	if _, err := runtimeBase.WithOverlay(providers.Overlay{
		Override: []providers.EntryOverride{{ID: "planned", Availability: ptr(providers.ProviderAvailable)}},
	}, providers.ValidationOptions{RuntimeTypes: gatewayRuntimeTypes}); err == nil {
		t.Fatal("an overlay made an unrunnable entry available")
	}
}

func TestDecodeOverlayRejectsWireFactOverrides(t *testing.T) {
	t.Parallel()
	for _, field := range []string{"runtime_type", "auth_methods", "risk_level", "default_base_url", "anonymous_automation"} {
		payload := `{"override":[{"id":"openai","` + field + `":"x"}]}`
		if _, err := providers.DecodeOverlay([]byte(payload)); err == nil || !strings.Contains(err.Error(), "unknown field") {
			t.Fatalf("override of %s: err=%v, want an unknown-field error", field, err)
		}
	}
}

// TestProductMetadataIsExpressibleAsOverlay shows the shape of a product such
// as Facet Studio: renamed ids as aliases, presentation, priority, curated
// models, product-private flags, product-only entries, and removals.
func TestProductMetadataIsExpressibleAsOverlay(t *testing.T) {
	t.Parallel()
	overlay, err := providers.DecodeOverlay([]byte(`{
		"override": [
			{"id": "openai", "label": "OpenAI", "icon": "openai", "domain": "openai.com", "priority": 100,
			 "aliases": ["gpt"], "common_models": ["gpt-5.4", "gpt-5.4-mini"],
			 "extensions": {"facet": {"create_allowed": true, "default_model_allowed": true}}},
			{"id": "github_copilot", "aliases": ["github-copilot"], "priority": 55, "categories": ["local"]}
		],
		"add": [
			{"id": "deepseek", "label": "DeepSeek", "description": "DeepSeek chat models.",
			 "runtime_type": "openai_compatible", "protocol": "openai", "availability": "available",
			 "auth_methods": ["api_key"], "connection_scope": "system_or_personal",
			 "default_provider_id": "deepseek", "default_base_url": "https://api.deepseek.com/v1",
			 "requires_api_key": true, "supports_model_discovery": true, "risk_level": "green",
			 "priority": 85, "icon": "deepseek", "domain": "deepseek.com"}
		],
		"remove": ["edge_tts"]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	product, err := providers.DefaultRegistry().WithOverlay(overlay, providers.ValidationOptions{RuntimeTypes: gatewayRuntimeTypes})
	if err != nil {
		t.Fatal(err)
	}
	entries := product.Entries()
	if entries[0].ID != "openai" || entries[1].ID != "deepseek" || entries[2].ID != "github_copilot" {
		t.Fatalf("priority order starts %s, %s, %s", entries[0].ID, entries[1].ID, entries[2].ID)
	}
	if entry, ok := product.Lookup("gpt"); !ok || entry.ID != "openai" || entry.Domain != "openai.com" {
		t.Fatalf("gpt=%+v ok=%v", entry, ok)
	}
	var flags struct {
		CreateAllowed bool `json:"create_allowed"`
	}
	openai, _ := product.ByID("openai")
	if err := json.Unmarshal(openai.Extensions["facet"], &flags); err != nil || !flags.CreateAllowed {
		t.Fatalf("product extension=%s err=%v", openai.Extensions["facet"], err)
	}
	if _, ok := product.ByID("edge_tts"); ok || product.Len() != 25 {
		t.Fatalf("len=%d: want 25 entries after one removal and one addition", product.Len())
	}
	if providers.DefaultRegistry().Len() != 25 {
		t.Fatal("an overlay changed the default registry")
	}
}

func TestAnonymousProfilesFollowTheRegistry(t *testing.T) {
	t.Parallel()
	overlaid, err := providers.DefaultRegistry().WithOverlay(providers.Overlay{Remove: []string{"llm7"}}, providers.ValidationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ids := []string{}
	for _, profile := range overlaid.AnonymousProfiles() {
		ids = append(ids, profile.RegistryID)
	}
	if !slices.Equal(ids, []string{"kilo_code", "opencode_zen", "ovh_ai_endpoints", "pollinations"}) {
		t.Fatalf("profiles=%v", ids)
	}
	if len(providers.AnonymousProviderProfiles()) != 5 {
		t.Fatal("removing an entry from an overlay changed the default profiles")
	}
}
