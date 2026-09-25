package anonymous_test

import (
	"context"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/anonymous"
	"github.com/xibodev/llmgw-core/providers"
)

// fixture is a reviewed profile whose provider id is the test's own.
func fixture(providerID string) providers.AnonymousProviderProfile {
	return providers.AnonymousProviderProfile{
		RegistryID: "llm7", ProviderID: providerID, RuntimeType: "openai_compatible", BaseURL: "https://fixture.invalid/v1",
	}
}

func orchestrator(t *testing.T, hooks anonymous.Hooks, u *upstream, options anonymous.Options) *anonymous.Orchestrator {
	t.Helper()
	options.Hooks, options.Catalog, options.Invoker = hooks, u, u
	if options.Profiles == nil {
		options.Profiles = []providers.AnonymousProviderProfile{fixture("fixture")}
	}
	o, err := anonymous.New(options)
	if err != nil {
		t.Fatal(err)
	}
	return o
}

// The port of TestGatewayProviderOrchestratorRegistersOnlyReviewedAnonymousProfiles.
func TestNewAdmitsOnlyReviewedAnonymousProfiles(t *testing.T) {
	t.Parallel()
	u := &upstream{}
	if _, err := anonymous.New(anonymous.Options{Catalog: u, Invoker: u, Hooks: &product{}}); err != nil {
		t.Fatalf("reviewed profiles: %v", err)
	}
	for _, profile := range []providers.AnonymousProviderProfile{
		{RegistryID: "openai_codex", ProviderID: "codex", RuntimeType: "openai_compatible"},
		{RegistryID: "custom_openai", ProviderID: "custom", RuntimeType: "openai_compatible"},
		{RegistryID: "llm7", ProviderID: "llm7", RuntimeType: "anthropic"},
		{RegistryID: "llm7", ProviderID: " ", RuntimeType: "openai_compatible"},
	} {
		options := anonymous.Options{Catalog: u, Invoker: u, Hooks: &product{}, Profiles: []providers.AnonymousProviderProfile{profile}}
		if _, err := anonymous.New(options); err == nil {
			t.Fatalf("profile %+v entered anonymous automation", profile)
		}
	}
	for _, options := range []anonymous.Options{
		{Invoker: u, Hooks: &product{}}, {Catalog: u, Hooks: &product{}}, {Catalog: u, Invoker: u},
		{Catalog: u, Invoker: u, Hooks: &product{}, Publish: "everything"},
	} {
		if _, err := anonymous.New(options); err == nil {
			t.Fatalf("options %+v were accepted", options)
		}
	}
}

// The ports of TestAnonymousProbeSelectorUsesEveryDiscoveredModel and
// TestAnonymousProbeSelectorDoesNotVerifySiblingsFromOneTarget: every
// model is probed once, and publishes on its own evidence.
func TestRunProbesEveryDiscoveredModelOnce(t *testing.T) {
	t.Parallel()
	u := &upstream{
		models:   []core.ModelInfo{{ID: "working"}, {ID: " broken "}, {ID: "working"}, {ID: ""}},
		failures: map[string]error{"broken": rejected(400)},
	}
	results := orchestrator(t, &product{}, u, anonymous.Options{}).RunOnce(context.Background())
	if len(results) != 1 || len(u.probes) != 2 || u.probes[0].model != "working" || u.probes[1].model != "broken" {
		t.Fatalf("results=%+v probes=%+v", results, u.probes)
	}
	result := results[0]
	if !result.Success || result.Status != anonymous.StatusPassed || result.Probed != 2 || result.Verified != 1 ||
		result.Failed != 1 || result.CompletionEvidence != core.CompletionFailed || result.Published != 1 ||
		result.Targets[0] != (core.Target{Provider: "fixture", Model: "working"}) || result.FailureCode != "" {
		t.Fatalf("result=%+v", result)
	}
}

func TestProbeVerificationModelProbesTheReviewedDefault(t *testing.T) {
	t.Parallel()
	u := &upstream{models: []core.ModelInfo{{ID: "paid"}, {ID: "codestral-latest"}}}
	o := orchestrator(t, &product{}, u, anonymous.Options{Probe: anonymous.ProbeVerificationModel, Publish: core.PublishDiscoveredTargets})
	results := o.RunOnce(context.Background())
	if len(u.probes) != 1 || u.probes[0].model != "codestral-latest" || len(results) != 1 || results[0].Published != 2 {
		t.Fatalf("results=%+v probes=%+v", results, u.probes)
	}
	empty := &upstream{}
	results = orchestrator(t, &product{}, empty, anonymous.Options{Probe: anonymous.ProbeVerificationModel}).RunOnce(context.Background())
	if len(results) != 1 || results[0].FailureCode != anonymous.FailureModelUnavailable || results[0].CatalogEvidence != core.CatalogEmpty {
		t.Fatalf("an empty catalog: %+v", results)
	}
}

func rejected(status int) error {
	return &core.ProviderError{Message: "upstream refused", Classification: core.ProviderErrorClassification{StatusCode: status}}
}
