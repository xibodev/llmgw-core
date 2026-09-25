package anonymous_test

import (
	"context"
	"errors"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/anonymous"
	"github.com/xibodev/llmgw-core/providers"
)

// The port of TestAnonymousProviderAutomationRunsOncePerDay: the claim
// keeps a second run from repeating the check, and the check is recorded
// under its generation.
func TestRunChecksEachProviderOncePerInterval(t *testing.T) {
	t.Parallel()
	now := time.Unix(1_800_000_000, 0).UTC()
	hooks := &product{}
	u := &upstream{models: []core.ModelInfo{{ID: "codestral-latest"}}}
	o := orchestrator(t, hooks, u, anonymous.Options{
		Profiles: []providers.AnonymousProviderProfile{fixture("daily-fixture")}, Now: func() time.Time { return now },
	})
	results := o.RunOnce(context.Background())
	if discovers, probes := u.calls(); len(results) != 1 || !results[0].Success || discovers != 1 || probes != 1 {
		t.Fatalf("results=%+v discovers=%d probes=%d", results, discovers, probes)
	}
	body := u.probes[0].body
	if body["max_tokens"] != float64(16) || body["model"] != "codestral-latest" || body["stream"] != false ||
		u.probes[0].instance != "daily-fixture" || u.probes[0].caller.Kind != core.CallerAnonymous {
		t.Fatalf("probe=%+v", u.probes[0])
	}
	records := hooks.recordedResults()
	if len(records) != 1 || records[0].providerID != "daily-fixture" || records[0].generation != 7 ||
		len(records[0].result.Connect.Probes) != 1 || results[0].RecordErr != nil {
		t.Fatalf("records=%+v", records)
	}
	now = now.Add(23 * time.Hour)
	if results := o.RunOnce(context.Background()); len(results) != 0 {
		t.Fatalf("the claim repeated the check: %+v", results)
	}
	now = now.Add(time.Hour)
	if results := o.RunOnce(context.Background()); len(results) != 1 || len(hooks.recordedResults()) != 2 {
		t.Fatalf("the next day's check: %+v", results)
	}
}

// The ports of TestDisabledAutomationDoesNotChangeProviders and the gate
// the gateway consults between providers.
func TestRunStopsWhileTheAutomationIsOff(t *testing.T) {
	t.Parallel()
	off := &product{disabled: true}
	if results := orchestrator(t, off, &upstream{}, anonymous.Options{}).RunOnce(context.Background()); results != nil || len(off.enrolled) != 0 {
		t.Fatalf("results=%+v enrolled=%v", results, off.enrolled)
	}
	profiles := []providers.AnonymousProviderProfile{fixture("first"), fixture("second")}
	// On for the run, the first provider and its claim; off for the second.
	turning := &product{enabledFor: 3}
	u := &upstream{models: []core.ModelInfo{{ID: "codestral-latest"}}}
	results := orchestrator(t, turning, u, anonymous.Options{Profiles: profiles}).RunOnce(context.Background())
	if len(results) != 1 || results[0].ProviderID != "first" || len(turning.enrolled) != 1 {
		t.Fatalf("results=%+v enrolled=%v", results, turning.enrolled)
	}
}

// The ports of the gateway's enrollment tests: a provider the automation
// does not manage is reported with its status and never checked.
func TestRunLeavesUnmanagedProvidersAlone(t *testing.T) {
	t.Parallel()
	hooks := &product{statuses: map[string]string{"collision": "collision", "disabled": "disabled"}}
	u := &upstream{models: []core.ModelInfo{{ID: "codestral-latest"}}}
	profiles := []providers.AnonymousProviderProfile{fixture("collision"), fixture("disabled"), fixture("managed")}
	results := orchestrator(t, hooks, u, anonymous.Options{Profiles: profiles}).RunOnce(context.Background())
	if len(results) != 3 || results[0].Status != "collision" || results[1].Status != "disabled" ||
		results[0].Operation != "" || results[2].Status != anonymous.StatusPassed || len(hooks.claims) != 1 || len(u.probes) != 1 {
		t.Fatalf("results=%+v claims=%v", results, hooks.claims)
	}
}

func TestCheckWithoutEvidenceGenerationIsNotRecorded(t *testing.T) {
	t.Parallel()
	hooks := &product{generationErr: errors.New("evidence store unavailable")}
	u := &upstream{models: []core.ModelInfo{{ID: "codestral-latest"}}}
	results := orchestrator(t, hooks, u, anonymous.Options{}).RunOnce(context.Background())
	want := anonymous.Result{
		ProviderID: "fixture", RegistryID: "llm7", Operation: anonymous.OperationVerify, Status: anonymous.StatusFailed,
		FailureCode: anonymous.FailureEvidenceUnavailable, CatalogEvidence: core.CatalogNotProbed, CompletionEvidence: core.CompletionNotProbed,
	}
	if len(results) != 1 || results[0].FailureCode != want.FailureCode || results[0].Status != want.Status ||
		results[0].CatalogEvidence != want.CatalogEvidence || len(hooks.recordedResults()) != 0 || u.discovers != 0 {
		t.Fatalf("results=%+v", results)
	}
	failing := &product{recordErr: errors.New("write refused")}
	results = orchestrator(t, failing, u, anonymous.Options{}).RunOnce(context.Background())
	if len(results) != 1 || !results[0].Success || results[0].RecordErr == nil {
		t.Fatalf("a Record error was lost: %+v", results)
	}
}
