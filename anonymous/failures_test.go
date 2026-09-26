package anonymous_test

import (
	"context"
	"errors"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/anonymous"
)

// rateLimited is a 429 as the wire kit reports it, with Retry-After.
func rateLimited() error {
	return &core.ProviderError{Message: "upstream returned HTTP 429", Class: core.ProviderErrorRateLimited,
		Classification: core.ProviderErrorClassification{StatusCode: 429, Retryable: true, FailoverEligible: true,
			CircuitFailure: true, RetryAfter: 17 * time.Second}}
}

// The ports of TestAnonymousProviderAutomationPreservesSharedFailureEvidence
// and TestAnonymousVerificationPreservesRetryAfterClassification.
func TestFailedProbeKeepsItsStatusAndRetryAfter(t *testing.T) {
	t.Parallel()
	hooks := &product{}
	u := &upstream{models: []core.ModelInfo{{ID: "codestral-latest"}}, failures: map[string]error{"codestral-latest": rateLimited()}}
	results := orchestrator(t, hooks, u, anonymous.Options{}).RunOnce(context.Background())
	if len(results) != 1 {
		t.Fatalf("results=%+v", results)
	}
	result := results[0]
	if result.Status != anonymous.StatusFailed || result.Success || result.AuthenticationState != anonymous.AuthenticationAccepted ||
		result.CatalogEvidence != core.CatalogDiscovered || result.CompletionEvidence != core.CompletionFailed ||
		result.FailureCode != anonymous.FailureVerificationFailed || !result.Retryable || result.RetryAfter != 17*time.Second ||
		result.VerificationError == "" || result.Published != 0 {
		t.Fatalf("result=%+v", result)
	}
	records := hooks.recordedResults()
	if len(records) != 1 || anonymous.ProbeFailureCode(records[0].result.Connect.Probes[0]) != string(core.ProviderErrorRateLimited) {
		t.Fatalf("records=%+v", records)
	}
}

// The port of TestAnonymousVerificationClassifiesPermanentAuthFailureWithoutRetry,
// as the core orchestrator reports it.
func TestRejectedProbeIsAPermanentAuthenticationFailure(t *testing.T) {
	t.Parallel()
	u := &upstream{models: []core.ModelInfo{{ID: "codestral-latest"}}, failures: map[string]error{"codestral-latest": rejected(401)}}
	results := orchestrator(t, &product{}, u, anonymous.Options{}).RunOnce(context.Background())
	if len(results) != 1 || results[0].FailureCode != anonymous.FailureAuthenticationRejected ||
		results[0].AuthenticationState != anonymous.AuthenticationRejected || results[0].CompletionEvidence != core.CompletionFailed ||
		results[0].Retryable || !results[0].Connect.Health.Permanent {
		t.Fatalf("results=%+v", results)
	}
}

// A catalog that fails ends the check before any probe, classified by its
// status, and the details never quote the upstream.
func TestFailedCatalogEndsTheCheck(t *testing.T) {
	t.Parallel()
	for status, code := range map[int]string{
		401: anonymous.FailureAuthenticationRejected, 403: anonymous.FailureAuthenticationRejected,
		500: anonymous.FailureCatalogFailed, 0: anonymous.FailureCatalogFailed,
	} {
		failure := rejected(status)
		if status == 0 {
			failure = errors.New("dial fixture.invalid: secret-bearing detail")
		}
		u := &upstream{catalogErr: failure}
		results := orchestrator(t, &product{}, u, anonymous.Options{}).RunOnce(context.Background())
		if len(results) != 1 || results[0].FailureCode != code || results[0].CatalogEvidence != core.CatalogFailed ||
			results[0].Details != "provider catalog failed" || len(u.probes) != 0 {
			t.Fatalf("status %d: results=%+v", status, results)
		}
	}
}

func TestUnreadableAnswerFailsTheProbe(t *testing.T) {
	t.Parallel()
	u := &upstream{models: []core.ModelInfo{{ID: "codestral-latest"}}}
	invoker := anonymous.InvokerFunc(func(context.Context, core.Caller, string, core.Request) (core.Response, error) {
		return core.Response{Body: []byte("not json"), ContentType: "text/plain"}, nil
	})
	o, err := anonymous.New(anonymous.Options{Catalog: u, Invoker: invoker, Hooks: &product{}})
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range o.RunOnce(context.Background()) {
		if result.Success || result.Verified != 0 {
			t.Fatalf("an unreadable answer verified %s: %+v", result.ProviderID, result)
		}
	}
}

func TestProbeFailureCode(t *testing.T) {
	t.Parallel()
	for probe, want := range map[core.CompletionProbeEvidence]string{
		{Status: core.CompletionVerified, FailureCode: "rate_limited"}: "",
		{Status: core.CompletionFailed, FailureCode: " rate_limited "}: "rate_limited",
		{Status: core.CompletionFailed, FailureCode: "none"}:           anonymous.FailureVerificationFailed,
		{Status: core.CompletionFailed}:                                anonymous.FailureVerificationFailed,
	} {
		if got := anonymous.ProbeFailureCode(probe); got != want {
			t.Errorf("ProbeFailureCode(%+v)=%q want %q", probe, got, want)
		}
	}
}
