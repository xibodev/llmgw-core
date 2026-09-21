package core_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

func TestCodexRequiresPersonalSubscriptionDeviceOAuth(t *testing.T) {
	valid := core.ProviderConnection{
		ProviderID: "codex", Kind: core.ProviderConnectionPersonalSubscription,
		AuthKind: core.ProviderAuthOAuthDevice, OwnerID: "owner-1",
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid Codex connection rejected: %v", err)
	}
	for _, connection := range []core.ProviderConnection{
		{ProviderID: "codex", Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous},
		{ProviderID: "openai_codex", Kind: core.ProviderConnectionSystem, AuthKind: core.ProviderAuthAPIKey},
		{ProviderID: "chatgpt", Kind: core.ProviderConnectionPersonalSubscription, AuthKind: core.ProviderAuthOAuthDevice},
	} {
		if err := connection.Validate(); err == nil {
			t.Fatalf("invalid Codex connection accepted: %+v", connection)
		}
	}
}

func TestProviderConnectionDoesNotMarshalCredential(t *testing.T) {
	connection := core.ProviderConnection{
		ProviderID: "openai", Kind: core.ProviderConnectionSystem, AuthKind: core.ProviderAuthAPIKey,
		Credential: &core.Credential{APIKey: "secret-value"},
	}
	encoded, err := json.Marshal(connection)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) == "" || strings.Contains(string(encoded), "secret-value") || strings.Contains(string(encoded), "credential") {
		t.Fatalf("credential leaked in JSON: %s", encoded)
	}
}

func TestCatalogDiscoveryDoesNotVerifyInference(t *testing.T) {
	catalog := core.CatalogEvidence{
		Status: core.CatalogDiscovered,
		Models: []core.ModelInfo{{ID: "model-a"}},
	}
	probe := core.CompletionProbeEvidence{Target: core.Target{Provider: "provider-a", Model: "model-a"}, Status: core.CompletionNotProbed}
	if catalog.Status != core.CatalogDiscovered || probe.InferenceVerified() {
		t.Fatalf("catalog discovery incorrectly implied verified inference: catalog=%+v probe=%+v", catalog, probe)
	}
}

func TestClassifyProviderFailure(t *testing.T) {
	now := time.Date(2026, time.September, 20, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name      string
		failure   core.ProviderFailure
		class     core.ProviderErrorClass
		permanent bool
		retryable bool
		retry     time.Duration
	}{
		{name: "unauthorized", failure: core.ProviderFailure{StatusCode: 401}, class: core.ProviderErrorAuth, permanent: true},
		{name: "forbidden", failure: core.ProviderFailure{StatusCode: 403}, class: core.ProviderErrorForbidden, permanent: true},
		{name: "rate limited", failure: core.ProviderFailure{StatusCode: 429, RetryAfter: "17", ObservedAt: now}, class: core.ProviderErrorRateLimited, retryable: true, retry: 17 * time.Second},
		{name: "transport", failure: core.ProviderFailure{Err: errors.New("connection reset")}, class: core.ProviderErrorTransport, retryable: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := core.ClassifyProviderFailure(test.failure)
			if got.ErrorClass != test.class || got.Permanent != test.permanent || got.Retryable != test.retryable || got.RetryAfter != test.retry {
				t.Fatalf("classification mismatch: %+v", got)
			}
		})
	}
}

func TestPublishExactTargetsRequiresSuccessfulInferenceWhenConfigured(t *testing.T) {
	catalog := core.CatalogEvidence{
		Status: core.CatalogDiscovered,
		Models: []core.ModelInfo{{ID: "model-a"}, {ID: "model-b"}},
	}
	probes := []core.CompletionProbeEvidence{
		{Target: core.Target{Provider: "provider-a", Model: "model-a"}, Status: core.CompletionVerified},
		{Target: core.Target{Provider: "provider-a", Model: "model-b"}, Status: core.CompletionFailed},
	}
	targets, err := core.PublishExactTargets("provider-a", catalog, probes, core.PublishVerifiedTargets)
	if err != nil {
		t.Fatal(err)
	}
	if len(targets) != 1 || targets[0] != (core.Target{Provider: "provider-a", Model: "model-a"}) {
		t.Fatalf("published targets=%+v", targets)
	}

	targets, err = core.PublishExactTargets("provider-a", catalog, nil, core.PublishVerifiedTargets)
	if err != nil || len(targets) != 0 {
		t.Fatalf("unverified targets published: targets=%+v err=%v", targets, err)
	}
}
