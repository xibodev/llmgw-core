package core_test

import (
	"context"
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

func TestPersonalAPIKeyIsNotASubscriptionConnection(t *testing.T) {
	connection := core.ProviderConnection{
		ProviderID: "openai", Kind: core.ProviderConnectionPersonal,
		AuthKind: core.ProviderAuthAPIKey, OwnerID: "owner-1",
	}
	if err := connection.Validate(); err != nil {
		t.Fatal(err)
	}
	if connection.Kind == core.ProviderConnectionPersonalSubscription {
		t.Fatal("personal API key was classified as a subscription")
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
		{name: "redirect", failure: core.ProviderFailure{StatusCode: 302}, class: core.ProviderErrorUpstream},
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

func TestProviderOrchestratorSeparatesEvidenceAndPublishesByPolicy(t *testing.T) {
	orchestrator := core.NewProviderOrchestrator()
	completed := make([]core.Target, 0, 1)
	err := orchestrator.Register("fixture", core.ProviderAdapter{
		ValidateAuthentication: func(context.Context, core.ProviderConnection) error { return nil },
		DiscoverModels: func(context.Context, core.ProviderConnection) ([]core.ModelInfo, error) {
			return []core.ModelInfo{{ID: "model-a"}, {ID: "model-b"}}, nil
		},
		SelectProbeTargets: func(core.ProviderConnection, []core.ModelInfo) []core.Target {
			return []core.Target{{Model: "model-b"}}
		},
		Complete: func(_ context.Context, _ core.ProviderConnection, target core.Target, _ map[string]any) (map[string]any, error) {
			completed = append(completed, target)
			return map[string]any{"ok": true}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	connection := core.ProviderConnection{ProviderID: "fixture", Kind: core.ProviderConnectionSystem, AuthKind: core.ProviderAuthAPIKey}
	result, err := orchestrator.Connect(context.Background(), core.ProviderConnectRequest{
		Connection: connection, PublicationPolicy: core.PublishVerifiedTargets,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Catalog.Status != core.CatalogDiscovered || len(result.Catalog.Models) != 2 {
		t.Fatalf("catalog evidence=%+v", result.Catalog)
	}
	if len(result.Probes) != 1 || result.Probes[0].Status != core.CompletionVerified || result.Probes[0].Target.Provider != "fixture" {
		t.Fatalf("probe evidence=%+v", result.Probes)
	}
	if result.Health.Status != core.ProviderHealthHealthy || result.Health.ErrorClass != core.ProviderErrorNone {
		t.Fatalf("health evidence=%+v", result.Health)
	}
	if len(result.Targets) != 1 || result.Targets[0] != (core.Target{Provider: "fixture", Model: "model-b"}) {
		t.Fatalf("published targets=%+v", result.Targets)
	}
	if len(completed) != 1 || completed[0] != result.Targets[0] {
		t.Fatalf("runtime calls=%+v", completed)
	}

	result, err = orchestrator.Connect(context.Background(), core.ProviderConnectRequest{
		Connection: connection, PublicationPolicy: core.PublishDiscoveredTargets,
	})
	if err != nil || len(result.Targets) != 2 {
		t.Fatalf("discovered targets=%+v err=%v", result.Targets, err)
	}
}

func TestProviderOrchestratorClassifiesProbeFailure(t *testing.T) {
	orchestrator := core.NewProviderOrchestrator()
	err := orchestrator.Register("limited", core.ProviderAdapter{
		DiscoverModels: func(context.Context, core.ProviderConnection) ([]core.ModelInfo, error) {
			return []core.ModelInfo{{ID: "model-a"}}, nil
		},
		Complete: func(context.Context, core.ProviderConnection, core.Target, map[string]any) (map[string]any, error) {
			return nil, core.NewProviderOperationError("probe", 429, "9", nil)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := orchestrator.Connect(context.Background(), core.ProviderConnectRequest{
		Connection:              core.ProviderConnection{ProviderID: "limited", Kind: core.ProviderConnectionSystem, AuthKind: core.ProviderAuthAPIKey},
		AuthenticationValidated: true,
		PublicationPolicy:       core.PublishVerifiedTargets,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Probes) != 1 || result.Probes[0].Status != core.CompletionFailed || len(result.Targets) != 0 {
		t.Fatalf("probe result=%+v targets=%+v", result.Probes, result.Targets)
	}
	if result.Health.ErrorClass != core.ProviderErrorRateLimited || !result.Health.Retryable || result.Health.RetryAfter != 9*time.Second {
		t.Fatalf("health evidence=%+v", result.Health)
	}
}

func TestProviderOrchestratorRequiresRegisteredCodexAdapter(t *testing.T) {
	connection := core.ProviderConnection{
		ProviderID: "codex", Kind: core.ProviderConnectionPersonalSubscription,
		AuthKind: core.ProviderAuthOAuthDevice, OwnerID: "owner-1",
	}
	_, err := core.NewProviderOrchestrator().Connect(context.Background(), core.ProviderConnectRequest{
		Connection: connection, PublicationPolicy: core.PublishVerifiedTargets,
	})
	if err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Fatalf("unregistered Codex adapter error=%v", err)
	}
}
