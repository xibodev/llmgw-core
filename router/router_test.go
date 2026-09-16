package router_test

import (
	"context"
	"errors"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/router"
)

type mockPolicy struct {
	deniedProvider string
}

func (m mockPolicy) Allows(ctx context.Context, principal *core.Principal, target core.Target) (bool, string) {
	if target.Provider == m.deniedProvider {
		return false, "policy blocked"
	}
	return true, ""
}

func TestRouterResolutionDirect(t *testing.T) {
	r := router.NewRouter(nil, nil)
	targets, err := r.Resolve(context.Background(), nil, "openai/gpt-4o")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(targets) != 1 || targets[0].Provider != "openai" || targets[0].Model != "gpt-4o" {
		t.Fatalf("unexpected targets: %+v", targets)
	}
}

func TestRouterResolutionRoutePolicyAndCircuitBreaker(t *testing.T) {
	routes := map[string]core.RouteConfig{
		"smart": {
			Targets: []core.Target{
				{Provider: "copilot", Model: "claude-opus-5"},
				{Provider: "groq", Model: "llama-3.3-70b"},
				{Provider: "openai", Model: "gpt-4o"},
			},
		},
	}

	policy := mockPolicy{deniedProvider: "groq"}
	r := router.NewRouter(routes, policy)

	// Resolve initially: copilot and openai should pass (groq filtered by policy)
	targets, err := r.Resolve(context.Background(), nil, "smart")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(targets) != 2 || targets[0].Provider != "copilot" || targets[1].Provider != "openai" {
		t.Fatalf("unexpected filtered targets: %+v", targets)
	}

	// Trip circuit breaker for copilot
	for i := 0; i < 5; i++ {
		r.RecordResult("copilot", errors.New("upstream timeout"))
	}

	// Now copilot is tripped, only openai remains
	targets, err = r.Resolve(context.Background(), nil, "smart")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(targets) != 1 || targets[0].Provider != "openai" {
		t.Fatalf("expected only openai after trip, got: %+v", targets)
	}

	// Reset circuit breaker on success
	r.RecordResult("copilot", nil)
	targets, err = r.Resolve(context.Background(), nil, "smart")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(targets) != 2 {
		t.Fatalf("expected copilot recovered, got: %+v", targets)
	}

	res, err := r.ResolveResolution(context.Background(), nil, "smart")
	if err != nil {
		t.Fatalf("unexpected error on ResolveResolution: %v", err)
	}
	if res.Category != "smart" || len(res.Targets) != 2 {
		t.Fatalf("unexpected resolution: %+v", res)
	}
}
