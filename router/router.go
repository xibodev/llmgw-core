package router

import (
	"context"
	"fmt"
	"sync"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// CircuitBreaker tracks failure rates and opens when an upstream fails repeatedly.
type CircuitBreaker struct {
	mu          sync.Mutex
	failures    map[string]int
	lastFailure map[string]time.Time
	threshold   int
	cooldown    time.Duration
}

func NewCircuitBreaker(threshold int, cooldown time.Duration) *CircuitBreaker {
	if threshold <= 0 {
		threshold = 5
	}
	if cooldown <= 0 {
		cooldown = 30 * time.Second
	}
	return &CircuitBreaker{
		failures:    make(map[string]int),
		lastFailure: make(map[string]time.Time),
		threshold:   threshold,
		cooldown:    cooldown,
	}
}

func (cb *CircuitBreaker) Allow(provider string) bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	fails := cb.failures[provider]
	if fails < cb.threshold {
		return true
	}

	last := cb.lastFailure[provider]
	if time.Since(last) > cb.cooldown {
		// Cooldown elapsed: allow half-open probe
		return true
	}
	return false
}

func (cb *CircuitBreaker) RecordSuccess(provider string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	delete(cb.failures, provider)
	delete(cb.lastFailure, provider)
}

func (cb *CircuitBreaker) RecordFailure(provider string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.failures[provider]++
	cb.lastFailure[provider] = time.Now()
}

// Router resolves requested model strings into ordered targets.
type Router struct {
	routes   map[string]core.RouteConfig
	cb       *CircuitBreaker
	policy   core.PolicyGate
}

func NewRouter(routes map[string]core.RouteConfig, policy core.PolicyGate) *Router {
	if policy == nil {
		policy = core.AllowAllPolicy{}
	}
	return &Router{
		routes: routes,
		cb:     NewCircuitBreaker(5, 30*time.Second),
		policy: policy,
	}
}

// Resolve returns ordered valid targets for a given model request and principal.
func (r *Router) Resolve(ctx context.Context, principal *core.Principal, requested string) ([]core.Target, error) {
	if route, exists := r.routes[requested]; exists {
		var candidates []core.Target
		for _, target := range route.Targets {
			if !r.cb.Allow(target.Provider) {
				continue
			}
			allowed, _ := r.policy.Allows(ctx, principal, target)
			if allowed {
				candidates = append(candidates, target)
			}
		}
		if len(candidates) == 0 {
			return nil, fmt.Errorf("no available targets for route %q (all targets circuit-broken or denied by policy)", requested)
		}
		return candidates, nil
	}

	// Direct provider/model target (e.g. "openai/gpt-4o" or "copilot/claude-opus-5")
	var target core.Target
	var found bool
	for i := 0; i < len(requested); i++ {
		if requested[i] == '/' {
			target = core.Target{
				Provider: requested[:i],
				Model:    requested[i+1:],
			}
			found = true
			break
		}
	}
	if !found {
		// Fallback single model with default provider or direct name
		target = core.Target{
			Provider: "default",
			Model:    requested,
		}
	}

	allowed, reason := r.policy.Allows(ctx, principal, target)
	if !allowed {
		return nil, fmt.Errorf("access denied to target %s/%s: %s", target.Provider, target.Model, reason)
	}

	return []core.Target{target}, nil
}

func (r *Router) RecordResult(provider string, err error) {
	if err == nil {
		r.cb.RecordSuccess(provider)
	} else {
		r.cb.RecordFailure(provider)
	}
}
