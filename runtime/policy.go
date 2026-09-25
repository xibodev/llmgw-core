package runtime

import (
	"sync"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

// PolicyFactory returns the resilience policy of one instance from a
// settings snapshot. It constructs and must not block: the Runtime calls it
// under its lock when it binds the instance's provider, once per instance
// and settings generation.
type PolicyFactory[S any] func(settings S, instance string) Policy

// Policy is how the Runtime guards the provider of one instance with
// execution.Resilient: how often an operation is tried, and the
// instance's circuit. The zero Policy guards nothing, and the provider is
// used as built.
//
// The gateway's per-provider policy maps onto it field for field: Retry
// is retry_max_attempts with execution.ExponentialBackoff of the backoff
// settings; Circuit is circuit_failure_threshold and
// circuit_cooldown_seconds with IgnoreRetryAfter; and Classify is
// execution.TransientClassification.
type Policy struct {
	// Retry repeats an operation whose failure permits a repeat. Below two
	// attempts nothing repeats.
	Retry execution.Retry
	// Circuit is the instance's circuit policy. The Runtime keeps every
	// instance's circuit in one execution.HealthTracker on its clock, keyed
	// by instance, so a circuit is shared by every caller and outlives
	// settings changes. A FailureThreshold of zero or less leaves the
	// instance without a circuit: nothing is recorded or refused, Retry-After
	// cooldowns included.
	Circuit execution.HealthPolicy
	// Classify and Repeatable are execution.Policy's.
	Classify   func(error) core.ProviderErrorClassification
	Repeatable func(core.Request) bool
}

// circuits holds the circuit of every instance a Runtime guards. Its zero
// value is ready for use.
type circuits struct {
	mu       sync.Mutex
	policies map[string]execution.HealthPolicy
	tracker  *execution.HealthTracker
}

// guard puts policy in effect for instance and returns the tracker that
// holds its circuit.
func (c *circuits) guard(instance string, policy execution.HealthPolicy, now func() time.Time) *execution.HealthTracker {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.tracker == nil {
		c.policies = map[string]execution.HealthPolicy{}
		c.tracker = execution.NewHealthTracker(execution.HealthOptions{Policy: c.policy, Now: now})
	}
	c.policies[instance] = policy
	return c.tracker
}

// policy is the tracker's policy of an instance: the one it was last
// bound with.
func (c *circuits) policy(instance string) execution.HealthPolicy {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.policies[instance]
}

// resilient applies instance's policy to the provider bind built. It runs
// under the Runtime's lock.
func (r *Runtime[S]) resilient(provider core.Provider, instance string) core.Provider {
	if r.options.Policy == nil {
		return provider
	}
	policy := r.options.Policy(r.settings, instance)
	guarded := execution.Policy{
		Retry: policy.Retry, Classify: policy.Classify, Repeatable: policy.Repeatable, Now: r.options.Now,
	}
	if policy.Circuit.FailureThreshold > 0 {
		guarded.Health = r.circuits.guard(instance, policy.Circuit, r.options.Now)
	} else if policy.Retry.Attempts < 2 {
		return provider
	}
	return execution.Resilient(provider, instance, guarded)
}
