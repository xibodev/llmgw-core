package execution_test

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

// invocation mirrors the gateway's provider invocation error, whose routing
// metadata follows from its status and flags: 408, 429 and transient 5xx
// statuses are retryable, every retryable failure is also a circuit
// failure, a statusless failure may fail over, and 401 and 403 never do.
type invocation struct {
	status         int
	retryable      bool // for a statusless failure
	failover       bool
	circuitFailure bool
	retryAfter     time.Duration
}

func (e *invocation) Error() string { return "upstream invocation failed" }

func (e *invocation) ProviderErrorClassification() core.ProviderErrorClassification {
	retryable := e.retryable
	if e.status != 0 {
		retryable = slices.Contains([]int{408, 429, 500, 502, 503, 504}, e.status)
	}
	return core.ProviderErrorClassification{
		StatusCode:       e.status,
		Retryable:        retryable,
		FailoverEligible: e.status != 401 && e.status != 403 && (e.failover || e.status == 0 || retryable),
		CircuitFailure:   e.circuitFailure || retryable,
		RetryAfter:       e.retryAfter,
	}
}

// gatewayObserve reads health as the gateway's breaker does: a circuit
// failure counts, any other invocation error ends the streak, and an error
// that never reached the upstream says nothing.
func gatewayObserve(err error) execution.Observation {
	var invocationErr *invocation
	switch {
	case err == nil:
		return execution.Observation{Effect: execution.EffectSuccess}
	case !errors.As(err, &invocationErr):
		return execution.Observation{}
	case core.ClassifyError(err).CircuitFailure:
		return execution.Observation{Effect: execution.EffectFailure}
	}
	return execution.Observation{Effect: execution.EffectSuccess}
}

// circuit is a provider's circuit settings as the gateway configures them.
type circuit struct {
	threshold       int
	cooldownSeconds float64
}

func gatewayTracker(clock *clock, defaults circuit, overrides map[string]circuit) *execution.HealthTracker {
	return execution.NewHealthTracker(execution.HealthOptions{
		Policy: func(provider string) execution.HealthPolicy {
			settings, ok := overrides[provider]
			if !ok {
				settings = defaults
			}
			return execution.HealthPolicy{
				FailureThreshold: settings.threshold,
				OpenDuration:     time.Duration(settings.cooldownSeconds * float64(time.Second)),
				IgnoreRetryAfter: true,
			}
		},
		Observe: gatewayObserve,
		Now:     clock.Now,
	})
}

// gatewayRetry repeats a provider as the gateway's resilience wrapper does:
// up to its attempts, waiting the larger of the exponential backoff and the
// upstream's Retry-After.
func gatewayRetry(attempts int, initialSeconds, multiplier, maxSeconds float64) execution.Retry {
	return execution.Retry{Attempts: attempts, Delay: func(failed int, err error) time.Duration {
		backoff := math.Min(initialSeconds*math.Pow(multiplier, float64(failed-1)), maxSeconds)
		return max(time.Duration(backoff*float64(time.Second)), core.ClassifyError(err).RetryAfter)
	}}
}

// TestHealthTrackerExpressesGatewayBreaker maps the gateway's per-provider
// circuit breaker onto the primitives without changing its behavior:
//
//   - Key: the provider instance's name.
//   - HealthPolicy, per provider so overrides apply: FailureThreshold from
//     circuit_failure_threshold, where zero disables the breaker as it does
//     today; OpenDuration from circuit_cooldown_seconds; no FailureWindow,
//     since a streak never expires; and IgnoreRetryAfter, since Retry-After
//     only lengthens the wait before repeating the same provider.
//   - Observe: gatewayObserve. A circuit failure, which the gateway extends
//     to every retryable status, counts; any other invocation error ends the
//     streak, statusless ones included; an error that never reached the
//     upstream, such as a configuration error, changes nothing.
//   - Half-open: once the cooldown passes the circuit admits requests with
//     the streak kept, so one failure reopens it and a success closes it.
//   - Executor: Key is the provider; Retry is gatewayRetry with
//     retry_max_attempts, or one attempt for a stateful Responses payload, so
//     an exhausted provider counts as one failure, as it does today. A
//     skipped provider replaces the 503 that an open circuit answers today,
//     which the chain already failed over.
func TestHealthTrackerExpressesGatewayBreaker(t *testing.T) {
	t.Parallel()

	t.Run("a definitive failure does not open the circuit", func(t *testing.T) {
		t.Parallel()
		health := gatewayTracker(newClock(), circuit{threshold: 1, cooldownSeconds: 60}, nil)
		for range 2 {
			assertAvailable(t, health, "provider")
			health.Record("provider", &invocation{status: 400})
		}
		assertAvailable(t, health, "provider")
	})

	t.Run("a definitive failure breaks the transient streak", func(t *testing.T) {
		t.Parallel()
		health := gatewayTracker(newClock(), circuit{threshold: 2, cooldownSeconds: 60}, nil)
		for _, status := range []int{503, 400, 503} {
			assertAvailable(t, health, "provider")
			health.Record("provider", &invocation{status: status})
		}
		assertAvailable(t, health, "provider")
		health.Record("provider", &invocation{status: 503})
		if available, _ := health.Available("provider"); available {
			t.Fatal("two transient failures in a row did not open the circuit")
		}
	})

	t.Run("a statusless invocation error ends the streak", func(t *testing.T) {
		t.Parallel()
		health := gatewayTracker(newClock(), circuit{threshold: 2, cooldownSeconds: 60}, nil)
		health.Record("provider", &invocation{status: 503})
		health.Record("provider", &invocation{})
		health.Record("provider", &invocation{status: 503})
		assertAvailable(t, health, "provider")
	})

	t.Run("an error that never reached the upstream changes nothing", func(t *testing.T) {
		t.Parallel()
		health := gatewayTracker(newClock(), circuit{threshold: 2, cooldownSeconds: 60}, nil)
		health.Record("provider", &invocation{status: 503})
		health.Record("provider", core.NewConfigurationError("provider is not configured", nil))
		health.Record("provider", &invocation{status: 503})
		if available, _ := health.Available("provider"); available {
			t.Fatal("a configuration error ended the streak")
		}
	})

	t.Run("a malformed response opens the circuit without a repeat", func(t *testing.T) {
		t.Parallel()
		health := gatewayTracker(newClock(), circuit{threshold: 1, cooldownSeconds: 60}, nil)
		script := &calls{outcomes: map[string][]error{"bad": {&invocation{circuitFailure: true}}}}
		executor := execution.Executor[string]{Health: health, Key: identity, Retry: gatewayRetry(3, 0, 2, 0)}
		result, err := execution.Execute(context.Background(), executor, []string{"bad", "good"}, script.run)
		if err != nil || result.Candidate != "good" || result.Attempts[0].Tries != 1 {
			t.Fatalf("result=%+v err=%v", result, err)
		}
		if available, _ := health.Available("bad"); available {
			t.Fatal("a malformed response did not open the circuit")
		}
	})

	t.Run("an open circuit is skipped", func(t *testing.T) {
		t.Parallel()
		health := gatewayTracker(newClock(), circuit{threshold: 1, cooldownSeconds: 60}, nil)
		script := &calls{outcomes: map[string][]error{"bad": {&invocation{status: 503}}}}
		executor := execution.Executor[string]{Health: health, Key: identity, Retry: gatewayRetry(1, 0, 2, 0)}
		for range 2 {
			if result, err := execution.Execute(context.Background(), executor, []string{"bad", "good"}, script.run); err != nil || result.Candidate != "good" {
				t.Fatalf("result=%+v err=%v", result, err)
			}
		}
		if got := script.made(); !slices.Equal(got, []string{"bad", "good", "good"}) {
			t.Fatalf("calls=%v, want the open circuit skipped", got)
		}
	})

	t.Run("half-open after the cooldown", func(t *testing.T) {
		t.Parallel()
		clock := newClock()
		health := gatewayTracker(clock, circuit{threshold: 2, cooldownSeconds: 60}, nil)
		health.Record("provider", &invocation{status: 502})
		health.Record("provider", &invocation{status: 502})
		assertUnavailableUntil(t, health, "provider", clock.Now().Add(time.Minute))
		clock.Advance(time.Minute)
		assertAvailable(t, health, "provider")
		health.Record("provider", &invocation{status: 502})
		assertUnavailableUntil(t, health, "provider", clock.Now().Add(time.Minute))
		clock.Advance(time.Minute)
		health.Record("provider", nil)
		health.Record("provider", &invocation{status: 502})
		assertAvailable(t, health, "provider")
	})

	t.Run("repeats count as one failure", func(t *testing.T) {
		t.Parallel()
		clock := newClock()
		health := gatewayTracker(clock, circuit{threshold: 2, cooldownSeconds: 60}, nil)
		script := &calls{outcomes: map[string][]error{"bad": {&invocation{status: 429, retryAfter: 5 * time.Second}, &invocation{status: 503}}}}
		executor := execution.Executor[string]{Health: health, Key: identity, Now: clock.Now, Sleep: clock.Sleep, Retry: gatewayRetry(3, 1, 2, 10)}
		result, err := execution.Execute(context.Background(), executor, []string{"bad", "good"}, script.run)
		if err != nil || result.Candidate != "good" || result.Attempts[0].Tries != 3 {
			t.Fatalf("result=%+v err=%v", result, err)
		}
		// The waits are max(1s, Retry-After 5s), then the 2s backoff.
		if result.Attempts[0].Duration != 7*time.Second {
			t.Fatalf("waits took %v, want 7s", result.Attempts[0].Duration)
		}
		if state := health.State("bad"); state.Streak != 1 {
			t.Fatalf("an exhausted provider counted %d failures, want one", state.Streak)
		}
		assertAvailable(t, health, "bad")
	})

	t.Run("overrides apply per provider and zero disables", func(t *testing.T) {
		t.Parallel()
		health := gatewayTracker(newClock(), circuit{cooldownSeconds: 30}, map[string]circuit{"copilot": {threshold: 2, cooldownSeconds: 30}})
		for range 10 {
			health.Record("disabled", &invocation{status: 503})
		}
		assertAvailable(t, health, "disabled")
		health.Record("copilot", &invocation{status: 503})
		health.Record("copilot", &invocation{status: 503})
		if available, _ := health.Available("copilot"); available {
			t.Fatal("the override's threshold did not apply")
		}
	})

	t.Run("Retry-After starts no cooldown", func(t *testing.T) {
		t.Parallel()
		health := gatewayTracker(newClock(), circuit{threshold: 2, cooldownSeconds: 60}, nil)
		health.Record("provider", &invocation{status: 429, retryAfter: 30 * time.Second})
		assertAvailable(t, health, "provider")
		if state := health.State("provider"); state.Streak != 1 {
			t.Fatalf("a 429 is a circuit failure at the gateway: %+v", state)
		}
	})
}

// failover mirrors Facet Studio's classified provider error. Every reason
// but format and context overflow is retriable, and a retriable failure
// both fails over and cools the candidate down.
type failover struct {
	reason string
	status int
}

func (e *failover) Error() string { return "failover(" + e.reason + ")" }

func (e *failover) retriable() bool {
	return e.reason != "format" && e.reason != "context_overflow"
}

func (e *failover) ProviderErrorClassification() core.ProviderErrorClassification {
	retriable := e.retriable()
	return core.ProviderErrorClassification{StatusCode: e.status, FailoverEligible: retriable, CircuitFailure: retriable}
}

// facetObserve reads health as Facet Studio's fallback does: a retriable
// failure cools down under its reason, and nothing else is ever recorded,
// including a streaming failure after visible output.
func facetObserve(err error) execution.Observation {
	var afterOutput *execution.AfterOutputError
	var classified *failover
	switch {
	case err == nil:
		return execution.Observation{Effect: execution.EffectSuccess}
	case errors.As(err, &afterOutput), !errors.As(err, &classified), !classified.retriable():
		return execution.Observation{}
	}
	return execution.Observation{Effect: execution.EffectFailure, Class: classified.reason}
}

// facetBackoff is Facet Studio's cooldown: min(1h, 1m·5^min(n-1, 3)) for
// the streak's n failures, or min(24h, 5h·2^min(n-1, 10)) for the n billing
// failures of a billing failure.
func facetBackoff(failure execution.Failure) time.Duration {
	if failure.Class == "billing" {
		exponent := min(max(failure.ClassFailures, 1)-1, 10)
		return time.Duration(math.Min(float64(24*time.Hour), float64(5*time.Hour)*math.Pow(2, float64(exponent))))
	}
	exponent := min(max(failure.Streak, 1)-1, 3)
	return min(time.Hour, time.Minute*time.Duration(math.Pow(5, float64(exponent))))
}

func facetTracker(clock *clock) *execution.HealthTracker {
	return execution.NewHealthTracker(execution.HealthOptions{
		Policy: func(string) execution.HealthPolicy {
			return execution.HealthPolicy{FailureThreshold: 1, FailureWindow: 24 * time.Hour, Backoff: facetBackoff, IgnoreRetryAfter: true}
		},
		Observe: facetObserve,
		Now:     clock.Now,
	})
}

func remaining(health *execution.HealthTracker, clock *clock, key string) time.Duration {
	if available, until := health.Available(key); !available {
		return until.Sub(clock.Now())
	}
	return 0
}

// TestHealthTrackerExpressesFacetStudioCooldown maps Facet Studio's cooldown
// tracker onto a HealthTracker without changing its behavior, case by case
// with its own tests:
//
//   - Key: the candidate's stable key.
//   - HealthPolicy: FailureThreshold 1, since every retriable failure cools
//     the candidate down; FailureWindow 24 hours; IgnoreRetryAfter; and
//     facetBackoff. The open deadline only moves later, which is how a
//     billing cooldown outlasts a standard one.
//   - Observe: facetObserve. The reason is the failure's class, so a billing
//     failure's cooldown grows with its own count, while the streak counts
//     every reason, as its error count does.
//   - IsAvailable and CooldownRemaining are Available; MarkSuccess is Record
//     with nil; ErrorCount and FailureCount are HealthState's Streak and
//     ClassFailures.
func TestHealthTrackerExpressesFacetStudioCooldown(t *testing.T) {
	t.Parallel()
	failure := func(reason string) error { return &failover{reason: reason} }

	t.Run("standard escalation", func(t *testing.T) {
		t.Parallel()
		clock := newClock()
		health := facetTracker(clock)
		assertAvailable(t, health, "openai")
		health.Record("openai", failure("rate_limit"))
		if available, _ := health.Available("openai"); available {
			t.Fatal("available after the first error")
		}
		clock.Advance(61 * time.Second)
		assertAvailable(t, health, "openai")
		health.Record("openai", failure("rate_limit"))
		clock.Advance(4 * time.Minute)
		if available, _ := health.Available("openai"); available {
			t.Fatal("available within the 5 minute cooldown")
		}
		clock.Advance(2 * time.Minute)
		assertAvailable(t, health, "openai")
	})

	for name, tc := range map[string]struct {
		reason string
		want   []time.Duration
	}{
		"standard cap": {"rate_limit", []time.Duration{time.Minute, 5 * time.Minute, 25 * time.Minute, time.Hour, time.Hour}},
		"billing cap":  {"billing", []time.Duration{5 * time.Hour, 10 * time.Hour, 20 * time.Hour, 24 * time.Hour, 24 * time.Hour}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			clock := newClock()
			health := facetTracker(clock)
			for _, want := range tc.want {
				health.Record("openai", failure(tc.reason))
				if got := remaining(health, clock, "openai"); got != want {
					t.Fatalf("cooldown=%v, want %v", got, want)
				}
				clock.Advance(want)
			}
		})
	}

	t.Run("billing escalation", func(t *testing.T) {
		t.Parallel()
		clock := newClock()
		health := facetTracker(clock)
		health.Record("openai", failure("billing"))
		clock.Advance(4 * time.Hour)
		if available, _ := health.Available("openai"); available {
			t.Fatal("available within the 5 hour billing cooldown")
		}
		clock.Advance(time.Hour + time.Second)
		assertAvailable(t, health, "openai")
	})

	t.Run("success reset", func(t *testing.T) {
		t.Parallel()
		health := facetTracker(newClock())
		health.Record("openai", failure("rate_limit"))
		health.Record("openai", failure("billing"))
		if state := health.State("openai"); state.Streak != 2 {
			t.Fatalf("error count=%d, want 2", state.Streak)
		}
		health.Record("openai", nil)
		state := health.State("openai")
		if state.Streak != 0 || state.ClassFailures["rate_limit"] != 0 || state.ClassFailures["billing"] != 0 {
			t.Fatalf("counts after success=%+v", state)
		}
		assertAvailable(t, health, "openai")
	})

	t.Run("failure window reset", func(t *testing.T) {
		t.Parallel()
		clock := newClock()
		health := facetTracker(clock)
		start := clock.Now()
		for range 4 {
			health.Record("openai", failure("rate_limit"))
			clock.Advance(2 * time.Second)
		}
		if state := health.State("openai"); state.Streak != 4 {
			t.Fatalf("error count=%d, want 4", state.Streak)
		}
		clock.Advance(start.Add(25 * time.Hour).Sub(clock.Now()))
		health.Record("openai", failure("rate_limit"))
		if state := health.State("openai"); state.Streak != 1 {
			t.Fatalf("error count after the window=%d, want 1", state.Streak)
		}
	})

	t.Run("per reason tracking", func(t *testing.T) {
		t.Parallel()
		health := facetTracker(newClock())
		for _, reason := range []string{"rate_limit", "rate_limit", "billing", "auth"} {
			health.Record("openai", failure(reason))
		}
		state := health.State("openai")
		if state.ClassFailures["rate_limit"] != 2 || state.ClassFailures["billing"] != 1 || state.ClassFailures["auth"] != 1 || state.Streak != 4 {
			t.Fatalf("state=%+v", state)
		}
	})

	t.Run("billing takes precedence", func(t *testing.T) {
		t.Parallel()
		clock := newClock()
		health := facetTracker(clock)
		health.Record("openai", failure("rate_limit"))
		health.Record("openai", failure("billing"))
		clock.Advance(2 * time.Minute)
		if available, _ := health.Available("openai"); available {
			t.Fatal("the billing cooldown did not outlast the standard one")
		}
		clock.Advance(5*time.Hour + time.Second - 2*time.Minute)
		assertAvailable(t, health, "openai")
	})

	t.Run("cooldown remaining", func(t *testing.T) {
		t.Parallel()
		clock := newClock()
		health := facetTracker(clock)
		if remaining(health, clock, "openai") != 0 {
			t.Fatal("a new provider has a cooldown")
		}
		health.Record("openai", failure("rate_limit"))
		clock.Advance(30 * time.Second)
		if got := remaining(health, clock, "openai"); got <= 0 || got > time.Minute {
			t.Fatalf("remaining=%v, want about 30s", got)
		}
	})

	t.Run("multiple providers", func(t *testing.T) {
		t.Parallel()
		health := facetTracker(newClock())
		health.Record("unknown", nil)
		health.Record("openai", failure("rate_limit"))
		health.Record("anthropic", failure("billing"))
		for _, key := range []string{"openai", "anthropic"} {
			if available, _ := health.Available(key); available {
				t.Fatalf("%s is not in cooldown", key)
			}
		}
		assertAvailable(t, health, "groq")
		assertAvailable(t, health, "unknown")
	})
}

// TestExecuteExpressesFacetStudioFallback maps Facet Studio's fallback chain
// onto Execute with the tracker above. Key is the candidate's stable key,
// and its classified errors report their routing metadata, so:
//
//   - A candidate in cooldown is skipped; the entry's Until gives the time
//     remaining that its skipped attempt reports.
//   - A retriable failure cools the candidate down and falls back.
//   - A non-retriable or unclassifiable error, or a streaming failure after
//     visible output returned as an *AfterOutputError, stops the chain and
//     records nothing.
//   - A success resets the candidate.
//   - A candidate its local rate limiter holds back returns an error that
//     permits failover and says nothing about health; the last candidate
//     waits for its token inside run instead, knowing its position from the
//     candidate the product passes.
//   - Its image chain is an Executor without Health whose run returns a
//     terminal error for image dimension and size errors.
//
// One deliberate difference: a caller's expired deadline stops the chain
// and records nothing, as cancellation does, where the chain today goes on
// and cools every remaining candidate down as a timeout.
func TestExecuteExpressesFacetStudioFallback(t *testing.T) {
	t.Parallel()
	run := func(outcomes map[string]error) func(context.Context, string) (string, error) {
		return func(_ context.Context, candidate string) (string, error) {
			if err := outcomes[candidate]; err != nil {
				return "", err
			}
			return candidate, nil
		}
	}

	t.Run("cooldown skip and retriable fallback", func(t *testing.T) {
		t.Parallel()
		clock := newClock()
		health := facetTracker(clock)
		executor := execution.Executor[string]{Health: health, Key: identity, Now: clock.Now}
		result, err := execution.Execute(context.Background(), executor, []string{"primary", "fallback"},
			run(map[string]error{"primary": &failover{reason: "rate_limit", status: 429}}))
		if err != nil || result.Candidate != "fallback" {
			t.Fatalf("result=%+v err=%v", result, err)
		}
		assertUnavailableUntil(t, health, "primary", clock.Now().Add(time.Minute))

		clock.Advance(20 * time.Second)
		result, err = execution.Execute(context.Background(), executor, []string{"primary", "fallback"}, run(nil))
		if err != nil || result.Candidate != "fallback" || !result.Attempts[0].Unavailable ||
			result.Attempts[0].Until.Sub(clock.Now()) != 40*time.Second {
			t.Fatalf("result=%+v err=%v", result, err)
		}
	})

	for name, stop := range map[string]error{
		"non-retriable":        &failover{reason: "format", status: 400},
		"unclassifiable":       errors.New("unclassifiable"),
		"after visible output": &execution.AfterOutputError{Err: &failover{reason: "network"}},
	} {
		t.Run(name+" stops unrecorded", func(t *testing.T) {
			t.Parallel()
			health := facetTracker(newClock())
			executor := execution.Executor[string]{Health: health, Key: identity}
			result, err := execution.Execute(context.Background(), executor, []string{"primary", "fallback"}, run(map[string]error{"primary": stop}))
			if err != stop || len(result.Attempts) != 1 {
				t.Fatalf("result=%+v err=%v", result, err)
			}
			assertAvailable(t, health, "primary")
		})
	}

	t.Run("local rate limit moves on unrecorded", func(t *testing.T) {
		t.Parallel()
		health := facetTracker(newClock())
		saturated := &core.ProviderError{Message: "waiting for local rate limit token",
			Classification: core.ProviderErrorClassification{FailoverEligible: true}}
		executor := execution.Executor[string]{Health: health, Key: identity}
		result, err := execution.Execute(context.Background(), executor, []string{"primary", "fallback"}, run(map[string]error{"primary": saturated}))
		if err != nil || result.Candidate != "fallback" {
			t.Fatalf("result=%+v err=%v", result, err)
		}
		assertAvailable(t, health, "primary")
	})

	t.Run("success resets", func(t *testing.T) {
		t.Parallel()
		clock := newClock()
		health := facetTracker(clock)
		for range 3 {
			health.Record("primary", &failover{reason: "timeout"})
			clock.Advance(time.Hour)
		}
		executor := execution.Executor[string]{Health: health, Key: identity}
		if _, err := execution.Execute(context.Background(), executor, []string{"primary"}, run(nil)); err != nil {
			t.Fatal(err)
		}
		if state := health.State("primary"); state.Streak != 0 {
			t.Fatalf("state=%+v", state)
		}
	})

	t.Run("every candidate in cooldown", func(t *testing.T) {
		t.Parallel()
		health := facetTracker(newClock())
		health.Record("primary", &failover{reason: "overloaded"})
		health.Record("fallback", &failover{reason: "billing"})
		executor := execution.Executor[string]{Health: health, Key: identity}
		result, err := execution.Execute(context.Background(), executor, []string{"primary", "fallback"}, run(nil))
		var unavailable *execution.UnavailableError
		if !errors.As(err, &unavailable) || len(result.Attempts) != 2 {
			t.Fatalf("result=%+v err=%v", result, err)
		}
	})

	t.Run("the caller's end stops unrecorded", func(t *testing.T) {
		t.Parallel()
		health := facetTracker(newClock())
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		defer cancel()
		timedOut := func(ctx context.Context, candidate string) (string, error) {
			<-ctx.Done()
			return "", &failover{reason: "timeout"}
		}
		executor := execution.Executor[string]{Health: health, Key: identity}
		if _, err := execution.Execute(ctx, executor, []string{"primary", "fallback"}, timedOut); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("err=%v", err)
		}
		assertAvailable(t, health, "primary")
		assertAvailable(t, health, "fallback")
	})
}
