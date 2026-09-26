package execution_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

func TestHealthTrackerOpensAtThresholdAndHalfOpens(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := tracker(clock, execution.HealthPolicy{FailureThreshold: 3, OpenDuration: 30 * time.Second})

	health.Record("a", overloaded())
	health.Record("a", overloaded())
	assertAvailable(t, health, "a")
	health.Record("a", overloaded())
	opened := clock.Now()
	assertUnavailableUntil(t, health, "a", opened.Add(30*time.Second))
	assertAvailable(t, health, "b")

	clock.Advance(30*time.Second - time.Nanosecond)
	assertUnavailableUntil(t, health, "a", opened.Add(30*time.Second))
	clock.Advance(time.Nanosecond)
	assertAvailable(t, health, "a")
	if state := health.State("a"); state.Streak != 3 {
		t.Fatalf("a half-open circuit forgot its streak: %+v", state)
	}

	// Half-open: the kept streak lets one failure reopen the circuit.
	health.Record("a", overloaded())
	assertUnavailableUntil(t, health, "a", clock.Now().Add(30*time.Second))

	clock.Advance(30 * time.Second)
	health.Record("a", nil)
	if state := health.State("a"); state.Streak != 0 || !state.OpenUntil.IsZero() || state.ClassFailures != nil {
		t.Fatalf("a success did not close the circuit: %+v", state)
	}
	health.Record("a", overloaded())
	health.Record("a", overloaded())
	assertAvailable(t, health, "a")
}

func TestHealthTrackerSuccessEndsAnOpenCircuit(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := tracker(clock, execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute})
	health.Record("a", overloaded())
	assertUnavailableUntil(t, health, "a", clock.Now().Add(time.Minute))
	// An operation admitted before the circuit opened can still succeed.
	health.Record("a", nil)
	assertAvailable(t, health, "a")
	if state := health.State("a"); state.Streak != 0 || !state.LastFailure.IsZero() {
		t.Fatalf("state after success=%+v", state)
	}
}

func TestHealthTrackerRetryAfterStartsACooldown(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := tracker(clock, execution.HealthPolicy{FailureThreshold: 1, OpenDuration: 30 * time.Second})

	health.Record("throttled", rateLimited(10*time.Second))
	assertUnavailableUntil(t, health, "throttled", clock.Now().Add(10*time.Second))
	if state := health.State("throttled"); state.Streak != 0 || !state.OpenUntil.IsZero() || !state.CooldownUntil.Equal(clock.Now().Add(10*time.Second)) {
		t.Fatalf("a rate limit is not a circuit failure: %+v", state)
	}
	// Neither a success nor a later, shorter Retry-After cuts a cooldown
	// short: the upstream asked callers to wait.
	health.Record("throttled", nil)
	health.Record("throttled", rateLimited(time.Second))
	assertUnavailableUntil(t, health, "throttled", clock.Now().Add(10*time.Second))
	clock.Advance(10 * time.Second)
	assertAvailable(t, health, "throttled")
	health.Record("throttled", nil)
	if state := health.State("throttled"); state.Streak != 0 || !state.CooldownUntil.IsZero() || state.ClassFailures != nil {
		t.Fatalf("a success after the cooldown left state behind: %+v", state)
	}

	// A circuit failure with a longer Retry-After stays out until the later.
	failure := &core.ProviderError{Classification: core.ProviderErrorClassification{
		StatusCode: 503, Retryable: true, FailoverEligible: true, CircuitFailure: true, RetryAfter: 2 * time.Minute}}
	health.Record("both", failure)
	assertUnavailableUntil(t, health, "both", clock.Now().Add(2*time.Minute))
	if state := health.State("both"); !state.OpenUntil.Equal(clock.Now().Add(30 * time.Second)) {
		t.Fatalf("state=%+v", state)
	}
	// A success closes the circuit but leaves the cooldown running.
	health.Record("both", nil)
	assertUnavailableUntil(t, health, "both", clock.Now().Add(2*time.Minute))
	if state := health.State("both"); state.Streak != 0 || !state.OpenUntil.IsZero() {
		t.Fatalf("a success left the circuit open: %+v", state)
	}
	ignoring := tracker(clock, execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute, IgnoreRetryAfter: true})
	ignoring.Record("throttled", rateLimited(time.Hour))
	assertAvailable(t, ignoring, "throttled")
	if state := ignoring.State("throttled"); !state.CooldownUntil.IsZero() {
		t.Fatalf("an ignored Retry-After started a cooldown: %+v", state)
	}
}

func TestHealthTrackerDefinitiveFailureEndsAStreak(t *testing.T) {
	t.Parallel()
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 2, OpenDuration: time.Minute})
	health.Record("a", overloaded())
	health.Record("a", rejected(400))
	health.Record("a", overloaded())
	assertAvailable(t, health, "a")
	health.Record("a", overloaded())
	if available, _ := health.Available("a"); available {
		t.Fatal("two consecutive circuit failures did not open the circuit")
	}
}

func TestHealthTrackerIgnoresOutcomesThatSayNothing(t *testing.T) {
	t.Parallel()
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute})
	canceledAfterStatus := &core.ProviderOperationError{Op: "stream", Failure: core.ProviderFailure{StatusCode: 503, Err: context.Canceled}}
	for name, err := range map[string]error{
		"configuration":         core.NewConfigurationError("provider is not configured", nil),
		"surface":               unsupported(),
		"canceled":              context.Canceled,
		"deadline":              context.DeadlineExceeded,
		"canceled after status": canceledAfterStatus,
		"unclassified":          errors.New("unclassified"),
		"after output":          &execution.AfterOutputError{Err: overloaded()},
	} {
		health.Record(name, err)
		assertAvailable(t, health, name)
		if state := health.State(name); state.Streak != 0 || !state.CooldownUntil.IsZero() {
			t.Fatalf("%s changed health: %+v", name, state)
		}
	}
}

func TestHealthTrackerFailureWindowForgetsAnOldStreak(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := tracker(clock, execution.HealthPolicy{FailureThreshold: 3, OpenDuration: time.Minute, FailureWindow: time.Hour})
	health.Record("a", overloaded())
	health.Record("a", overloaded())
	clock.Advance(time.Hour)
	health.Record("a", overloaded())
	if state := health.State("a"); state.Streak != 3 {
		t.Fatalf("a failure exactly one window later was forgotten: %+v", state)
	}
	clock.Advance(time.Hour + time.Nanosecond)
	health.Record("a", overloaded())
	if state := health.State("a"); state.Streak != 1 || state.ClassFailures[""] != 1 {
		t.Fatalf("a failure after the window extended the old streak: %+v", state)
	}
	assertAvailable(t, health, "a")
}

func TestHealthTrackerBackoffSeesTheStreakAndTheClass(t *testing.T) {
	t.Parallel()
	clock := newClock()
	var seen []execution.Failure
	health := execution.NewHealthTracker(execution.HealthOptions{
		Policy: func(string) execution.HealthPolicy {
			return execution.HealthPolicy{FailureThreshold: 2, Backoff: func(failure execution.Failure) time.Duration {
				seen = append(seen, failure)
				if failure.Class == "billing" {
					return time.Duration(failure.ClassFailures) * time.Hour
				}
				return time.Duration(failure.Streak) * time.Minute
			}}
		},
		Observe: func(err error) execution.Observation {
			if err == nil {
				return execution.Observation{Effect: execution.EffectSuccess}
			}
			if core.ClassifyError(err).StatusCode == 402 {
				return execution.Observation{Effect: execution.EffectFailure, Class: "billing"}
			}
			return execution.Observation{Effect: execution.EffectFailure}
		},
		Now: clock.Now,
	})
	billing := &core.ProviderError{Classification: core.ProviderErrorClassification{StatusCode: 402}}

	health.Record("a", overloaded())
	health.Record("a", overloaded())
	assertUnavailableUntil(t, health, "a", clock.Now().Add(2*time.Minute))
	health.Record("a", billing)
	assertUnavailableUntil(t, health, "a", clock.Now().Add(time.Hour))
	// A standard failure while billing holds the key never shortens it.
	health.Record("a", overloaded())
	assertUnavailableUntil(t, health, "a", clock.Now().Add(time.Hour))

	want := []execution.Failure{
		{Key: "a", Streak: 2, ClassFailures: 2},
		{Key: "a", Class: "billing", Streak: 3, ClassFailures: 1},
		{Key: "a", Streak: 4, ClassFailures: 3},
	}
	if len(seen) != len(want) {
		t.Fatalf("Backoff saw %+v, want %+v", seen, want)
	}
	for i := range want {
		if seen[i] != want[i] {
			t.Fatalf("Backoff saw %+v, want %+v", seen, want)
		}
	}
	if state := health.State("a"); state.ClassFailures[""] != 3 || state.ClassFailures["billing"] != 1 {
		t.Fatalf("class failures=%v", state.ClassFailures)
	}
}

func TestHealthTrackerPoliciesDifferPerKeyAndFollowChanges(t *testing.T) {
	t.Parallel()
	clock := newClock()
	var mu sync.Mutex
	policies := map[string]execution.HealthPolicy{
		"strict":   {FailureThreshold: 1, OpenDuration: time.Minute},
		"lenient":  {FailureThreshold: 3, OpenDuration: time.Minute},
		"disabled": {},
	}
	health := execution.NewHealthTracker(execution.HealthOptions{
		Policy: func(key string) execution.HealthPolicy {
			mu.Lock()
			defer mu.Unlock()
			return policies[key]
		},
		Now: clock.Now,
	})
	for _, key := range []string{"strict", "lenient", "disabled"} {
		health.Record(key, overloaded())
	}
	if available, _ := health.Available("strict"); available {
		t.Fatal("strict opened after one failure")
	}
	assertAvailable(t, health, "lenient")
	for range 10 {
		health.Record("disabled", overloaded())
	}
	assertAvailable(t, health, "disabled")
	if state := health.State("disabled"); state.Streak != 0 {
		t.Fatalf("a policy without circuit breaking counted failures: %+v", state)
	}

	mu.Lock()
	policies["strict"] = execution.HealthPolicy{}
	mu.Unlock()
	assertAvailable(t, health, "strict")
}

func TestHealthTrackerDefaultsToTheRouterBreakerPolicy(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := execution.NewHealthTracker(execution.HealthOptions{Now: clock.Now})
	for range execution.DefaultFailureThreshold - 1 {
		health.Record("a", overloaded())
	}
	assertAvailable(t, health, "a")
	health.Record("a", overloaded())
	assertUnavailableUntil(t, health, "a", clock.Now().Add(execution.DefaultOpenDuration))

	wallClock := execution.NewHealthTracker(execution.HealthOptions{})
	wallClock.Record("a", rateLimited(time.Hour))
	if available, until := wallClock.Available("a"); available || until.Before(time.Now()) {
		t.Fatalf("the real clock did not start a cooldown: available=%v until=%v", available, until)
	}
}

func TestHealthTrackerResetForgetsAKey(t *testing.T) {
	t.Parallel()
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Hour})
	health.Record("a", overloaded())
	health.Record("a", rateLimited(time.Hour))
	health.Reset("a")
	assertAvailable(t, health, "a")
	if state := health.State("a"); state.Streak != 0 || !state.OpenUntil.IsZero() || !state.CooldownUntil.IsZero() {
		t.Fatalf("state after reset=%+v", state)
	}
}

func TestHealthTrackerStateIsACopy(t *testing.T) {
	t.Parallel()
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 5})
	health.Record("a", overloaded())
	health.State("a").ClassFailures[""] = 99
	if state := health.State("a"); state.ClassFailures[""] != 1 {
		t.Fatalf("State shares its map with the tracker: %+v", state)
	}
	if state := health.State("never"); state.Streak != 0 || state.ClassFailures != nil {
		t.Fatalf("an unknown key has state: %+v", state)
	}
}

func TestObserve(t *testing.T) {
	t.Parallel()
	cases := map[string]struct {
		err        error
		effect     execution.Effect
		retryAfter time.Duration
	}{
		"success":            {nil, execution.EffectSuccess, 0},
		"upstream failure":   {overloaded(), execution.EffectFailure, 0},
		"transport failure":  {disconnected(), execution.EffectFailure, 0},
		"bad request":        {rejected(400), execution.EffectSuccess, 0},
		"unauthorized":       {rejected(401), execution.EffectSuccess, 0},
		"rate limited":       {rateLimited(7 * time.Second), execution.EffectSuccess, 7 * time.Second},
		"configuration":      {core.NewConfigurationError("not configured", nil), execution.EffectNeutral, 0},
		"surface":            {unsupported(), execution.EffectNeutral, 0},
		"canceled":           {context.Canceled, execution.EffectNeutral, 0},
		"canceled operation": {&core.ProviderOperationError{Failure: core.ProviderFailure{StatusCode: 500, Err: context.Canceled}}, execution.EffectNeutral, 0},
		"unclassified":       {errors.New("unclassified"), execution.EffectNeutral, 0},
		"after output":       {&execution.AfterOutputError{Err: overloaded()}, execution.EffectNeutral, 0},
		"all unavailable":    {&execution.UnavailableError{RetryAfter: 5 * time.Second}, execution.EffectNeutral, 5 * time.Second},
		"wrapped failure":    {errors.Join(errors.New("context"), overloaded()), execution.EffectFailure, 0},
	}
	for name, tc := range cases {
		got := execution.Observe(tc.err)
		if got.Effect != tc.effect || got.RetryAfter != tc.retryAfter || got.Class != "" {
			t.Errorf("%s: Observe()=%+v, want effect %v retry after %v", name, got, tc.effect, tc.retryAfter)
		}
	}
}

func TestHealthTrackerIsSafeForConcurrentUse(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := execution.NewHealthTracker(execution.HealthOptions{
		Policy: func(key string) execution.HealthPolicy {
			if key == "counted" {
				return execution.HealthPolicy{FailureThreshold: 1 << 30}
			}
			return execution.HealthPolicy{FailureThreshold: 3, OpenDuration: time.Second}
		},
		Now: clock.Now,
	})
	const workers, rounds = 16, 200
	var wg sync.WaitGroup
	for worker := range workers {
		wg.Go(func() {
			keys := []string{"a", "b", "c", "d"}
			for round := range rounds {
				key := keys[(worker+round)%len(keys)]
				switch round % 5 {
				case 0:
					health.Record(key, nil)
				case 1:
					health.Record(key, rateLimited(time.Millisecond))
				case 2:
					health.Available(key)
				case 3:
					_ = health.State(key)
				default:
					health.Record(key, overloaded())
				}
				health.Record("counted", overloaded())
				if round%50 == 0 {
					clock.Advance(time.Second)
				}
			}
		})
	}
	wg.Wait()
	if state := health.State("counted"); state.Streak != workers*rounds {
		t.Fatalf("streak=%d, want %d: a concurrent failure was lost", state.Streak, workers*rounds)
	}
}
