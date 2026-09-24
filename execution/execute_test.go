package execution_test

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

// calls scripts run's outcome per candidate and records the order of calls.
// Each candidate's outcomes are consumed in turn; the last one repeats.
type calls struct {
	mu       sync.Mutex
	order    []string
	outcomes map[string][]error
}

func (c *calls) run(_ context.Context, candidate string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.order = append(c.order, candidate)
	outcomes := c.outcomes[candidate]
	if len(outcomes) == 0 {
		return "served by " + candidate, nil
	}
	err := outcomes[0]
	if len(outcomes) > 1 {
		c.outcomes[candidate] = outcomes[1:]
	}
	if err != nil {
		return "discarded", err
	}
	return "served by " + candidate, nil
}

func (c *calls) made() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.order...)
}

func identity(candidate string) string { return candidate }

func TestExecuteServesTheFirstCandidateThatSucceeds(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := tracker(clock, execution.HealthPolicy{FailureThreshold: 5, OpenDuration: time.Minute})
	failure := overloaded()
	script := &calls{outcomes: map[string][]error{"a": {failure}}}
	executor := execution.Executor[string]{Health: health, Key: identity, Now: clock.Now}

	result, err := execution.Execute(context.Background(), executor, []string{"a", "b", "c"}, script.run)
	if err != nil || result.Value != "served by b" || result.Candidate != "b" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if got := script.made(); !slices.Equal(got, []string{"a", "b"}) {
		t.Fatalf("calls=%v, want a then b", got)
	}
	if len(result.Attempts) != 2 {
		t.Fatalf("trace=%+v", result.Attempts)
	}
	first, second := result.Attempts[0], result.Attempts[1]
	if first.Candidate != "a" || first.Key != "a" || first.Tries != 1 || first.Err != failure ||
		first.Disposition != core.DispositionRetryable || first.Class != core.ProviderErrorUpstream ||
		first.Classification.StatusCode != 503 || first.Unavailable {
		t.Fatalf("failed entry=%+v", first)
	}
	if second.Candidate != "b" || second.Tries != 1 || second.Err != nil || second.Disposition != "" || second.Class != "" {
		t.Fatalf("served entry=%+v", second)
	}
	if state := health.State("a"); state.Streak != 1 {
		t.Fatalf("a's failure was not recorded: %+v", state)
	}
}

func TestExecuteSkipsUnavailableCandidates(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := tracker(clock, execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute})
	health.Record("a", overloaded())
	script := &calls{}
	executor := execution.Executor[string]{Health: health, Key: identity}

	result, err := execution.Execute(context.Background(), executor, []string{"a", "b"}, script.run)
	if err != nil || result.Candidate != "b" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if got := script.made(); !slices.Equal(got, []string{"b"}) {
		t.Fatalf("an unavailable candidate was tried: %v", got)
	}
	skipped := result.Attempts[0]
	if !skipped.Unavailable || !skipped.Until.Equal(clock.Now().Add(time.Minute)) || skipped.Tries != 0 || skipped.Err != nil {
		t.Fatalf("skipped entry=%+v", skipped)
	}
}

func TestExecuteStopsAtATerminalDisposition(t *testing.T) {
	t.Parallel()
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 3, OpenDuration: time.Minute})
	health.Record("a", overloaded())
	terminal := rejected(400)
	script := &calls{outcomes: map[string][]error{"a": {terminal}}}
	executor := execution.Executor[string]{Health: health, Key: identity}

	result, err := execution.Execute(context.Background(), executor, []string{"a", "b"}, script.run)
	if err != terminal || result.Value != "" || result.Candidate != "" {
		t.Fatalf("result=%+v err=%v, want a's own error", result, err)
	}
	if got := script.made(); !slices.Equal(got, []string{"a"}) {
		t.Fatalf("calls=%v: a terminal error must stop the execution", got)
	}
	if len(result.Attempts) != 1 || result.Attempts[0].Disposition != core.DispositionTerminal {
		t.Fatalf("trace=%+v", result.Attempts)
	}
	// The default Observe reads a definitive rejection as an answer.
	if state := health.State("a"); state.Streak != 0 {
		t.Fatalf("a definitive rejection did not end the streak: %+v", state)
	}

	unclassified := errors.New("unclassified")
	script = &calls{outcomes: map[string][]error{"a": {unclassified}}}
	if _, err := execution.Execute(context.Background(), executor, []string{"a", "b"}, script.run); err != unclassified || len(script.made()) != 1 {
		t.Fatalf("an unclassified error must stop the execution: err=%v calls=%v", err, script.made())
	}
}

func TestExecuteMovesOnAfterFailoverAndRetryableDispositions(t *testing.T) {
	t.Parallel()
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 5, OpenDuration: time.Minute})
	script := &calls{outcomes: map[string][]error{"a": {unsupported()}, "b": {overloaded()}}}
	executor := execution.Executor[string]{Health: health, Key: identity}

	result, err := execution.Execute(context.Background(), executor, []string{"a", "b", "c"}, script.run)
	if err != nil || result.Candidate != "c" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	dispositions := []core.Disposition{}
	for _, attempt := range result.Attempts {
		dispositions = append(dispositions, attempt.Disposition)
	}
	if !slices.Equal(dispositions, []core.Disposition{core.DispositionFailover, core.DispositionRetryable, ""}) {
		t.Fatalf("dispositions=%v", dispositions)
	}
	if health.State("a").Streak != 0 || health.State("b").Streak != 1 {
		t.Fatalf("a surface error says nothing about health, an overload does: a=%+v b=%+v", health.State("a"), health.State("b"))
	}
}

func TestExecuteReturnsTheLastTriedCandidatesError(t *testing.T) {
	t.Parallel()
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute})
	health.Record("c", overloaded())
	first, second := overloaded(), unsupported()
	script := &calls{outcomes: map[string][]error{"a": {first}, "b": {second}}}
	executor := execution.Executor[string]{Health: health, Key: identity}

	result, err := execution.Execute(context.Background(), executor, []string{"a", "b", "c"}, script.run)
	if err != second {
		t.Fatalf("err=%v, want the last tried candidate's", err)
	}
	if len(result.Attempts) != 3 || !result.Attempts[2].Unavailable {
		t.Fatalf("trace=%+v", result.Attempts)
	}
}

func TestExecuteReportsEveryCandidateUnavailable(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := tracker(clock, execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute})
	health.Record("late", overloaded())
	clock.Advance(20 * time.Second)
	health.Record("soon", rateLimited(10*time.Second))
	script := &calls{}
	executor := execution.Executor[string]{Health: health, Key: identity, Now: clock.Now}

	result, err := execution.Execute(context.Background(), executor, []string{"late", "soon"}, script.run)
	var unavailable *execution.UnavailableError
	if !errors.As(err, &unavailable) || len(script.made()) != 0 {
		t.Fatalf("err=%v calls=%v", err, script.made())
	}
	if !unavailable.Until.Equal(clock.Now().Add(10*time.Second)) || unavailable.RetryAfter != 10*time.Second || unavailable.Error() == "" {
		t.Fatalf("unavailable=%+v, want the earliest candidate", unavailable)
	}
	if classification := core.ClassifyError(err); classification.Disposition() != core.DispositionFailover || classification.RetryAfter != 10*time.Second {
		t.Fatalf("classification=%+v: an enclosing execution may move on after the wait", classification)
	}
	if len(result.Attempts) != 2 || !result.Attempts[0].Unavailable || !result.Attempts[1].Unavailable {
		t.Fatalf("trace=%+v", result.Attempts)
	}
	var missing *execution.UnavailableError
	if missing.ProviderErrorClassification() != (core.ProviderErrorClassification{}) {
		t.Fatal("a nil UnavailableError is not safe to classify")
	}
}

// timeless is a Health that knows keys are down but not until when.
type timeless struct{}

func (timeless) Available(string) (bool, time.Time) { return false, time.Time{} }
func (timeless) Record(string, error)               {}

func TestExecuteReportsUnavailabilityWithoutATime(t *testing.T) {
	t.Parallel()
	executor := execution.Executor[string]{Health: timeless{}, Key: identity}
	_, err := execution.Execute(context.Background(), executor, []string{"a"}, (&calls{}).run)
	var unavailable *execution.UnavailableError
	if !errors.As(err, &unavailable) || !unavailable.Until.IsZero() || unavailable.RetryAfter != 0 {
		t.Fatalf("err=%v", err)
	}
}

func TestExecuteNeedsCandidatesAndAKeyForHealth(t *testing.T) {
	t.Parallel()
	script := &calls{}
	if _, err := execution.Execute(context.Background(), execution.Executor[string]{}, nil, script.run); !errors.Is(err, execution.ErrNoCandidates) {
		t.Fatalf("err=%v, want ErrNoCandidates", err)
	}
	executor := execution.Executor[string]{Health: execution.NewHealthTracker(execution.HealthOptions{})}
	if _, err := execution.Execute(context.Background(), executor, []string{"a"}, script.run); err == nil || len(script.made()) != 0 {
		t.Fatalf("an executor with Health but no Key ran: err=%v calls=%v", err, script.made())
	}
	result, err := execution.Execute(context.Background(), execution.Executor[string]{}, []string{"a"}, script.run)
	if err != nil || result.Candidate != "a" || result.Attempts[0].Key != "" {
		t.Fatalf("the zero Executor must try candidates without health: result=%+v err=%v", result, err)
	}
}

func TestExecuteRepeatsARetryableCandidateWhenAsked(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := tracker(clock, execution.HealthPolicy{FailureThreshold: 2, OpenDuration: time.Minute})
	type delay struct {
		failed     int
		retryAfter time.Duration
	}
	var delays []delay
	executor := execution.Executor[string]{
		Health: health, Key: identity, Now: clock.Now, Sleep: clock.Sleep,
		Retry: execution.Retry{Attempts: 3, Delay: func(failed int, err error) time.Duration {
			retryAfter := core.ClassifyError(err).RetryAfter
			delays = append(delays, delay{failed, retryAfter})
			return max(time.Duration(failed)*time.Second, retryAfter)
		}},
	}

	script := &calls{outcomes: map[string][]error{"a": {rateLimited(5 * time.Second), overloaded(), nil}}}
	result, err := execution.Execute(context.Background(), executor, []string{"a", "b"}, script.run)
	if err != nil || result.Candidate != "a" || result.Attempts[0].Tries != 3 || result.Attempts[0].Err != nil {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if !slices.Equal(delays, []delay{{1, 5 * time.Second}, {2, 0}}) {
		t.Fatalf("delays=%+v", delays)
	}
	if result.Attempts[0].Duration != 7*time.Second {
		t.Fatalf("duration=%v, want the 5s and 2s waits", result.Attempts[0].Duration)
	}
	if state := health.State("a"); state.Streak != 0 {
		t.Fatalf("a repeat that served still counted its failed tries: %+v", state)
	}

	// An exhausted candidate is recorded once, then the next one is tried.
	script = &calls{outcomes: map[string][]error{"a": {overloaded()}}}
	result, err = execution.Execute(context.Background(), executor, []string{"a", "b"}, script.run)
	if err != nil || result.Candidate != "b" || result.Attempts[0].Tries != 3 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if !slices.Equal(script.made(), []string{"a", "a", "a", "b"}) {
		t.Fatalf("calls=%v", script.made())
	}
	if state := health.State("a"); state.Streak != 1 {
		t.Fatalf("an exhausted candidate must count once: %+v", state)
	}
}

func TestExecuteNeverRepeatsAFailoverDisposition(t *testing.T) {
	t.Parallel()
	script := &calls{outcomes: map[string][]error{"a": {unsupported()}}}
	executor := execution.Executor[string]{Retry: execution.Retry{Attempts: 3}}
	result, err := execution.Execute(context.Background(), executor, []string{"a", "b"}, script.run)
	if err != nil || !slices.Equal(script.made(), []string{"a", "b"}) || result.Attempts[0].Tries != 1 {
		t.Fatalf("calls=%v trace=%+v err=%v", script.made(), result.Attempts, err)
	}
}

func TestExecuteRepeatsAtOnceWithoutADelayAndWaitsOnARealTimer(t *testing.T) {
	t.Parallel()
	script := &calls{outcomes: map[string][]error{"a": {overloaded(), nil}}}
	executor := execution.Executor[string]{Retry: execution.Retry{Attempts: 2}}
	if result, err := execution.Execute(context.Background(), executor, []string{"a"}, script.run); err != nil || result.Attempts[0].Tries != 2 {
		t.Fatalf("result=%+v err=%v", result, err)
	}

	script = &calls{outcomes: map[string][]error{"a": {overloaded(), nil}}}
	executor.Retry.Delay = func(int, error) time.Duration { return time.Millisecond }
	if result, err := execution.Execute(context.Background(), executor, []string{"a"}, script.run); err != nil || result.Attempts[0].Duration < time.Millisecond {
		t.Fatalf("result=%+v err=%v", result, err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	script = &calls{outcomes: map[string][]error{"a": {overloaded()}}}
	executor.Retry.Delay = func(int, error) time.Duration {
		cancel()
		return time.Hour
	}
	if _, err := execution.Execute(ctx, executor, []string{"a", "b"}, script.run); !errors.Is(err, context.Canceled) || len(script.made()) != 1 {
		t.Fatalf("a canceled wait did not stop the execution: err=%v calls=%v", err, script.made())
	}
}

func TestExecuteStopsWhenTheCallerGivesUp(t *testing.T) {
	t.Parallel()
	policy := execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute}

	t.Run("before the first candidate", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		script := &calls{}
		result, err := execution.Execute(ctx, execution.Executor[string]{}, []string{"a"}, script.run)
		if !errors.Is(err, context.Canceled) || len(script.made()) != 0 || len(result.Attempts) != 0 {
			t.Fatalf("result=%+v err=%v calls=%v", result, err, script.made())
		}
	})

	// A failure that looks like instability records nothing once the caller
	// has left: the context's end, not the candidate, explains it.
	t.Run("canceled during a try", func(t *testing.T) {
		t.Parallel()
		health := tracker(newClock(), policy)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		cause := overloaded()
		run := func(context.Context, string) (string, error) {
			cancel()
			return "", cause
		}
		result, err := execution.Execute(ctx, execution.Executor[string]{Health: health, Key: identity}, []string{"a", "b"}, run)
		if !errors.Is(err, context.Canceled) || len(result.Attempts) != 1 || result.Attempts[0].Err != cause {
			t.Fatalf("err=%v trace=%+v", err, result.Attempts)
		}
		assertAvailable(t, health, "a")
	})

	t.Run("deadline during a try", func(t *testing.T) {
		t.Parallel()
		health := tracker(newClock(), policy)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		defer cancel()
		run := func(ctx context.Context, _ string) (string, error) {
			<-ctx.Done()
			return "", overloaded()
		}
		result, err := execution.Execute(ctx, execution.Executor[string]{Health: health, Key: identity}, []string{"a", "b"}, run)
		if !errors.Is(err, context.DeadlineExceeded) || len(result.Attempts) != 1 {
			t.Fatalf("err=%v trace=%+v", err, result.Attempts)
		}
		assertAvailable(t, health, "a")
	})

	t.Run("during a repeat's wait", func(t *testing.T) {
		t.Parallel()
		clock := newClock()
		health := tracker(clock, policy)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		script := &calls{outcomes: map[string][]error{"a": {overloaded()}}}
		executor := execution.Executor[string]{
			Health: health, Key: identity,
			Retry: execution.Retry{Attempts: 3, Delay: func(int, error) time.Duration { return time.Second }},
			Sleep: func(ctx context.Context, d time.Duration) error {
				cancel()
				return ctx.Err()
			},
		}
		_, err := execution.Execute(ctx, executor, []string{"a", "b"}, script.run)
		if !errors.Is(err, context.Canceled) || !slices.Equal(script.made(), []string{"a"}) {
			t.Fatalf("err=%v calls=%v", err, script.made())
		}
		assertAvailable(t, health, "a")
	})
}

func TestExecuteTraceNamesErrorClasses(t *testing.T) {
	t.Parallel()
	cases := map[string]struct {
		err   error
		class core.ProviderErrorClass
	}{
		"its own class":     {core.NewConfigurationError("not configured", nil), core.ProviderErrorConfiguration},
		"a status":          {core.NewProviderOperationError("chat", 429, "3", nil), core.ProviderErrorRateLimited},
		"an auth status":    {&core.ProviderError{Classification: core.ProviderErrorClassification{StatusCode: 401}}, core.ProviderErrorAuth},
		"a transport error": {core.NewProviderOperationError("chat", 0, "", errors.New("connection reset")), core.ProviderErrorTransport},
		"no class":          {unsupported(), ""},
	}
	for name, tc := range cases {
		run := func(context.Context, string) (string, error) { return "", tc.err }
		result, _ := execution.Execute(context.Background(), execution.Executor[string]{}, []string{"a"}, run)
		if got := result.Attempts[0].Class; got != tc.class {
			t.Errorf("%s: class=%q, want %q", name, got, tc.class)
		}
	}
}

func TestExecuteIsSafeForConcurrentUse(t *testing.T) {
	t.Parallel()
	health := execution.NewHealthTracker(execution.HealthOptions{
		Policy: func(string) execution.HealthPolicy {
			return execution.HealthPolicy{FailureThreshold: 2, OpenDuration: time.Millisecond}
		},
	})
	executor := execution.Executor[string]{Health: health, Key: identity, Retry: execution.Retry{Attempts: 2}}
	var wg sync.WaitGroup
	for worker := range 16 {
		wg.Go(func() {
			for round := range 100 {
				run := func(_ context.Context, candidate string) (string, error) {
					if (worker+round)%3 == 0 && candidate == "a" {
						return "", overloaded()
					}
					return candidate, nil
				}
				result, err := execution.Execute(context.Background(), executor, []string{"a", "b"}, run)
				if err != nil || (result.Candidate != "a" && result.Candidate != "b") {
					t.Errorf("result=%+v err=%v", result, err)
					return
				}
			}
		})
	}
	wg.Wait()
}
