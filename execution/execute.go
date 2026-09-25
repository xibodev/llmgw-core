package execution

import (
	"context"
	"errors"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// ErrNoCandidates is the error of an execution given no candidates.
var ErrNoCandidates = errors.New("execution: no candidates")

var errNoKey = errors.New("execution: an Executor with Health needs a Key")

// Executor tries candidates in order. The zero value tries each candidate
// once and tracks no health.
type Executor[C any] struct {
	// Health skips the candidates it reports unavailable and records the
	// outcome of every candidate tried. Nil tracks nothing.
	Health Health
	// Key returns a candidate's health key: the instance it runs on. It is
	// required with Health.
	Key func(C) string
	// Retry repeats a candidate that fails with a retryable disposition
	// before moving on. The zero value never repeats one.
	Retry Retry
	// Now returns the current time, for durations in the trace and the
	// RetryAfter of an UnavailableError, so it should agree with Health's
	// clock. Nil uses time.Now.
	Now func() time.Time
	// Sleep waits d before a repeat, or returns ctx's error once ctx ends.
	// Nil waits on a timer.
	Sleep func(ctx context.Context, d time.Duration) error
}

// Retry repeats a candidate whose try fails with core.DispositionRetryable.
// A failover disposition is never repeated: it permits another candidate
// only.
type Retry struct {
	// Attempts is how many times one candidate may be tried, the first try
	// included. Below two, a candidate is tried once.
	Attempts int
	// Delay returns how long to wait before trying again, given how many
	// tries have failed and the last one's error. A product that honours
	// Retry-After waits at least core.ClassifyError(err).RetryAfter. Nil
	// tries again at once.
	Delay func(failed int, err error) time.Duration
}

// Attempt is one candidate's entry in an execution's trace.
type Attempt[C any] struct {
	// Candidate is the candidate the entry describes.
	Candidate C
	// Key is the candidate's health key, when the execution tracks health.
	Key string
	// Unavailable reports a candidate skipped without a try because Health
	// reported it unavailable until Until.
	Unavailable bool
	Until       time.Time
	// Tries counts the tries, repeats included.
	Tries int
	// Err is the last try's error: nil when the candidate served or was
	// skipped.
	Err error
	// Classification is Err's routing metadata, Class names its kind, and
	// Disposition is what it permitted. A terminal disposition ended the
	// execution.
	Classification core.ProviderErrorClassification
	Class          core.ProviderErrorClass
	Disposition    core.Disposition
	// Duration spans the tries and the waits between them.
	Duration time.Duration
}

// Result is an execution's outcome.
type Result[C, R any] struct {
	// Value is what the serving candidate returned, and Candidate is that
	// candidate. Both are zero when no candidate served.
	Value     R
	Candidate C
	// Attempts is the trace: an entry for each candidate reached, in order.
	Attempts []Attempt[C]
}

// UnavailableError reports that Health found every candidate unavailable,
// so none was tried.
type UnavailableError struct {
	// Until is the earliest time a candidate becomes available, and
	// RetryAfter is how long after the error that is. Both are zero when
	// Health gave no time.
	Until      time.Time
	RetryAfter time.Duration
}

// Error says that no candidate was tried; the trace says why.
func (e *UnavailableError) Error() string {
	return "execution: every candidate is unavailable"
}

// ProviderErrorClassification permits failover, so an enclosing execution
// can move on, and carries RetryAfter.
func (e *UnavailableError) ProviderErrorClassification() core.ProviderErrorClassification {
	if e == nil {
		return core.ProviderErrorClassification{}
	}
	return core.ProviderErrorClassification{FailoverEligible: true, RetryAfter: e.RetryAfter}
}

// Execute tries candidates in order until one serves, calling run for each.
//
// A candidate Health reports unavailable is skipped. When run fails, the
// error's disposition decides what follows: a terminal one ends the
// execution with that error, while failover and retryable ones move on to
// the next candidate, after repeating a retryable one as Retry allows.
// Health records the outcome of each candidate's last try. Once ctx ends,
// Execute stops with ctx's error and records nothing more. A value run
// returns with an error is discarded, so run releases whatever it opened
// when it fails.
//
// The error is nil when a candidate served. Otherwise it is ctx's error,
// the error of the last candidate tried, an *UnavailableError when every
// candidate was skipped, or ErrNoCandidates. The trace is complete either
// way.
func Execute[C, R any](ctx context.Context, executor Executor[C], candidates []C, run func(context.Context, C) (R, error)) (Result[C, R], error) {
	var result Result[C, R]
	if executor.Health != nil && executor.Key == nil {
		return result, errNoKey
	}
	if len(candidates) == 0 {
		return result, ErrNoCandidates
	}
	now := executor.Now
	if now == nil {
		now = time.Now
	}
	var last error
	var until time.Time
	for _, candidate := range candidates {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		entry := Attempt[C]{Candidate: candidate}
		if executor.Health != nil {
			entry.Key = executor.Key(candidate)
			if available, next := executor.Health.Available(entry.Key); !available {
				entry.Unavailable, entry.Until = true, next
				result.Attempts = append(result.Attempts, entry)
				if !next.IsZero() && (until.IsZero() || next.Before(until)) {
					until = next
				}
				continue
			}
		}
		value, stop, err := try(ctx, executor, now, &entry, run)
		result.Attempts = append(result.Attempts, entry)
		if err == nil {
			result.Value, result.Candidate = value, candidate
			return result, nil
		}
		if stop {
			return result, err
		}
		last = err
	}
	// A candidate that was tried explains the failure better than one that
	// was skipped.
	if last != nil {
		return result, last
	}
	unavailable := &UnavailableError{Until: until}
	if !until.IsZero() {
		unavailable.RetryAfter = max(0, until.Sub(now()))
	}
	return result, unavailable
}

// try runs one candidate, repeating it as Retry allows, and records the
// outcome of its last try. stop reports that the execution ends with err.
func try[C, R any](ctx context.Context, executor Executor[C], now func() time.Time, entry *Attempt[C], run func(context.Context, C) (R, error)) (value R, stop bool, err error) {
	started := now()
	defer func() { entry.Duration = now().Sub(started) }()
	for {
		entry.Tries++
		value, err = run(ctx, entry.Candidate)
		entry.describe(err)
		if err == nil {
			executor.record(entry.Key, nil)
			return value, false, nil
		}
		var zero R
		if ctxErr := ctx.Err(); ctxErr != nil {
			// The caller gave up, which says nothing about the candidate.
			return zero, true, ctxErr
		}
		if entry.Disposition == core.DispositionRetryable && entry.Tries < executor.Retry.Attempts {
			if waitErr := executor.wait(ctx, entry.Tries, err); waitErr != nil {
				return zero, true, waitErr
			}
			continue
		}
		executor.record(entry.Key, err)
		return zero, entry.Disposition == core.DispositionTerminal, err
	}
}

func (e Executor[C]) record(key string, err error) {
	if e.Health != nil {
		e.Health.Record(key, err)
	}
}

func (e Executor[C]) wait(ctx context.Context, failed int, err error) error {
	var delay time.Duration
	if e.Retry.Delay != nil {
		delay = e.Retry.Delay(failed, err)
	}
	if delay <= 0 {
		return ctx.Err()
	}
	if e.Sleep != nil {
		return e.Sleep(ctx, delay)
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// describe sets the entry's view of err, the error of its latest try.
func (a *Attempt[C]) describe(err error) {
	a.Err = err
	a.Classification, a.Class, a.Disposition = core.ProviderErrorClassification{}, "", ""
	if err == nil {
		return
	}
	a.Classification = core.ClassifyError(err)
	a.Class = errorClass(err, a.Classification)
	a.Disposition = a.Classification.Disposition()
}

// errorClass names the kind of err: the class it gives itself, else the one
// its status or a transport failure implies, as health evidence names them.
func errorClass(err error, classification core.ProviderErrorClassification) core.ProviderErrorClass {
	var providerErr *core.ProviderError
	if errors.As(err, &providerErr) && providerErr.Class != "" {
		return providerErr.Class
	}
	switch {
	case classification.StatusCode != 0:
		return core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: classification.StatusCode, Err: err}).ErrorClass
	case classification.CircuitFailure:
		return core.ProviderErrorTransport
	}
	return ""
}
