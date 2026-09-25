package execution

import (
	"context"
	"net/http"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// guard runs one operation for r: it refuses it while the circuit is
// open, tries it as often as the policy allows, and records the outcome of
// its last try. discard releases what a failed try returned with its
// error, such as a stream.
func guard[T any](ctx context.Context, r *resilient, repeatable bool, try func(context.Context) (T, error), discard func(T)) (T, error) {
	var zero T
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	if err := r.admit(); err != nil {
		return zero, err
	}
	attempts := 1
	if repeatable {
		attempts = max(attempts, r.policy.Retry.Attempts)
	}
	for tries := 1; ; tries++ {
		value, err := try(ctx)
		if ctxErr := ctx.Err(); ctxErr != nil {
			// The caller gave up, which says nothing about the instance.
			if discard != nil {
				discard(value)
			}
			return zero, ctxErr
		}
		if err == nil {
			r.record(nil)
			return value, nil
		}
		if discard != nil {
			discard(value)
		}
		view := r.view(err)
		if tries < attempts && core.ClassifyError(view).Retryable {
			retry := Executor[string]{Retry: r.policy.Retry, Sleep: r.policy.Sleep}
			if waitErr := retry.wait(ctx, tries, view); waitErr != nil {
				return zero, waitErr
			}
			continue
		}
		r.record(view)
		return zero, err
	}
}

// admit refuses an operation while Health reports the instance
// unavailable.
func (r *resilient) admit() error {
	if r.policy.Health == nil {
		return nil
	}
	available, until := r.policy.Health.Available(r.key)
	if available {
		return nil
	}
	refusal := &CircuitOpenError{Key: r.key, Until: until}
	if !until.IsZero() {
		now := r.policy.Now
		if now == nil {
			now = time.Now
		}
		refusal.RetryAfter = max(0, until.Sub(now()))
	}
	return refusal
}

func (r *resilient) record(err error) {
	if r.policy.Health != nil {
		r.policy.Health.Record(r.key, err)
	}
}

func (r *resilient) repeatable(request core.Request) bool {
	if r.policy.Repeatable != nil {
		return r.policy.Repeatable(request)
	}
	return RepeatableRequest(request)
}

// view returns err as Classify reads it, so that Retry.Delay and Health,
// which read errors through core.ClassifyError, see the same reading.
func (r *resilient) view(err error) error {
	if r.policy.Classify == nil {
		return err
	}
	return &classified{err: err, classification: r.policy.Classify(err)}
}

// classified is an error read the policy's way. core.ClassifyError finds
// its classification first, and errors.Is and errors.As still reach the
// error itself.
type classified struct {
	err            error
	classification core.ProviderErrorClassification
}

func (c *classified) Error() string { return c.err.Error() }

func (c *classified) Unwrap() error { return c.err }

func (c *classified) ProviderErrorClassification() core.ProviderErrorClassification {
	return c.classification
}

// CircuitOpenError is Resilient's answer while Health reports an instance
// unavailable: its circuit is open, or a Retry-After cooldown runs.
// Nothing was sent. It is a 503, as in the gateway, which permits failover
// but no repeat, since a repeat would only meet the open circuit again.
//
// It says nothing about the instance, so record it in no Health: Observe
// would read its status as an answer.
type CircuitOpenError struct {
	// Key is the instance's health key.
	Key string
	// Until is when the instance becomes available, and RetryAfter is how
	// long after the error that is. Both are zero when Health gave no time.
	Until      time.Time
	RetryAfter time.Duration
}

// Error names the instance and how long it stays unavailable.
func (e *CircuitOpenError) Error() string {
	message := "execution: the circuit of " + e.Key + " is open"
	if e.RetryAfter > 0 {
		message += " for another " + e.RetryAfter.Truncate(100*time.Millisecond).String()
	}
	return message
}

// ProviderErrorClassification is a 503 that permits failover and carries
// RetryAfter.
func (e *CircuitOpenError) ProviderErrorClassification() core.ProviderErrorClassification {
	if e == nil {
		return core.ProviderErrorClassification{}
	}
	return core.ProviderErrorClassification{
		StatusCode: http.StatusServiceUnavailable, FailoverEligible: true, RetryAfter: e.RetryAfter,
	}
}

// countsTokens reports whether core.CountTokens finds a TokenCounter in
// provider or in a provider it unwraps to.
func countsTokens(provider core.Provider) bool {
	for provider != nil {
		if _, ok := provider.(core.TokenCounter); ok {
			return true
		}
		wrapper, ok := provider.(interface{ Unwrap() core.Provider })
		if !ok {
			return false
		}
		provider = wrapper.Unwrap()
	}
	return false
}
