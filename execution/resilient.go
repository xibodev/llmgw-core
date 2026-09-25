package execution

import (
	"context"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// Policy is how Resilient serves one instance: how often it tries an
// operation, and the circuit that guards the instance. It holds what the
// gateway's per-provider retry and circuit-breaker policy holds. The zero
// value tries every operation once and guards nothing.
type Policy struct {
	// Retry repeats a try whose error Classify reads as retryable, up to
	// Retry.Attempts tries of one operation, waiting Retry.Delay between
	// them. ExponentialBackoff is the gateway's Delay.
	Retry Retry
	// Health is the instance's circuit, under the key Resilient is given.
	// While it reports the key unavailable, every operation is refused
	// with a *CircuitOpenError and nothing is sent. It records the outcome
	// of each operation's last try, so an operation whose repeats all fail
	// counts once. Nil guards nothing.
	Health Health
	// Classify reads a try's error: whether it may repeat, whether it
	// counts against the circuit, and how long the upstream asked callers
	// to wait. Retry.Delay and Health read the error as Classify does. Nil
	// uses core.ClassifyError.
	Classify func(error) core.ProviderErrorClassification
	// Repeatable reports whether a request may be sent more than once; one
	// it refuses is tried once, whatever Retry says. Nil uses
	// RepeatableRequest.
	Repeatable func(core.Request) bool
	// Now returns the current time, for a CircuitOpenError's RetryAfter,
	// so it should agree with Health's clock. Nil uses time.Now.
	Now func() time.Time
	// Sleep waits d before a repeat, or returns ctx's error once ctx ends.
	// Nil waits on a timer.
	Sleep func(ctx context.Context, d time.Duration) error
}

// Resilient returns provider guarded by policy, under key: each operation
// is refused while the circuit is open, repeated while its failure permits
// a repeat, and its outcome recorded.
//
// Invoke and CountTokens repeat the operation. Stream repeats only opening
// the stream: once a stream is open it is the caller's, and its later
// failures are neither repeated nor recorded. A caller that gives up gets
// ctx's error and nothing is recorded. Otherwise the last try's error is
// returned as the provider returned it.
//
// ListModels and NativeSurfaces pass through, as in the gateway: a catalog
// read neither repeats nor counts against the circuit, so an open circuit
// never keeps a catalog from being rediscovered. The wrapper unwraps to
// provider, so core.PreservesWire reads provider's declarations, and it
// counts tokens only when provider can.
//
// A provider that keys an upstream invocation on its context, as Zen does,
// needs that identity in ctx before the call, so repeats reuse it.
func Resilient(provider core.Provider, key string, policy Policy) core.Provider {
	if provider == nil {
		return nil
	}
	wrapper := &resilient{provider: provider, key: key, policy: policy}
	if countsTokens(provider) {
		return &resilientCounter{wrapper}
	}
	return wrapper
}

// resilient is the wrapper Resilient returns for a provider that cannot
// count tokens.
type resilient struct {
	provider core.Provider
	key      string
	policy   Policy
}

func (r *resilient) NativeSurfaces(model string) []core.ModelSurface {
	return r.provider.NativeSurfaces(model)
}

func (r *resilient) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	return guard(ctx, r, r.repeatable(request), func(ctx context.Context) (core.Response, error) {
		return r.provider.Invoke(ctx, request)
	}, nil)
}

func (r *resilient) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	return guard(ctx, r, r.repeatable(request), func(ctx context.Context) (core.StreamIter, error) {
		return r.provider.Stream(ctx, request)
	}, closeStream)
}

func (r *resilient) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	return r.provider.ListModels(ctx, credential)
}

// Unwrap returns the wrapped provider, which receives every request
// unchanged.
func (r *resilient) Unwrap() core.Provider { return r.provider }

// resilientCounter is the wrapper of a provider that counts tokens: a
// count is guarded like any operation. Counting changes nothing upstream,
// so it always may repeat.
type resilientCounter struct{ *resilient }

func (r *resilientCounter) CountTokens(ctx context.Context, request core.TokenCountRequest) (core.TokenCount, error) {
	return guard(ctx, r.resilient, true, func(ctx context.Context) (core.TokenCount, error) {
		return core.CountTokens(ctx, r.provider, request)
	}, nil)
}

func closeStream(stream core.StreamIter) {
	if stream != nil {
		_ = stream.Close()
	}
}
