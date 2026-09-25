package execution_test

import (
	"context"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

// The port of TestRetryDelayHonorsLongerRetryAfter and
// TestRetryAfterIsTheDelayTheWrapperHonours: the wait is the longer of the
// backoff and the upstream's Retry-After.
func TestExponentialBackoffHonoursLongerRetryAfter(t *testing.T) {
	t.Parallel()
	delay := execution.ExponentialBackoff(10*time.Millisecond, 2, time.Second)
	for _, tc := range []struct {
		failed int
		err    error
		want   time.Duration
	}{
		{1, &invocation{status: 429, retryAfter: 2 * time.Second}, 2 * time.Second},
		{1, &invocation{status: 429}, 10 * time.Millisecond},
		{2, &invocation{status: 503}, 20 * time.Millisecond},
		{10, &invocation{status: 503}, time.Second},
		{3, &invocation{status: 503, retryAfter: 3 * time.Second}, 3 * time.Second},
		{0, nil, 10 * time.Millisecond},
	} {
		if got := delay(tc.failed, tc.err); got != tc.want {
			t.Errorf("delay(%d, %v)=%v want %v", tc.failed, tc.err, got, tc.want)
		}
	}
	if got := execution.ExponentialBackoff(time.Second, -1, time.Minute)(2, nil); got != 0 {
		t.Errorf("a negative backoff waits %v", got)
	}
}

// Repeats wait out the backoff through Sleep, and an exhausted operation
// counts as one failure.
func TestResilientWaitsBetweenRepeats(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := gatewayCircuit(clock, 2)
	inner := &scripted{outcomes: []error{&invocation{status: 429, retryAfter: 5 * time.Second}, &invocation{status: 503}, &invocation{status: 503}}}
	provider := execution.Resilient(inner, "provider", execution.Policy{
		Retry:  execution.Retry{Attempts: 3, Delay: execution.ExponentialBackoff(time.Second, 2, 10*time.Second)},
		Health: health, Now: clock.Now, Sleep: clock.Sleep,
	})
	started := clock.Now()
	if err := operate(provider, "chat"); err == nil || inner.invokes != 3 {
		t.Fatalf("err=%v invokes=%d", err, inner.invokes)
	}
	if waited := clock.Now().Sub(started); waited != 7*time.Second {
		t.Fatalf("waited %v, want max(1s, 5s) then 2s", waited)
	}
	if state := health.State("provider"); state.Streak != 1 {
		t.Fatalf("an exhausted operation counted %d failures", state.Streak)
	}
}

// A stream repeats only its opening. A failed open's stream is closed,
// and a failure after the stream opened is the caller's.
func TestResilientStreamRetriesOnlyAtOpen(t *testing.T) {
	t.Parallel()
	health := gatewayCircuit(newClock(), 1)
	inner := &scripted{outcomes: []error{&invocation{status: 503}}}
	provider := execution.Resilient(inner, "provider", execution.Policy{Retry: execution.Retry{Attempts: 3}, Health: health})
	opened, err := provider.Stream(context.Background(), chatRequest)
	if err != nil || inner.streams != 2 || inner.opened[0].closed() != 1 {
		t.Fatalf("err=%v streams=%d", err, inner.streams)
	}
	inner.opened[1].frames, inner.opened[1].err = []string{keepalive}, &invocation{status: 502}
	if _, err := drain(t, opened); err == nil || inner.streams != 2 {
		t.Fatalf("a failure after the stream opened: err=%v streams=%d", err, inner.streams)
	}
	assertAvailable(t, health, "provider")
}

func TestResilientRecordsNothingForACallerThatGaveUp(t *testing.T) {
	t.Parallel()
	health := gatewayCircuit(newClock(), 1)
	ctx, cancel := context.WithCancel(context.Background())
	inner := &scripted{always: &invocation{status: 503}}
	provider := execution.Resilient(canceler{inner, cancel}, "provider", execution.Policy{Retry: execution.Retry{Attempts: 3}, Health: health})
	if _, err := provider.Invoke(ctx, chatRequest); err != context.Canceled || inner.invokes != 1 {
		t.Fatalf("err=%v invokes=%d", err, inner.invokes)
	}
	if _, err := provider.Invoke(ctx, chatRequest); err != context.Canceled || inner.invokes != 1 {
		t.Fatalf("a canceled caller reached the provider: err=%v", err)
	}
	assertAvailable(t, health, "provider")
}

// canceler cancels its caller's context during the call, as a client that
// disconnects does.
type canceler struct {
	*scripted
	cancel context.CancelFunc
}

func (c canceler) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	c.cancel()
	return c.scripted.Invoke(ctx, request)
}
