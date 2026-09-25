package execution_test

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

func TestResilientDefinitiveFailureDoesNotOpenCircuit(t *testing.T) {
	t.Parallel()
	inner := &scripted{always: &invocation{status: 400}}
	provider := execution.Resilient(inner, "provider", execution.Policy{
		Retry: execution.Retry{Attempts: 3}, Health: gatewayCircuit(newClock(), 1),
	})
	for range 2 {
		_ = operate(provider, "chat")
	}
	if inner.invokes != 2 {
		t.Fatalf("invokes=%d want=2", inner.invokes)
	}
}

// A definitive failure ends the transient streak, on every guarded
// operation, token counts included.
func TestResilientDefinitiveFailureBreaksTransientStreak(t *testing.T) {
	t.Parallel()
	for _, kind := range []string{"chat", "messages", "token count"} {
		t.Run(kind, func(t *testing.T) {
			t.Parallel()
			inner := &scripted{outcomes: []error{&invocation{status: 503}, &invocation{status: 400}, &invocation{status: 503}}}
			provider := execution.Resilient(counter{inner}, "provider", execution.Policy{Health: gatewayCircuit(newClock(), 2)})
			var err error
			for range 4 {
				err = operate(provider, kind)
			}
			if tries := inner.invokes + inner.counts; err != nil || tries != 4 {
				t.Fatalf("err=%v tries=%d want=4", err, tries)
			}
		})
	}
}

func TestResilientMalformedResponseOpensCircuitWithoutRetry(t *testing.T) {
	t.Parallel()
	clock := newClock()
	inner := &scripted{always: &invocation{circuitFailure: true}}
	provider := execution.Resilient(inner, "provider", execution.Policy{
		Retry: execution.Retry{Attempts: 3}, Health: gatewayCircuit(clock, 1), Now: clock.Now,
	})
	_ = operate(provider, "chat")
	err := operate(provider, "chat")
	if inner.invokes != 1 {
		t.Fatalf("invokes=%d want=1", inner.invokes)
	}
	var open *execution.CircuitOpenError
	classification := core.ClassifyError(err)
	if !errors.As(err, &open) || open.Key != "provider" || !open.Until.Equal(clock.Now().Add(time.Minute)) ||
		classification.StatusCode != http.StatusServiceUnavailable || classification.Disposition() != core.DispositionFailover ||
		classification.RetryAfter != time.Minute || classification.CircuitFailure {
		t.Fatalf("err=%v classification=%+v", err, classification)
	}
	if err.Error() != "execution: the circuit of provider is open for another 1m0s" {
		t.Fatalf("message=%q", err.Error())
	}
}

// Once the cooldown passes the circuit is half-open: one failure reopens
// it and a success closes it. Catalog reads pass while it is open.
func TestResilientCircuitHalfOpensAndLeavesCatalogsAlone(t *testing.T) {
	t.Parallel()
	clock := newClock()
	inner := &scripted{outcomes: []error{&invocation{status: 502}, &invocation{status: 502}}}
	provider := execution.Resilient(inner, "provider", execution.Policy{Health: gatewayCircuit(clock, 1), Now: clock.Now})
	_ = operate(provider, "chat")
	if _, err := provider.ListModels(context.Background(), nil); err != nil || inner.lists != 1 {
		t.Fatalf("ListModels while open: err=%v lists=%d", err, inner.lists)
	}
	if err := operate(provider, "chat stream"); !isOpen(err) {
		t.Fatalf("stream while open: %v", err)
	}
	clock.Advance(time.Minute)
	if err := operate(provider, "chat"); err == nil || isOpen(err) {
		t.Fatalf("half-open try: %v", err)
	}
	if err := operate(provider, "chat"); !isOpen(err) {
		t.Fatalf("one failure did not reopen the circuit: %v", err)
	}
	clock.Advance(time.Minute)
	for range 2 {
		if err := operate(provider, "chat"); err != nil {
			t.Fatalf("a success did not close the circuit: %v", err)
		}
	}
	if invokes, streams := inner.tries(); invokes != 4 || streams != 0 {
		t.Fatalf("invokes=%d streams=%d", invokes, streams)
	}
}

func isOpen(err error) bool {
	var open *execution.CircuitOpenError
	return errors.As(err, &open)
}
