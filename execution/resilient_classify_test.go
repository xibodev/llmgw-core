package execution_test

import (
	"context"
	"fmt"
	"strconv"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
	"github.com/xibodev/llmgw-core/providers"
)

// The gateway's reading of core's own errors, after the port of its
// error_classification_test.go: only the transient statuses repeat, and
// every repeat counts against the circuit.
func TestTransientClassificationReadsStatusesAsTheGateway(t *testing.T) {
	t.Parallel()
	for _, status := range []int{400, 401, 403, 404, 408, 409, 429, 500, 501, 502, 503, 504, 505} {
		transient := status == 408 || status == 429 || status == 500 || (status >= 502 && status <= 504)
		for _, err := range []error{
			&providers.InvocationError{Msg: "fixture", Status: status, RetryAfter: 7 * time.Second},
			core.NewProviderOperationError("fixture", status, "", nil),
			&core.ProviderError{Classification: core.ProviderErrorClassification{StatusCode: status, CircuitFailure: status >= 500}},
		} {
			for _, candidate := range []error{err, fmt.Errorf("attempt: %w", err)} {
				own, got := core.ClassifyError(candidate), execution.TransientClassification(candidate)
				if got.StatusCode != status || got.Retryable != transient || got.CircuitFailure != transient ||
					got.FailoverEligible != own.FailoverEligible || got.RetryAfter != own.RetryAfter {
					t.Errorf("%d %T: got %+v, own %+v", status, err, got, own)
				}
			}
		}
	}
}

func TestTransientClassificationKeepsWhatNoStatusDecides(t *testing.T) {
	t.Parallel()
	for name, tc := range map[string]struct {
		err  error
		want core.ProviderErrorClassification
	}{
		"transport":       {disconnected(), core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}},
		"malformed":       {&providers.InvocationError{Msg: "fixture", CircuitFailure: true}, core.ProviderErrorClassification{CircuitFailure: true}},
		"statusless":      {&providers.InvocationError{Msg: "fixture"}, core.ProviderErrorClassification{}},
		"retryable only":  {&core.ProviderError{Classification: core.ProviderErrorClassification{Retryable: true}}, core.ProviderErrorClassification{Retryable: true, CircuitFailure: true}},
		"canceled answer": {core.NewProviderOperationError("decode", 200, "", context.DeadlineExceeded), core.ProviderErrorClassification{StatusCode: 200}},
		"decode failure":  {core.NewProviderOperationError("decode", 200, "", fmt.Errorf("invalid")), core.ProviderErrorClassification{StatusCode: 200, CircuitFailure: true}},
		"canceled":        {&core.ProviderOperationError{Failure: core.ProviderFailure{StatusCode: 503, Err: context.Canceled}}, core.ProviderErrorClassification{StatusCode: 503}},
		"configuration":   {core.NewConfigurationError("fixture", nil), core.ProviderErrorClassification{FailoverEligible: true}},
		"surface":         {unsupported(), core.ProviderErrorClassification{FailoverEligible: true}},
		"unclassified":    {fmt.Errorf("fixture"), core.ProviderErrorClassification{}},
		"nil":             {nil, core.ProviderErrorClassification{}},
	} {
		if got := execution.TransientClassification(tc.err); got != tc.want {
			t.Errorf("%s: got %+v want %+v", name, got, tc.want)
		}
	}
}

// Opting in changes what repeats and what the circuit counts; the default
// keeps core's reading, so a 501 repeats and a 429 is an answer.
func TestResilientOptsIntoTheGatewayReading(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		classify  func(error) core.ProviderErrorClassification
		status    int
		tries     int
		available bool
	}{
		{nil, 501, 3, false},
		{execution.TransientClassification, 501, 1, true},
		{nil, 429, 3, true},
		{execution.TransientClassification, 429, 3, false},
		{execution.TransientClassification, 400, 1, true},
	} {
		t.Run(strconv.Itoa(tc.status), func(t *testing.T) {
			health := gatewayCircuit(newClock(), 1)
			inner := &scripted{always: &providers.InvocationError{Msg: "fixture", Status: tc.status}}
			provider := execution.Resilient(inner, "provider", execution.Policy{
				Retry: execution.Retry{Attempts: 3}, Health: health, Classify: tc.classify,
			})
			_ = operate(provider, "chat")
			if available, _ := health.Available("provider"); inner.invokes != tc.tries || available != tc.available {
				t.Fatalf("tries=%d available=%v, want %d and %v", inner.invokes, available, tc.tries, tc.available)
			}
		})
	}
}
