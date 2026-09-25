package execution_test

import (
	"context"
	"strconv"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

// gatewayCircuit is the gateway's breaker as a Runtime composes it: the
// default Observe, a streak that never expires, and Retry-After that only
// lengthens the wait before a repeat.
func gatewayCircuit(clock *clock, threshold int) *execution.HealthTracker {
	return tracker(clock, execution.HealthPolicy{FailureThreshold: threshold, OpenDuration: time.Minute, IgnoreRetryAfter: true})
}

// operate performs one operation of the named kind and returns its error.
func operate(provider core.Provider, kind string) error {
	ctx := context.Background()
	var err error
	switch kind {
	case "chat":
		_, err = provider.Invoke(ctx, chatRequest)
	case "chat stream":
		_, err = provider.Stream(ctx, chatRequest)
	case "responses":
		_, err = provider.Invoke(ctx, responsesRequest)
	case "responses stream":
		_, err = provider.Stream(ctx, responsesRequest)
	case "messages":
		_, err = provider.Invoke(ctx, messagesRequest)
	case "token count":
		_, err = core.CountTokens(ctx, provider, core.TokenCountRequest{Request: messagesRequest})
	}
	return err
}

// The ports of the gateway's policy_v043_test.go and its native Messages
// tests. Errors classify themselves as the gateway's do.

func TestResilientRetriesOnlyTransientInvocationFailures(t *testing.T) {
	t.Parallel()
	for _, status := range []int{0, 408, 429, 500, 502, 503, 504, 400, 401, 403, 404} {
		failure := error(&invocation{status: status})
		if status == 0 {
			failure = &invocation{retryable: true, circuitFailure: true}
		}
		want := 1
		if core.ClassifyError(failure).Retryable {
			want = 3
		}
		for _, kind := range []string{"chat", "chat stream", "responses", "responses stream", "messages"} {
			t.Run(strconv.Itoa(status)+" "+kind, func(t *testing.T) {
				t.Parallel()
				inner := &scripted{always: failure}
				_ = operate(execution.Resilient(inner, "provider", execution.Policy{Retry: execution.Retry{Attempts: 3}}), kind)
				if invokes, streams := inner.tries(); invokes+streams != want {
					t.Fatalf("tries=%d want=%d", invokes+streams, want)
				}
			})
		}
	}
	inner := &scripted{always: core.NewConfigurationError("compatibility", nil)}
	_ = operate(execution.Resilient(inner, "provider", execution.Policy{Retry: execution.Retry{Attempts: 2}}), "messages")
	if inner.invokes != 1 {
		t.Fatalf("a configuration error was tried %d times", inner.invokes)
	}
}

func TestResilientStatefulResponsesDisableRetries(t *testing.T) {
	t.Parallel()
	for name, body := range map[string]string{
		"previous response": `{"input":"hello","previous_response_id":"resp_fixture"}`,
		"conversation":      `{"input":"hello","conversation":"conv_fixture"}`,
		"vector store":      `{"input":"hello","tools":[{"type":"file_search","vector_store_ids":["vs_fixture"]}]}`,
		"stored response":   `{"input":"hello","store":true}`,
		"hosted tool":       `{"input":"hello","tools":[{"type":"mcp","server_label":"fixture"}]}`,
		"mcp approval":      `{"input":[{"type":"mcp_approval_response","approve":true}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			inner := &scripted{always: &invocation{retryable: true, circuitFailure: true}}
			provider := execution.Resilient(inner, "provider", execution.Policy{Retry: execution.Retry{Attempts: 3}})
			stateful := request(core.ModelSurfaceResponses, body)
			_, _ = provider.Invoke(context.Background(), stateful)
			_, _ = provider.Stream(context.Background(), stateful)
			if invokes, streams := inner.tries(); invokes != 1 || streams != 1 {
				t.Fatalf("invokes=%d streams=%d", invokes, streams)
			}
		})
	}
}
