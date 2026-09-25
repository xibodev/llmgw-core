package providers

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// The completion is what the gateway's OllamaProvider.Complete returns for
// the same answer, encoded as the gateway encodes it.
func TestOllamaConvertsTheAnswerAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	answer := `{"model":"qwen3:8b","created_at":"2026-01-01T00:00:00Z","message":{"role":"assistant","content":"It is a <cat> & a dog.",` +
		`"tool_calls":[{"function":{"name":"lookup","arguments":{"q":"cat","limit":3.0}}}]},"done":true,"prompt_eval_count":12,"eval_count":5}`
	want := `{"choices":[{"finish_reason":"tool_calls","index":0,"message":{"content":"It is a \u003ccat\u003e \u0026 a dog.","role":"assistant",` +
		`"tool_calls":[{"function":{"arguments":"{\"limit\":3,\"q\":\"cat\"}","name":"lookup"},"id":"call_0","index":0,"type":"function"}]}}],` +
		`"id":"chatcmpl-ollama","model":"qwen3:8b","object":"chat.completion","usage":{"completion_tokens":5,"prompt_tokens":12,"total_tokens":17}}`
	provider, _ := ollamaDaemon(t, 200, answer)
	response, err := provider.Invoke(t.Context(), ollamaChatRequest("qwen3:8b", `{"messages":[{"role":"user","content":"What is it?"}]}`))
	if err != nil || string(response.Body) != want || response.ContentType != core.ContentTypeJSON || response.Losses != nil {
		t.Fatalf("response = %s %q %v, err = %v", response.Body, response.ContentType, response.Losses, err)
	}
	provider, _ = ollamaDaemon(t, 200, `{"message":{"content":"Hi","tool_calls":[]},"prompt_eval_count":2.9}`)
	response, err = provider.Invoke(t.Context(), ollamaChatRequest("m", `{"messages":[]}`))
	if want := `{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"Hi","role":"assistant"}}],"id":"chatcmpl-ollama","model":"m","object":"chat.completion","usage":{"completion_tokens":0,"prompt_tokens":2,"total_tokens":2}}`; err != nil || string(response.Body) != want {
		t.Fatalf("response = %s, err = %v", response.Body, err)
	}
	for _, answer := range []string{`not json`, `null`, `{"done":true}`, `{"message":"hi"}`, `{"message":{"content":"","tool_calls":[]},"done":true}`} {
		provider, _ := ollamaDaemon(t, 200, answer)
		_, err := provider.Invoke(t.Context(), ollamaChatRequest("m", `{"messages":[]}`))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorUpstream ||
			failure.Classification != (core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}) {
			t.Fatalf("answer %s: err = %#v, want an unusable answer", answer, err)
		}
	}
}

func TestOllamaRefusesWhatItCannotServeBeforeSending(t *testing.T) {
	t.Parallel()
	provider, calls := ollamaDaemon(t, 200, `{}`)
	responses := ollamaChatRequest("m", `{"input":"hi"}`)
	responses.Surface = core.ModelSurfaceResponses
	var surfaceErr *core.SurfaceError
	if _, err := provider.Invoke(t.Context(), responses); !errors.As(err, &surfaceErr) {
		t.Fatalf("Responses: err = %v, want a surface error", err)
	}
	if _, err := provider.Stream(t.Context(), responses); !errors.As(err, &surfaceErr) {
		t.Fatalf("Responses stream: err = %v, want a surface error", err)
	}
	plain := ollamaChatRequest("m", `{"messages":[]}`)
	plain.ContentType = "text/plain"
	for name, request := range map[string]core.Request{
		"no model":        ollamaChatRequest(" ", `{"messages":[]}`),
		"not JSON":        plain,
		"not an object":   ollamaChatRequest("m", `[]`),
		"null":            ollamaChatRequest("m", `null`),
		"trailing data":   ollamaChatRequest("m", `{"messages":[]} {}`),
		"string messages": ollamaChatRequest("m", `{"messages":"hi"}`),
		"message number":  ollamaChatRequest("m", `{"messages":[1]}`),
	} {
		_, invokeErr := provider.Invoke(t.Context(), request)
		_, streamErr := provider.Stream(t.Context(), request)
		for _, err := range []error{invokeErr, streamErr} {
			var failure *core.ProviderError
			if !errors.As(err, &failure) || failure.Class != core.ProviderErrorInvalidRequest || core.ClassifyError(err).Disposition() != core.DispositionTerminal {
				t.Fatalf("%s: err = %#v, want an invalid request", name, err)
			}
		}
	}
	if got := calls(); len(got) != 0 {
		t.Fatalf("upstream = %+v, want nothing sent", got)
	}
}

// Ported from the gateway's TestProviderNon2xxErrorsAreSanitizedAndKeepStatus,
// with the classification a Runtime reads.
func TestOllamaClassifiesFailuresAsTheGateway(t *testing.T) {
	t.Parallel()
	email, gatewayToken, gskToken := "provider-owner@example.test", syntheticGatewayToken(), "gsk_"+strings.Repeat("z", 24)
	body := fmt.Sprintf(`{"error":{"message":"account %s used %s with Bearer %s"}}`, email, gatewayToken, gskToken)
	request := ollamaChatRequest("m", `{"messages":[{"role":"user","content":"hi"}]}`)
	for _, tc := range []struct {
		status int
		want   core.ProviderErrorClassification
	}{
		{429, core.ProviderErrorClassification{StatusCode: 429, Retryable: true, FailoverEligible: true, CircuitFailure: true, RetryAfter: 7 * time.Second}},
		{503, core.ProviderErrorClassification{StatusCode: 503, Retryable: true, FailoverEligible: true, CircuitFailure: true, RetryAfter: 7 * time.Second}},
		{401, core.ProviderErrorClassification{StatusCode: 401, RetryAfter: 7 * time.Second}},
		{404, core.ProviderErrorClassification{StatusCode: 404, RetryAfter: 7 * time.Second}},
	} {
		provider, _ := ollamaDaemon(t, tc.status, body)
		_, invokeErr := provider.Invoke(t.Context(), request)
		_, streamErr := provider.Stream(t.Context(), request)
		for _, err := range []error{invokeErr, streamErr} {
			var failure *core.ProviderError
			var cause *InvocationError
			if !errors.As(err, &failure) || !errors.As(err, &cause) || failure.Classification != tc.want ||
				failure.Message != fmt.Sprintf("Ollama returned HTTP %d", tc.status) || !strings.Contains(cause.Msg, redactedDiagnostic) {
				t.Fatalf("HTTP %d: err = %#v", tc.status, err)
			}
			for _, secret := range []string{email, gatewayToken, gskToken} {
				if strings.Contains(failure.Message+cause.Msg, secret) {
					t.Fatalf("HTTP %d: the error exposed %q", tc.status, secret)
				}
			}
		}
	}

	closed := httptest.NewServer(http.NotFoundHandler())
	closed.Close()
	provider, err := NewOllama(OllamaConfig{BaseURL: closed.URL})
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	for ctx, want := range map[context.Context]core.ProviderErrorClassification{
		t.Context(): {Retryable: true, FailoverEligible: true, CircuitFailure: true},
		canceled:    {},
	} {
		_, invokeErr := provider.Invoke(ctx, request)
		_, streamErr := provider.Stream(ctx, request)
		for _, err := range []error{invokeErr, streamErr} {
			var failure *core.ProviderError
			if !errors.As(err, &failure) || failure.Class != core.ProviderErrorTransport || failure.Classification != want {
				t.Fatalf("transport failure = %#v, want %+v", err, want)
			}
		}
	}
}
