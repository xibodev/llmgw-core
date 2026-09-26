package providers

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

func TestOpenAICompatibleRefusesBeforeSending(t *testing.T) {
	t.Parallel()
	_, server := newOpenAIBackend(t, func(http.ResponseWriter, *http.Request, int) { t.Error("a refused request reached the upstream") })
	provider := newTestOpenAICompatible(t, server, nil)
	chat := func(body string) core.Request { return openAIRequest(core.ModelSurfaceChatCompletions, "m", body, nil) }
	var failure *core.ProviderError
	for name, request := range map[string]core.Request{
		"text body":       {Surface: core.ModelSurfaceChatCompletions, Model: "m", Body: []byte(`{}`), ContentType: "text/plain"},
		"trailing body":   chat(`{} {}`),
		"array body":      chat(`[]`),
		"string messages": chat(`{"messages":"hi"}`),
		"message string":  chat(`{"messages":["hi"]}`),
		"forced string":   chat(`{"messages":[],"force_api_support":"yes"}`),
		"blank model":     openAIRequest(core.ModelSurfaceChatCompletions, " ", `{}`, nil),
	} {
		if _, err := provider.Invoke(context.Background(), request); !errors.As(err, &failure) || failure.Class != core.ProviderErrorInvalidRequest {
			t.Fatalf("%s: err = %v", name, err)
		}
	}
	for _, kind := range []string{core.TokenTypeGCPServiceAccount, core.TokenTypeAnthropicSetupToken} {
		credential := &core.Credential{Token: `{"private_key":"fixture"}`, TokenType: kind}
		_, err := provider.Stream(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "m", `{}`, credential))
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration || strings.Contains(err.Error(), "fixture") {
			t.Fatalf("%s credential: err = %v", kind, err)
		}
		if _, err := provider.ListModels(context.Background(), credential); !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration {
			t.Fatalf("%s catalog: err = %v", kind, err)
		}
	}
	var surface *core.SurfaceError
	for _, request := range []core.Request{
		openAIRequest(core.ModelSurfaceResponses, "chat-only", `{"input":"hi"}`, nil),
		openAIRequest(core.ModelSurfaceMessages, "chat-only", `{"messages":[]}`, nil),
	} {
		if _, err := provider.Invoke(context.Background(), request); !errors.As(err, &surface) || !core.ClassifyError(err).FailoverEligible {
			t.Fatalf("%s: err = %v", request.Surface, err)
		}
	}
}

// Ported from the gateway's TestV043OpenAIRejectsStructurallyInvalidSuccessPayloads:
// an answer that cannot be used counts against the provider and permits
// failover, but repeating it would get the same answer.
func TestOpenAICompatibleRejectsStructurallyInvalidAnswers(t *testing.T) {
	t.Parallel()
	for name, test := range map[string]struct {
		body    string
		surface core.ModelSurface
	}{
		"chat null":      {"null", core.ModelSurfaceChatCompletions},
		"chat empty":     {`{}`, core.ModelSurfaceChatCompletions},
		"chat no choice": {`{"choices":[]}`, core.ModelSurfaceChatCompletions},
		"chat bad":       {`{"choices":[17]}`, core.ModelSurfaceChatCompletions},
		"chat text":      {`Hello`, core.ModelSurfaceChatCompletions},
		"responses null": {"null", core.ModelSurfaceResponses},
		"responses {}":   {`{}`, core.ModelSurfaceResponses},
		"no output":      {`{"id":"resp_fixture","output":{}}`, core.ModelSurfaceResponses},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			_, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) { _, _ = io.WriteString(w, test.body) })
			provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) { config.RegistryID = "openai" })
			_, err := provider.Invoke(context.Background(), openAIRequest(test.surface, "m", `{"input":"hi"}`, nil))
			var failure *core.ProviderError
			if !errors.As(err, &failure) || failure.Class != core.ProviderErrorUpstream ||
				failure.Classification != (core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}) {
				t.Fatalf("err = %#v", err)
			}
		})
	}
}

// Ported from the gateway's TestOpenAIRejectsHTTP200SoftError.
func TestOpenAICompatibleRetriesAnErrorInASuccessfulAnswer(t *testing.T) {
	t.Parallel()
	for body, soft := range map[string]bool{
		`{"error":{"message":"service temporarily overloaded"}}`:              true,
		`{"error":"overloaded","choices":[{"message":{}}]}`:                   true,
		`{"error":{"code":17},"choices":[{"message":{}}]}`:                    true,
		`{"error":{"message":"  "},"choices":[{"message":{"content":"ok"}}]}`: false,
		`{"error":null,"choices":[{"message":{"content":"ok"}}]}`:             false,
	} {
		_, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) { _, _ = io.WriteString(w, body) })
		provider := newTestOpenAICompatible(t, server, nil)
		_, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "m", `{}`, nil))
		retryable := core.ClassifyError(err) == core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}
		if (err != nil) != soft || retryable != soft || (soft && strings.Contains(err.Error(), "overloaded")) {
			t.Fatalf("%s: err = %v", body, err)
		}
	}
}

// A refusal is classified by the gateway's status set, with Retry-After,
// on every operation, and its message never quotes the upstream.
func TestOpenAICompatibleFailuresCarryTheirStatusAndRetryAfter(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		status     int
		retryAfter string
		class      core.ProviderErrorClass
		transient  bool
	}{
		{http.StatusTooManyRequests, "7", core.ProviderErrorRateLimited, true},
		{http.StatusServiceUnavailable, "2", core.ProviderErrorUpstream, true},
		{http.StatusUnauthorized, "", core.ProviderErrorAuth, false},
		{http.StatusForbidden, "", core.ProviderErrorForbidden, false},
		{http.StatusBadRequest, "", core.ProviderErrorUpstream, false},
		{http.StatusNotImplemented, "", core.ProviderErrorUpstream, false},
	} {
		_, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
			if test.retryAfter != "" {
				w.Header().Set("Retry-After", test.retryAfter)
			}
			w.WriteHeader(test.status)
			_, _ = io.WriteString(w, `{"error":{"code":"fixture_code","message":"private upstream text"}}`)
		})
		provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) { config.RegistryID = "openai" })
		for _, surface := range []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses} {
			request := openAIRequest(surface, "m", `{"input":"hi","messages":[]}`, nil)
			_, invokeErr := provider.Invoke(context.Background(), request)
			_, streamErr := provider.Stream(context.Background(), request)
			for _, err := range []error{invokeErr, streamErr} {
				var failure *core.ProviderError
				wantRetry := time.Duration(0)
				if test.retryAfter != "" {
					wantRetry = map[string]time.Duration{"7": 7 * time.Second, "2": 2 * time.Second}[test.retryAfter]
				}
				want := core.ProviderErrorClassification{
					StatusCode: test.status, Retryable: test.transient, FailoverEligible: test.transient, CircuitFailure: test.transient, RetryAfter: wantRetry,
				}
				if !errors.As(err, &failure) || failure.Class != test.class || failure.Classification != want ||
					failure.Message != fmt.Sprintf("OpenAI-compatible returned HTTP %d (code=fixture_code)", test.status) {
					t.Fatalf("%d %s: err = %#v", test.status, surface, err)
				}
				var cause *InvocationError
				if !errors.As(err, &cause) || !strings.Contains(cause.Msg, "private upstream text") {
					t.Fatalf("%d %s: cause = %#v", test.status, surface, cause)
				}
			}
		}
	}
}

func TestOpenAICompatibleReportsTransportFailuresAndCallerCancellation(t *testing.T) {
	t.Parallel()
	_, server := newOpenAIBackend(t, answerOpenAIChat)
	provider := newTestOpenAICompatible(t, server, nil)
	server.Close()
	request := openAIRequest(core.ModelSurfaceChatCompletions, "m", `{}`, nil)
	var failure *core.ProviderError
	if _, err := provider.Invoke(context.Background(), request); !errors.As(err, &failure) || failure.Class != core.ProviderErrorTransport ||
		failure.Classification != (core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}) {
		t.Fatalf("unreachable: err = %#v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := provider.Stream(ctx, request); !errors.As(err, &failure) || failure.Classification != (core.ProviderErrorClassification{}) ||
		!errors.Is(err, context.Canceled) {
		t.Fatalf("canceled: err = %#v", err)
	}
}

// A stream that breaks after it opened fails as the gateway's reader fails:
// the records before the break arrive as sent, and the break permits
// failover only while nothing reached the caller, which is the product's
// to decide.
func TestOpenAICompatibleStreamsRecordsAsSentUntilTheyBreak(t *testing.T) {
	t.Parallel()
	const first = "data: {\"id\":\"chatcmpl_fixture\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Partial\"}}]}\n\n"
	_, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		w.Header().Set("Content-Type", core.ContentTypeEventStream)
		_, _ = io.WriteString(w, ": keepalive\n\n"+first)
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler)
	})
	provider := newTestOpenAICompatible(t, server, nil)
	stream, err := provider.Stream(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "m", `{"messages":[]}`, nil))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if frame, err := stream.Next(); err != nil || string(frame) != first {
		t.Fatalf("frame = %q, err = %v", frame, err)
	}
	_, err = stream.Next()
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorTransport ||
		failure.Classification != (core.ProviderErrorClassification{FailoverEligible: true}) {
		t.Fatalf("break: err = %#v", err)
	}
	if losses := core.StreamLosses(stream); losses != nil {
		t.Fatalf("losses = %v", losses)
	}
}
