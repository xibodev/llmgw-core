package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	copilotauth "github.com/xibodev/llm-provider-auth/copilot"
	core "github.com/xibodev/llmgw-core"
)

// A second 401 fails with its status and permits nothing, so a Runtime
// refreshes the credential or reports it; every operation replays once.
func TestCopilotReportsARepeatedRejectionWith401(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilotChat)
	backend.update(func(b *copilotBackend) { b.rejectAll = true })
	provider := newFixtureCopilot(t, backend)
	credential := copilotCredential("oauth-a")
	chat := copilotRequest(core.ModelSurfaceChatCompletions, "gpt-fixture", `{"messages":[]}`, credential)
	for name, operation := range map[string]func() error{
		"invoke": func() error { _, err := provider.Invoke(context.Background(), chat); return err },
		"stream": func() error { _, err := provider.Stream(context.Background(), chat); return err },
		"list":   func() error { _, err := provider.ListModels(context.Background(), credential); return err },
	} {
		err := operation()
		assertCopilotFailure(t, err, core.ProviderErrorAuth, core.ProviderErrorClassification{StatusCode: http.StatusUnauthorized})
		if _, calls := backend.take(); len(calls) != 2 {
			t.Fatalf("%s: %d calls, want the rejected request replayed once", name, len(calls))
		}
	}
}

func TestCopilotReportsSessionExchangeFailuresAsTheGateway(t *testing.T) {
	t.Parallel()
	for status, want := range map[int]struct {
		class          core.ProviderErrorClass
		classification core.ProviderErrorClassification
	}{
		http.StatusUnauthorized:       {core.ProviderErrorAuth, core.ProviderErrorClassification{StatusCode: 401}},
		http.StatusForbidden:          {core.ProviderErrorForbidden, core.ProviderErrorClassification{StatusCode: 403}},
		http.StatusNotFound:           {core.ProviderErrorUpstream, core.ProviderErrorClassification{StatusCode: 404, FailoverEligible: true}},
		http.StatusTooManyRequests:    {core.ProviderErrorRateLimited, core.ProviderErrorClassification{StatusCode: 429, Retryable: true, FailoverEligible: true, CircuitFailure: true}},
		http.StatusServiceUnavailable: {core.ProviderErrorUpstream, core.ProviderErrorClassification{StatusCode: 503, Retryable: true, FailoverEligible: true, CircuitFailure: true}},
	} {
		backend := newCopilotBackend(t, answerCopilotChat)
		backend.update(func(b *copilotBackend) { b.exchangeFails["oauth-a"] = status })
		err := invokeCopilotChat(t, newFixtureCopilot(t, backend), copilotCredential("oauth-a"))
		failure := assertCopilotFailure(t, err, want.class, want.classification)
		var auth *copilotauth.AuthError
		if !errors.As(err, &auth) || failure.Message != auth.Msg {
			t.Fatalf("%d: message %q, want the auth client's %v", status, failure.Message, auth)
		}
		if rejected := errors.Is(err, copilotauth.ErrOAuthTokenRejected); rejected != (status == http.StatusUnauthorized) {
			t.Fatalf("%d: ErrOAuthTokenRejected = %v", status, rejected)
		}
		if _, calls := backend.take(); len(calls) != 0 {
			t.Fatalf("%d: %d API calls without a session", status, len(calls))
		}
	}
}

func TestCopilotClassifiesAPIFailuresAsTheGateway(t *testing.T) {
	t.Parallel()
	upstream := core.ProviderErrorUpstream
	for name, testCase := range map[string]struct {
		status         int
		header, body   string
		class          core.ProviderErrorClass
		classification core.ProviderErrorClassification
	}{
		"bad request":        {400, "", `{"error":{"message":"private upstream text","code":"model_not_supported","type":"invalid_request_error"}}`, upstream, core.ProviderErrorClassification{StatusCode: 400}},
		"forbidden":          {403, "", ``, core.ProviderErrorForbidden, core.ProviderErrorClassification{StatusCode: 403}},
		"not found":          {404, "", ``, upstream, core.ProviderErrorClassification{StatusCode: 404}},
		"rate limited":       {429, "7", ``, core.ProviderErrorRateLimited, core.ProviderErrorClassification{StatusCode: 429, Retryable: true, FailoverEligible: true, CircuitFailure: true, RetryAfter: 7 * time.Second}},
		"server error":       {500, "", ``, upstream, core.ProviderErrorClassification{StatusCode: 500, Retryable: true, FailoverEligible: true, CircuitFailure: true}},
		"not implemented":    {501, "", ``, upstream, core.ProviderErrorClassification{StatusCode: 501}},
		"soft error":         {200, "", `{"error":{"code":"overloaded"},"choices":[{}]}`, upstream, core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}},
		"not JSON":           {200, "", `not-json`, upstream, core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"empty object":       {200, "", `{}`, upstream, core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"no choices":         {200, "", `{"choices":[]}`, upstream, core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"choice not objects": {200, "", `{"choices":["stop"]}`, upstream, core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			backend := newCopilotBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
				if testCase.header != "" {
					w.Header().Set("Retry-After", testCase.header)
				}
				w.WriteHeader(testCase.status)
				_, _ = io.WriteString(w, testCase.body)
			})
			err := invokeCopilotChat(t, newFixtureCopilot(t, backend), copilotCredential("oauth-a"))
			failure := assertCopilotFailure(t, err, testCase.class, testCase.classification)
			if strings.Contains(failure.Message, "private") {
				t.Fatalf("message %q repeats the upstream's text", failure.Message)
			}
			if name == "bad request" && failure.Message != "Copilot upstream request failed with status 400 (code=model_not_supported, type=invalid_request_error)" {
				t.Fatalf("message = %q", failure.Message)
			}
		})
	}
}

// A blank error beside the choices is no error to the gateway.
func TestCopilotServesAnAnswerWithABlankError(t *testing.T) {
	t.Parallel()
	for _, answer := range []string{`{"error":"","choices":[{}]}`, `{"error":{"message":" "},"choices":[{}]}`, `{"error":null,"choices":[{}]}`} {
		backend := newCopilotBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) { _, _ = io.WriteString(w, answer) })
		if err := invokeCopilotChat(t, newFixtureCopilot(t, backend), nil); err != nil {
			t.Fatalf("%s: %v", answer, err)
		}
	}
}

type copilotRoundTrip func(*http.Request) (*http.Response, error)

func (f copilotRoundTrip) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

func TestCopilotRefusesWhatItCannotSendAndSendsNothing(t *testing.T) {
	t.Parallel()
	chat := func(body string, credential *core.Credential) core.Request {
		return copilotRequest(core.ModelSurfaceChatCompletions, "gpt-fixture", body, credential)
	}
	configuration := core.ProviderErrorClassification{FailoverEligible: true}
	for name, testCase := range map[string]struct {
		adjust  func(*CopilotConfig, *copilotauth.Config)
		request core.Request
		kind    error
		class   core.ProviderErrorClass
		want    core.ProviderErrorClassification
	}{
		"proxy disabled": {func(_ *CopilotConfig, auth *copilotauth.Config) { auth.AllowProxy = false }, chat(`{}`, copilotCredential("oauth-a")),
			copilotauth.ErrProxyDisabled, core.ProviderErrorConfiguration, configuration},
		"no product token": {func(_ *CopilotConfig, auth *copilotauth.Config) { auth.OAuthToken = "" }, chat(`{}`, nil),
			copilotauth.ErrNoOAuthToken, core.ProviderErrorConfiguration, configuration},
		"credential without a token": {nil, chat(`{}`, &core.Credential{APIKey: "not-an-oauth-token"}),
			nil, core.ProviderErrorConfiguration, configuration},
		"not JSON":             {nil, core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m", Body: []byte(`{}`), ContentType: "text/plain"}, nil, core.ProviderErrorInvalidRequest, core.ProviderErrorClassification{}},
		"not an object":        {nil, chat(`[]`, nil), nil, core.ProviderErrorInvalidRequest, core.ProviderErrorClassification{}},
		"trailing data":        {nil, chat(`{} {}`, nil), nil, core.ProviderErrorInvalidRequest, core.ProviderErrorClassification{}},
		"messages not objects": {nil, chat(`{"messages":["hi"]}`, nil), nil, core.ProviderErrorInvalidRequest, core.ProviderErrorClassification{}},
		"messages not a list":  {nil, chat(`{"messages":{"role":"user"}}`, nil), nil, core.ProviderErrorInvalidRequest, core.ProviderErrorClassification{}},
		"control not boolean":  {nil, chat(`{"force_api_support":"yes"}`, nil), nil, core.ProviderErrorInvalidRequest, core.ProviderErrorClassification{}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			backend := newCopilotBackend(t, answerCopilotChat)
			var adjust []func(*CopilotConfig, *copilotauth.Config)
			if testCase.adjust != nil {
				adjust = append(adjust, testCase.adjust)
			}
			provider := newFixtureCopilot(t, backend, adjust...)
			_, err := provider.Invoke(context.Background(), testCase.request)
			assertCopilotFailure(t, err, testCase.class, testCase.want)
			if testCase.kind != nil && !errors.Is(err, testCase.kind) {
				t.Fatalf("err = %v, want %v", err, testCase.kind)
			}
			if exchanges, calls := backend.take(); len(exchanges) != 0 || len(calls) != 0 {
				t.Fatalf("sent %v and %+v", exchanges, calls)
			}
		})
	}
}

func TestCopilotReportsTransportFailuresAndCancellation(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilotChat)
	refused := errors.New("connection refused")
	provider := newFixtureCopilot(t, backend, func(config *CopilotConfig, _ *copilotauth.Config) {
		config.Client = &http.Client{Transport: copilotRoundTrip(func(*http.Request) (*http.Response, error) { return nil, refused })}
	})
	err := invokeCopilotChat(t, provider, copilotCredential("oauth-a"))
	assertCopilotFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true})
	if !errors.Is(err, refused) {
		t.Fatalf("err = %v, want the transport's error as its cause", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = provider.Invoke(ctx, copilotRequest(core.ModelSurfaceChatCompletions, "gpt-fixture", `{}`, copilotCredential("oauth-b")))
	assertCopilotFailure(t, err, "", core.ProviderErrorClassification{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want the caller's cancellation", err)
	}
}

// Auth takes no context, so a caller that gives up stops waiting for the
// exchange; the exchange completes and its session serves the next request.
func TestCopilotStopsWaitingForAnExchangeWhenTheCallerGivesUp(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilotChat)
	hold := make(chan struct{})
	backend.update(func(b *copilotBackend) { b.hold = hold })
	provider := newFixtureCopilot(t, backend)
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() {
		_, err := provider.Invoke(ctx, copilotRequest(core.ModelSurfaceChatCompletions, "gpt-fixture", `{}`, copilotCredential("oauth-a")))
		result <- err
	}()
	<-hold
	cancel()
	err := <-result
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want the caller's cancellation", err)
	}
	assertCopilotFailure(t, err, "", core.ProviderErrorClassification{})
	backend.update(func(b *copilotBackend) { b.hold = nil })
	hold <- struct{}{}
	// The next request joins the exchange still in flight or finds its
	// session kept; either way no second exchange is made.
	if err := invokeCopilotChat(t, provider, copilotCredential("oauth-a")); err != nil {
		t.Fatal(err)
	}
	if exchanges, calls := backend.take(); len(exchanges) != 1 || len(calls) != 1 {
		t.Fatalf("exchanges = %v, calls = %d, want the abandoned exchange's session reused", exchanges, len(calls))
	}
}
