package providers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

type codexRoundTrip func(*http.Request) (*http.Response, error)

func (f codexRoundTrip) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

func newFixtureCodex(t *testing.T, server *httptest.Server) *Codex {
	t.Helper()
	provider, err := NewCodex(CodexConfig{
		Instructions: "Required instructions", ClientVersion: fixtureCodexClientVersion,
		ResponsesURL: server.URL + "/responses", ModelsURL: server.URL + "/models", Client: server.Client(),
		Now: func() time.Time { return time.Date(2026, time.September, 21, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func fixtureCodexCredential() *core.Credential {
	return &core.Credential{Token: "caller-token", AccountID: "account-fixture", TokenType: "Bearer"}
}

func codexResponsesRequest(body string) core.Request {
	return core.Request{
		Surface: core.ModelSurfaceResponses, Model: "exact-model", Body: []byte(body),
		ContentType: core.ContentTypeJSON, Credential: fixtureCodexCredential(),
	}
}

// assertCodexFailure checks the canonical error a Codex operation returns.
func assertCodexFailure(t *testing.T, err error, class core.ProviderErrorClass, want core.ProviderErrorClassification) {
	t.Helper()
	var failure *core.ProviderError
	if !errors.As(err, &failure) {
		t.Fatalf("error = %T %v, want a *core.ProviderError", err, err)
	}
	if failure.Class != class || failure.Classification != want || core.ClassifyError(err) != want {
		t.Fatalf("error %q: class %q classification %+v, want %q %+v", err, failure.Class, failure.Classification, class, want)
	}
}

const codexCompletedStream = `data: {"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}}` + "\n\n"

func TestNewCodexRequiresInstructionsAndClientVersion(t *testing.T) {
	t.Parallel()
	if _, err := NewCodex(CodexConfig{ClientVersion: fixtureCodexClientVersion}); err == nil || err.Error() != "Codex instructions are required" {
		t.Fatalf("missing instructions: err = %v", err)
	}
	if _, err := NewCodex(CodexConfig{Instructions: "required", ClientVersion: " "}); err == nil || err.Error() != "Codex client version is required" {
		t.Fatalf("missing client version: err = %v", err)
	}
	// The verified version is there to pass, not a default.
	if _, err := NewCodex(CodexConfig{Instructions: "required", ClientVersion: CodexVerifiedClientVersion}); err != nil {
		t.Fatalf("verified client version: err = %v", err)
	}
}

// The same payload must reach Codex identically through both provider APIs:
// they share one transport, and only the credential's origin differs.
func TestCodexSendsWhatCodexProviderSends(t *testing.T) {
	t.Parallel()
	type captured struct {
		header http.Header
		body   string
	}
	requests := make(chan captured, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		requests <- captured{header: r.Header.Clone(), body: string(body)}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, codexCompletedStream)
	}))
	defer server.Close()
	payload := map[string]any{
		"input":        []any{map[string]any{"role": "user", "content": "continue"}},
		"instructions": "Caller instructions", "previous_response_id": "resp_previous", "conversation": "conv_1",
		"include": []any{"reasoning.encrypted_content"}, "reasoning": map[string]any{"effort": "high", "summary": "auto"},
		"tools": []any{map[string]any{"type": "web_search"}}, "prompt_cache_key": "fixture-cache",
		"force_api_support": true, "stream": false, "background": false,
	}
	legacy, err := newFixtureCodexProvider(t, server, "Required instructions").CompleteResponses(context.Background(), "exact-model", payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(payload)
	response, err := newFixtureCodex(t, server).Invoke(context.Background(), codexResponsesRequest(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	fromLegacy, fromCodex := <-requests, <-requests
	if !reflect.DeepEqual(fromCodex, fromLegacy) {
		t.Fatalf("Codex request drifted from CodexProvider:\n got %+v\nwant %+v", fromCodex, fromLegacy)
	}
	want, _ := json.Marshal(legacy)
	if response.ContentType != core.ContentTypeJSON || string(response.Body) != string(want) || response.Losses != nil {
		t.Fatalf("response = %s %s %v, want %s", response.ContentType, response.Body, response.Losses, want)
	}
}

// Shorthand string input reaches Codex as the single user message it
// abbreviates, which the gateway's characterization pins.
func TestCodexSendsShorthandInputAsOneUserMessage(t *testing.T) {
	t.Parallel()
	type captured struct {
		request *http.Request
		body    map[string]any
	}
	requests := make(chan captured, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		requests <- captured{request: r.Clone(context.Background()), body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, codexCompletedStream)
	}))
	defer server.Close()
	_, err := newFixtureCodex(t, server).Invoke(context.Background(), codexResponsesRequest(`{
		"input": "continue", "instructions": "Caller instructions", "previous_response_id": "resp_previous",
		"conversation": "conv_1", "include": ["reasoning.encrypted_content"], "reasoning": {"effort": "high", "summary": "auto"},
		"tools": [{"type": "web_search"}], "prompt_cache_key": "fixture-cache", "force_api_support": true, "stream": false
	}`))
	if err != nil {
		t.Fatal(err)
	}
	request := <-requests
	assertCodexRequestFixture(t, request.request, request.body, "request-codex-native.json")
}

func TestCodexStreamsTheUpstreamFramesUnchanged(t *testing.T) {
	t.Parallel()
	frames := []string{
		": keepalive\r\nretry: 1000\r\n\r\n",
		"event: response.created\r\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\r\ndata: \"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"output\":[]}}\r\n\r\n",
		"event: response.future.delta\ndata: {\"type\":\"response.future.delta\",\"sequence_number\":1,\"custom\":{\"nested\":true}}\n\n",
		"data: {\"type\":\"response.output_text.delta\",\"sequence_number\":2,\"delta\":\"hello\"}\n\n",
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":3,\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n\n",
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join(frames, ""))
	}))
	defer server.Close()
	stream, err := newFixtureCodex(t, server).Stream(context.Background(), codexResponsesRequest(`{"input":"go"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var got []string
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, string(frame))
	}
	if !reflect.DeepEqual(got, frames) {
		t.Fatalf("frames = %q, want %q", got, frames)
	}
}

func TestCodexStreamFailureAfterAFrameIsAProviderError(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\ndata: {not-json}\n\n")
	}))
	defer server.Close()
	stream, err := newFixtureCodex(t, server).Stream(context.Background(), codexResponsesRequest(`{"input":"go"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if frame, err := stream.Next(); err != nil || !strings.Contains(string(frame), "partial") {
		t.Fatalf("first frame = %q err = %v", frame, err)
	}
	_, err = stream.Next()
	assertCodexFailure(t, err, core.ProviderErrorUpstream, core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true})
	var invocation *InvocationError
	if !errors.As(err, &invocation) {
		t.Fatal("the transport's *InvocationError is no longer the cause")
	}
}

func TestCodexFailsClosedWithoutAChatGPTToken(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls.Add(1) }))
	defer server.Close()
	provider := newFixtureCodex(t, server)
	configuration := core.ProviderErrorClassification{FailoverEligible: true}
	for name, credential := range map[string]*core.Credential{
		"no credential": nil,
		"empty":         {},
		"blank token":   {Token: " \t", AccountID: "account-fixture"},
		"an API key":    core.CredentialFromRecord("key", core.APIKeyRecord("fixture-api-key")),
	} {
		t.Run(name, func(t *testing.T) {
			request := codexResponsesRequest(`{"input":"hello"}`)
			request.Credential = credential
			_, err := provider.Invoke(context.Background(), request)
			assertCodexFailure(t, err, core.ProviderErrorConfiguration, configuration)
			_, err = provider.Stream(context.Background(), request)
			assertCodexFailure(t, err, core.ProviderErrorConfiguration, configuration)
			_, err = provider.ListModels(context.Background(), credential)
			assertCodexFailure(t, err, core.ProviderErrorConfiguration, configuration)
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("upstream received %d requests without a token", calls.Load())
	}
}

func TestCodexServesResponsesAndChatNatively(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls.Add(1) }))
	defer server.Close()
	provider := newFixtureCodex(t, server)
	if got := provider.NativeSurfaces("any-model"); !reflect.DeepEqual(got, []core.ModelSurface{core.ModelSurfaceResponses, core.ModelSurfaceChatCompletions}) {
		t.Fatalf("native surfaces = %v", got)
	}
	responsesOnly, err := NewCodex(CodexConfig{Instructions: "required", ClientVersion: fixtureCodexClientVersion, ResponsesOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if got := responsesOnly.NativeSurfaces("any-model"); !reflect.DeepEqual(got, []core.ModelSurface{core.ModelSurfaceResponses}) {
		t.Fatalf("Responses-only native surfaces = %v", got)
	}
	chat := codexResponsesRequest(`{"messages":[{"role":"user","content":"hello"}]}`)
	chat.Surface = core.ModelSurfaceChatCompletions
	messages := codexResponsesRequest(`{"max_tokens":8,"messages":[{"role":"user","content":"hello"}]}`)
	messages.Surface = core.ModelSurfaceMessages
	for name, refused := range map[string]struct {
		provider *Codex
		request  core.Request
	}{
		"Messages":                 {provider, messages},
		"Chat when Responses-only": {responsesOnly, chat},
	} {
		var surface *core.SurfaceError
		if _, err := refused.provider.Invoke(context.Background(), refused.request); !errors.As(err, &surface) || !core.ClassifyError(err).FailoverEligible {
			t.Fatalf("%s invoke err = %v, want a *core.SurfaceError", name, err)
		}
		if _, err := refused.provider.Stream(context.Background(), refused.request); !errors.As(err, &surface) {
			t.Fatalf("%s stream err = %v, want a *core.SurfaceError", name, err)
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("upstream received %d requests for a surface Codex lacks", calls.Load())
	}
}

// A malformed request fails everywhere; a valid one Codex cannot serve may
// succeed on another Responses target, so it permits failover.
func TestCodexRefusesWhatItCannotServeBeforeSending(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls.Add(1) }))
	defer server.Close()
	provider := newFixtureCodex(t, server)
	for name, tc := range map[string]struct {
		model, contentType, body string
		class                    core.ProviderErrorClass
	}{
		"sampling field":      {body: `{"input":"hi","temperature":0.2}`, class: core.ProviderErrorUnsupported},
		"stored response":     {body: `{"input":"hi","store":true}`, class: core.ProviderErrorUnsupported},
		"background response": {body: `{"input":"hi","background":true}`, class: core.ProviderErrorUnsupported},
		"unproven tool":       {body: `{"input":"hi","tools":[{"type":"computer"}]}`, class: core.ProviderErrorUnsupported},
		"no input":            {body: `{"instructions":"hi"}`, class: core.ProviderErrorInvalidRequest},
		"instructions object": {body: `{"input":"hi","instructions":{}}`, class: core.ProviderErrorInvalidRequest},
		"padded model":        {model: " exact-model ", body: `{"input":"hi"}`, class: core.ProviderErrorInvalidRequest},
		"form body":           {contentType: "application/x-www-form-urlencoded", body: `input=hi`, class: core.ProviderErrorInvalidRequest},
		"array body":          {body: `[{"input":"hi"}]`, class: core.ProviderErrorInvalidRequest},
		"null body":           {body: `null`, class: core.ProviderErrorInvalidRequest},
		"trailing data":       {body: `{"input":"hi"} {"input":"again"}`, class: core.ProviderErrorInvalidRequest},
	} {
		t.Run(name, func(t *testing.T) {
			request := codexResponsesRequest(tc.body)
			if tc.model != "" {
				request.Model = tc.model
			}
			if tc.contentType != "" {
				request.ContentType = tc.contentType
			}
			want := core.ProviderErrorClassification{}
			if tc.class == core.ProviderErrorUnsupported {
				want.FailoverEligible = true
			}
			_, err := provider.Invoke(context.Background(), request)
			assertCodexFailure(t, err, tc.class, want)
			_, err = provider.Stream(context.Background(), request)
			assertCodexFailure(t, err, tc.class, want)
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("upstream received %d requests Codex cannot serve", calls.Load())
	}
}

// A 401 must reach the Runtime as status 401, from inference and the
// catalog alike, so it refreshes the credential and replays.
func TestCodexClassifiesUpstreamStatuses(t *testing.T) {
	t.Parallel()
	for name, tc := range map[string]struct {
		status     int
		retryAfter string
		class      core.ProviderErrorClass
		want       core.ProviderErrorClassification
	}{
		"rejected token": {status: 401, class: core.ProviderErrorAuth, want: core.ProviderErrorClassification{StatusCode: 401}},
		"forbidden":      {status: 403, class: core.ProviderErrorForbidden, want: core.ProviderErrorClassification{StatusCode: 403}},
		"bad request":    {status: 400, class: core.ProviderErrorUpstream, want: core.ProviderErrorClassification{StatusCode: 400}},
		"rate limited": {status: 429, retryAfter: "7", class: core.ProviderErrorRateLimited,
			want: core.ProviderErrorClassification{StatusCode: 429, Retryable: true, FailoverEligible: true, RetryAfter: 7 * time.Second}},
		"unavailable": {status: 503, class: core.ProviderErrorUpstream,
			want: core.ProviderErrorClassification{StatusCode: 503, Retryable: true, FailoverEligible: true, CircuitFailure: true}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if tc.retryAfter != "" {
					w.Header().Set("Retry-After", tc.retryAfter)
				}
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, `{"error":{"code":"fixture_code","message":"Bearer fixture-secret"}}`)
			}))
			defer server.Close()
			provider := newFixtureCodex(t, server)
			_, err := provider.Invoke(context.Background(), codexResponsesRequest(`{"input":"hi"}`))
			assertCodexFailure(t, err, tc.class, tc.want)
			if strings.Contains(err.Error(), "fixture-secret") {
				t.Fatalf("error exposed the upstream body: %v", err)
			}
			_, err = provider.Stream(context.Background(), codexResponsesRequest(`{"input":"hi"}`))
			assertCodexFailure(t, err, tc.class, tc.want)
			_, err = provider.ListModels(context.Background(), fixtureCodexCredential())
			assertCodexFailure(t, err, tc.class, tc.want)
			var catalog *CatalogError
			if !errors.As(err, &catalog) || catalog.Status != tc.status {
				t.Fatalf("catalog cause = %#v", catalog)
			}
		})
	}
}

func TestCodexClassifiesMalformedResponses(t *testing.T) {
	t.Parallel()
	malformed := core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}
	for name, tc := range map[string]struct{ contentType, body string }{
		"malformed event":        {"text/event-stream", "data: {not-json}\n\n"},
		"no terminal event":      {"text/event-stream", "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n"},
		"JSON instead of events": {"application/json", `{"id":"resp_1","output":[]}`},
		"catalog envelope":       {"application/json", `{"items":[]}`},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", tc.contentType)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer server.Close()
			provider := newFixtureCodex(t, server)
			if name == "catalog envelope" {
				_, err := provider.ListModels(context.Background(), fixtureCodexCredential())
				assertCodexFailure(t, err, core.ProviderErrorUpstream, malformed)
				return
			}
			_, err := provider.Invoke(context.Background(), codexResponsesRequest(`{"input":"hi"}`))
			assertCodexFailure(t, err, core.ProviderErrorUpstream, malformed)
		})
	}
}

// Go's client timeout wraps context.DeadlineExceeded, like a caller that
// gave up. Only the caller's own context tells them apart.
func TestCodexClassifiesTransportFailuresByWhoGaveUp(t *testing.T) {
	t.Parallel()
	transport := core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}
	config := func(base string, client *http.Client) CodexConfig {
		return CodexConfig{
			Instructions: "Required instructions", ClientVersion: fixtureCodexClientVersion,
			ResponsesURL: base + "/responses", ModelsURL: base + "/models", Client: client,
		}
	}
	t.Run("connection failure", func(t *testing.T) {
		t.Parallel()
		provider, err := NewCodex(config("http://codex.invalid", &http.Client{Transport: codexRoundTrip(func(*http.Request) (*http.Response, error) {
			return nil, errors.New("fixture connection reset")
		})}))
		if err != nil {
			t.Fatal(err)
		}
		_, err = provider.Invoke(context.Background(), codexResponsesRequest(`{"input":"hi"}`))
		assertCodexFailure(t, err, core.ProviderErrorTransport, transport)
		_, err = provider.ListModels(context.Background(), fixtureCodexCredential())
		assertCodexFailure(t, err, core.ProviderErrorTransport, transport)
	})
	t.Run("client timeout", func(t *testing.T) {
		t.Parallel()
		server, release := blockingCodexServer(nil)
		defer server.Close()
		defer close(release)
		provider, err := NewCodex(config(server.URL, &http.Client{Timeout: 50 * time.Millisecond}))
		if err != nil {
			t.Fatal(err)
		}
		_, err = provider.Invoke(context.Background(), codexResponsesRequest(`{"input":"hi"}`))
		assertCodexFailure(t, err, core.ProviderErrorTransport, transport)
		_, err = provider.ListModels(context.Background(), fixtureCodexCredential())
		assertCodexFailure(t, err, core.ProviderErrorTransport, transport)
	})
	t.Run("caller cancellation", func(t *testing.T) {
		t.Parallel()
		started := make(chan struct{}, 1)
		server, release := blockingCodexServer(started)
		defer server.Close()
		defer close(release)
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			<-started
			cancel()
		}()
		_, err := newFixtureCodex(t, server).Invoke(ctx, codexResponsesRequest(`{"input":"hi"}`))
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want the caller's cancellation", err)
		}
		assertCodexFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{})
	})
}

// blockingCodexServer never answers. Its handlers return when the client
// goes away or release closes; the server only notices a client leaving
// once the handler has read the request body.
func blockingCodexServer(started chan<- struct{}) (*httptest.Server, chan struct{}) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		if started != nil {
			started <- struct{}{}
		}
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	return server, release
}

func TestCodexListsTheModelsTheCredentialCanUse(t *testing.T) {
	t.Parallel()
	catalog := readCodexFixture(t, "catalog-current.json")
	headers := make(chan http.Header, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/models" || r.URL.Query().Get("client_version") != fixtureCodexClientVersion {
			t.Errorf("catalog request = %s %s", r.Method, r.URL)
		}
		headers <- r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(catalog)
	}))
	defer server.Close()
	models, err := newFixtureCodex(t, server).ListModels(context.Background(), fixtureCodexCredential())
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(models)
	assertSameJSON(t, got, readCodexFixture(t, "catalog-current.golden.json"))
	header := <-headers
	if header.Get("Authorization") != "Bearer caller-token" || header.Get("Chatgpt-Account-Id") != "account-fixture" ||
		header.Get("Originator") != "codex_cli_rs" || header.Get("Openai-Beta") != "" {
		t.Fatalf("catalog headers = %v", header)
	}
	// CodexProvider keeps its historical beta header and every row.
	legacy, err := newFixtureCodexProvider(t, server, "Required instructions").ListModels(context.Background(), nil)
	if err != nil || len(legacy) != 6 {
		t.Fatalf("legacy catalog = %d rows, err = %v", len(legacy), err)
	}
	if header := <-headers; header.Get("Openai-Beta") != "responses=experimental" {
		t.Fatalf("legacy catalog headers = %v", header)
	}
}
