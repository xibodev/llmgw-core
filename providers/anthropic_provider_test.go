package providers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// anthropicFixtureEvents is the Messages stream the gateway's
// characterization upstream sends for claude-fixture.
const anthropicFixtureEvents = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-fixture\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" from messages\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":4}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

const anthropicFixtureMessage = `{"id":"msg_fixture","type":"message","role":"assistant","model":"claude-fixture","content":[{"type":"text","text":"Hello from messages"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":4}}`

// anthropicRecorded is one request a fixture Anthropic received.
type anthropicRecorded struct {
	method, path string
	header       http.Header
	body         string
}

// anthropicBackend is a synthetic Anthropic API that records every request.
type anthropicBackend struct {
	mu     sync.Mutex
	calls  []anthropicRecorded
	answer func(w http.ResponseWriter, r *http.Request, body []byte)
}

func (b *anthropicBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	b.mu.Lock()
	b.calls = append(b.calls, anthropicRecorded{method: r.Method, path: r.URL.Path, header: r.Header.Clone(), body: string(body)})
	b.mu.Unlock()
	b.answer(w, r, body)
}

// take returns the requests since the last take.
func (b *anthropicBackend) take() []anthropicRecorded {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

// answerAnthropic answers as the gateway's characterization upstream does.
func answerAnthropic(w http.ResponseWriter, r *http.Request, body []byte) {
	switch r.URL.Path {
	case "/v1/models":
		_, _ = io.WriteString(w, `{"data":[{"id":"claude-fixture","type":"model","display_name":"Claude Fixture","created_at":"2025-01-01T00:00:00Z"}],"has_more":false}`)
	case "/v1/messages/count_tokens":
		_, _ = io.WriteString(w, `{"input_tokens":12}`)
	case "/v1/messages":
		if strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", core.ContentTypeEventStream)
			_, _ = io.WriteString(w, anthropicFixtureEvents)
			return
		}
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, anthropicFixtureMessage)
	default:
		http.NotFound(w, r)
	}
}

func newAnthropicBackend(t *testing.T, answer func(http.ResponseWriter, *http.Request, []byte)) (*anthropicBackend, *httptest.Server) {
	t.Helper()
	if answer == nil {
		answer = answerAnthropic
	}
	backend := &anthropicBackend{answer: answer}
	server := httptest.NewServer(backend)
	t.Cleanup(server.Close)
	return backend, server
}

func newTestAnthropic(t *testing.T, server *httptest.Server, adjust ...func(*AnthropicConfig)) *Anthropic {
	t.Helper()
	config := AnthropicConfig{BaseURL: server.URL, Client: server.Client(), CatalogClient: server.Client()}
	for _, change := range adjust {
		change(&config)
	}
	provider, err := NewAnthropic(config)
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func anthropicMessagesRequest(body string, credential *core.Credential) core.Request {
	return core.Request{
		Surface: core.ModelSurfaceMessages, Model: "claude-fixture", ContentType: core.ContentTypeJSON,
		Body: []byte(body), Credential: credential,
	}
}

func anthropicAPIKey(key string) *core.Credential {
	return &core.Credential{APIKey: key, TokenType: core.TokenTypeAPIKey}
}

func drainAnthropic(t *testing.T, stream core.StreamIter) (string, error) {
	t.Helper()
	defer stream.Close()
	var frames strings.Builder
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			return frames.String(), nil
		}
		if err != nil {
			return frames.String(), err
		}
		frames.Write(frame)
	}
}

// Ported from the gateway's TestAnthropicDeclaresOnlyMessagesWireNative.
func TestAnthropicPreservesOnlyMessages(t *testing.T) {
	t.Parallel()
	provider, err := NewAnthropic(AnthropicConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if surfaces := provider.NativeSurfaces("claude-fixture"); !reflect.DeepEqual(surfaces, []core.ModelSurface{core.ModelSurfaceMessages}) {
		t.Fatalf("native surfaces = %v", surfaces)
	}
	if !core.PreservesWire(provider, "claude-fixture", core.ModelSurfaceMessages) {
		t.Fatal("Anthropic Messages was not declared preserved")
	}
	for _, surface := range []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses} {
		if core.PreservesWire(provider, "claude-fixture", surface) {
			t.Fatalf("Anthropic declared %s preserved", surface)
		}
	}
}

// Ported from the gateway's TestAnthropicCredentialHeadersAcrossRequestSurfaces,
// with the kinds a core credential names and a keyless instance.
func TestAnthropicCredentialHeadersAcrossOperations(t *testing.T) {
	t.Parallel()
	for name, check := range map[string]struct {
		credential            *core.Credential
		apiKey, authorization string
	}{
		"api key":            {credential: anthropicAPIKey("fixture-key"), apiKey: "fixture-key"},
		"key without a kind": {credential: &core.Credential{Token: "fixture-key"}, apiKey: "fixture-key"},
		"keyless":            {},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			backend, server := newAnthropicBackend(t, nil)
			provider := newTestAnthropic(t, server)
			ctx := context.Background()
			request := anthropicMessagesRequest(`{"max_tokens":64,"messages":[{"role":"user","content":"Say hello"}]}`, check.credential)
			if _, err := provider.Invoke(ctx, request); err != nil {
				t.Fatal(err)
			}
			stream, err := provider.Stream(ctx, request)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := drainAnthropic(t, stream); err != nil {
				t.Fatal(err)
			}
			if _, err := provider.CountTokens(ctx, core.TokenCountRequest{Request: request}); err != nil {
				t.Fatal(err)
			}
			if _, err := provider.ListModels(ctx, check.credential); err != nil {
				t.Fatal(err)
			}
			calls := backend.take()
			if len(calls) != 4 {
				t.Fatalf("upstream calls = %d, want 4", len(calls))
			}
			for _, call := range calls {
				header := call.header
				if header.Get("x-api-key") != check.apiKey || header.Get("Authorization") != "" ||
					header.Get("anthropic-version") != "2023-06-01" || header.Get("Content-Type") != core.ContentTypeJSON {
					t.Fatalf("%s %s headers = %v", call.method, call.path, header)
				}
			}
		})
	}
}

// Ported from the gateway's TestAnthropicNativeMessagesPreservesOpaquePayloadAndResponse.
// The body is encoded as the gateway encodes it: keys sorted, HTML escaped,
// numbers as written.
func TestAnthropicInvokePassesMessagesThrough(t *testing.T) {
	t.Parallel()
	const answer = `{"id":"msg_native","model":"upstream-model","stop_sequence":"END","content":[{"type":"thinking","thinking":"kept","signature":"sig"},{"type":"unknown","value":1}],"usage":{"input_tokens":9007199254740993,"cache_read_input_tokens":1}}`
	backend, server := newAnthropicBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
		_, _ = io.WriteString(w, answer+"\n")
	})
	provider := newTestAnthropic(t, server)
	request := anthropicMessagesRequest(`{"model":"picker-alias","stream":true,"system":[{"type":"text","text":"client"}],`+
		`"messages":[{"role":"user","content":"hi <b>"}],"thinking":{"type":"enabled","budget_tokens":1024.0},"_llmgw_preamble":"policy"}`,
		anthropicAPIKey("fixture-key"))
	request.Model = "resolved-model"
	response, err := provider.Invoke(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if string(response.Body) != answer || response.ContentType != core.ContentTypeJSON || len(response.Losses) != 0 {
		t.Fatalf("response = %+v, body %s", response, response.Body)
	}
	want := `{"messages":[{"content":"hi \u003cb\u003e","role":"user"}],"model":"resolved-model","stream":false,` +
		`"system":[{"text":"policy","type":"text"},{"text":"client","type":"text"}],"thinking":{"budget_tokens":1024.0,"type":"enabled"}}`
	if calls := backend.take(); len(calls) != 1 || calls[0].method != http.MethodPost || calls[0].path != "/v1/messages" || calls[0].body != want {
		t.Fatalf("upstream = %+v", calls)
	}
}

type anthropicPreambleKey struct{}

// The preamble merges into the system prompt as the gateway merges its
// preamble, from the hook first and else from the body's field, which never
// reaches Anthropic either way.
func TestAnthropicMergesThePreamble(t *testing.T) {
	t.Parallel()
	for name, check := range map[string]struct {
		hook, body, system string
	}{
		"before a string":         {hook: "policy", body: `"system":"client",`, system: `"policy\n\nclient"`},
		"before a block list":     {hook: "policy", body: `"system":[{"type":"text","text":"client"}],`, system: `[{"text":"policy","type":"text"},{"text":"client","type":"text"}]`},
		"without a system prompt": {hook: "policy", system: `"policy"`},
		"in place of an object":   {hook: "policy", body: `"system":{"unexpected":true},`, system: `"policy"`},
		"over the body's field":   {hook: "policy", body: `"system":"client","_llmgw_preamble":"field",`, system: `"policy\n\nclient"`},
		"from the body's field":   {body: `"system":"client","_llmgw_preamble":"field",`, system: `"field\n\nclient"`},
		"none":                    {body: `"system":"client","_llmgw_preamble":7,`, system: `"client"`},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			backend, server := newAnthropicBackend(t, nil)
			provider := newTestAnthropic(t, server, func(config *AnthropicConfig) {
				config.Preamble = func(ctx context.Context) string {
					if ctx.Value(anthropicPreambleKey{}) != "operation" {
						t.Error("the hook did not get the operation's context")
					}
					return check.hook
				}
			})
			ctx := context.WithValue(context.Background(), anthropicPreambleKey{}, "operation")
			request := anthropicMessagesRequest(`{`+check.body+`"messages":[{"role":"user","content":"hi"}]}`, nil)
			if _, err := provider.Invoke(ctx, request); err != nil {
				t.Fatal(err)
			}
			stream, err := provider.Stream(ctx, request)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := drainAnthropic(t, stream); err != nil {
				t.Fatal(err)
			}
			for _, call := range backend.take() {
				var body map[string]json.RawMessage
				if err := json.Unmarshal([]byte(call.body), &body); err != nil {
					t.Fatal(err)
				}
				if _, leaked := body["_llmgw_preamble"]; leaked || string(body["system"]) != check.system {
					t.Fatalf("upstream body = %s", call.body)
				}
			}
		})
	}
}

// Ported from the gateway's TestAnthropicNativeMessagesRejectsStructurallyInvalidSuccessPayloads.
func TestAnthropicInvokeRefusesUnusableAnswers(t *testing.T) {
	t.Parallel()
	for _, answer := range []string{"null", "{}", "[]", `{"content":"text"}`, "not json"} {
		t.Run(answer, func(t *testing.T) {
			t.Parallel()
			_, server := newAnthropicBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
				_, _ = io.WriteString(w, answer)
			})
			_, err := newTestAnthropic(t, server).Invoke(context.Background(), anthropicMessagesRequest(`{"messages":[]}`, nil))
			assertAnthropicFailure(t, err, core.ProviderErrorUpstream, core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true})
		})
	}
}

func assertAnthropicFailure(t *testing.T, err error, class core.ProviderErrorClass, want core.ProviderErrorClassification) *core.ProviderError {
	t.Helper()
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Class != class || failure.Classification != want {
		t.Fatalf("error = %#v, want class %s and %+v", err, class, want)
	}
	return failure
}

// anthropicEvents renders events as a data-only SSE stream.
func anthropicEvents(events ...string) string {
	var stream strings.Builder
	for _, event := range events {
		stream.WriteString("data: " + event + "\n\n")
	}
	return stream.String()
}

// Ported from the gateway's TestAnthropicStreamNormalAndOversizedRecords,
// over the native stream: records pass through byte for byte, and a stream
// fails when it cannot be read or ends before message_stop.
func TestAnthropicStreamRelaysRecordsByteForByte(t *testing.T) {
	t.Parallel()
	unfinished := strings.TrimSuffix(anthropicFixtureEvents, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	overloaded := "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
	for name, check := range map[string]struct {
		upstream, frames string
		abort            bool
		class            core.ProviderErrorClass
		want             core.ProviderErrorClassification
	}{
		"complete":             {upstream: anthropicFixtureEvents, frames: anthropicFixtureEvents},
		"crlf":                 {upstream: "data: {\"type\":\"ping\"}\r\n\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n", frames: "data: {\"type\":\"ping\"}\r\n\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n"},
		"ended early":          {upstream: unfinished, frames: unfinished, class: core.ProviderErrorUpstream, want: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"ended with an error":  {upstream: overloaded, frames: overloaded, class: core.ProviderErrorUpstream, want: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
		"an oversized record":  {upstream: "data: " + strings.Repeat("x", maxStreamRecordWireSize) + "\n\n", class: core.ProviderErrorUpstream},
		"a stream broken off":  {upstream: unfinished, frames: unfinished, abort: true, class: core.ProviderErrorTransport, want: core.ProviderErrorClassification{FailoverEligible: true}},
		"a last record ending": {upstream: "data: {\"type\":\"message_stop\"}", frames: "data: {\"type\":\"message_stop\"}\n\n"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			backend, server := newAnthropicBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
				w.Header().Set("Content-Type", core.ContentTypeEventStream)
				_, _ = io.WriteString(w, check.upstream)
				if check.abort {
					w.(http.Flusher).Flush()
					panic(http.ErrAbortHandler)
				}
			})
			stream, err := newTestAnthropic(t, server).Stream(context.Background(), anthropicMessagesRequest(
				`{"max_tokens":64,"messages":[{"role":"user","content":"Say hello"}]}`, anthropicAPIKey("fixture-key")))
			if err != nil {
				t.Fatal(err)
			}
			frames, err := drainAnthropic(t, stream)
			if frames != check.frames {
				t.Fatalf("frames = %q", frames)
			}
			if check.class == "" && err != nil {
				t.Fatal(err)
			} else if check.class != "" {
				assertAnthropicFailure(t, err, check.class, check.want)
			}
			if calls := backend.take(); len(calls) != 1 || calls[0].body != `{"max_tokens":64,"messages":[{"content":"Say hello","role":"user"}],"model":"claude-fixture","stream":true}` {
				t.Fatalf("upstream = %+v", calls)
			}
		})
	}
}

// Ported from the gateway's TestProviderNon2xxErrorsAreSanitizedAndKeepStatus
// for every operation: a refusal keeps its status and Retry-After, and its
// text reaches only the redacted diagnostic.
func TestAnthropicRefusalsKeepStatusAndRetryAfter(t *testing.T) {
	t.Parallel()
	gatewayToken, email := syntheticGatewayToken(), "provider-owner@example.test"
	_, server := newAnthropicBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprintf(w, `{"type":"error","error":{"type":"rate_limit_error","message":"account %s used %s"}}`, email, gatewayToken)
	})
	provider := newTestAnthropic(t, server)
	ctx := context.Background()
	request := anthropicMessagesRequest(`{"messages":[]}`, anthropicAPIKey("fixture-key"))
	for name, operation := range map[string]func() error{
		"anthropic": func() error { _, err := provider.Invoke(ctx, request); return err },
		"anthropic stream": func() error {
			_, err := provider.Stream(ctx, request)
			return err
		},
		"anthropic token count": func() error {
			_, err := provider.CountTokens(ctx, core.TokenCountRequest{Request: request})
			return err
		},
	} {
		label := strings.TrimSuffix(strings.TrimSuffix(name, " stream"), " setup token")
		failure := assertAnthropicFailure(t, operation(), core.ProviderErrorRateLimited, core.ProviderErrorClassification{
			StatusCode: http.StatusTooManyRequests, Retryable: true, FailoverEligible: true, CircuitFailure: true, RetryAfter: 7 * time.Second,
		})
		var cause *InvocationError
		if failure.Message != label+" returned HTTP 429 (type=rate_limit_error)" || !errors.As(failure, &cause) ||
			!strings.HasPrefix(cause.Msg, label+": upstream returned 429: account "+redactedDiagnostic) {
			t.Fatalf("%s: message = %q, cause = %#v", name, failure.Message, cause)
		}
		for _, text := range []string{failure.Message, cause.Msg} {
			if strings.Contains(text, gatewayToken) || strings.Contains(text, email) {
				t.Fatalf("%s: error exposed diagnostic data: %q", name, text)
			}
		}
	}
}

// A request that gets no answer may be repeated, unless the caller gave up.
func TestAnthropicTransportFailures(t *testing.T) {
	t.Parallel()
	_, server := newAnthropicBackend(t, nil)
	provider := newTestAnthropic(t, server)
	server.Close()
	request := anthropicMessagesRequest(`{"messages":[]}`, nil)
	_, err := provider.Invoke(context.Background(), request)
	assertAnthropicFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = provider.Stream(ctx, request)
	assertAnthropicFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{})
}

// What Anthropic cannot serve is refused before anything is sent, and no
// refusal quotes the credential.
func TestAnthropicRefusesBeforeSending(t *testing.T) {
	t.Parallel()
	backend, server := newAnthropicBackend(t, nil)
	provider := newTestAnthropic(t, server)
	ctx := context.Background()
	chat := anthropicMessagesRequest(`{"messages":[]}`, nil)
	chat.Surface = core.ModelSurfaceChatCompletions
	var surface *core.SurfaceError
	if _, err := provider.Invoke(ctx, chat); !errors.As(err, &surface) || !core.ClassifyError(err).FailoverEligible {
		t.Fatalf("chat invoke = %v", err)
	}
	if _, err := provider.Stream(ctx, chat); !errors.As(err, &surface) {
		t.Fatalf("chat stream = %v", err)
	}
	if _, err := provider.CountTokens(ctx, core.TokenCountRequest{Request: chat}); !errors.Is(err, core.ErrTokenCountUnsupported) {
		t.Fatalf("chat token count = %v", err)
	}
	for _, body := range []string{`[]`, `{"messages":[]} {}`, `null`} {
		_, err := provider.Invoke(ctx, anthropicMessagesRequest(body, nil))
		assertAnthropicFailure(t, err, core.ProviderErrorInvalidRequest, core.ProviderErrorClassification{})
	}
	text := anthropicMessagesRequest(`{"messages":[]}`, nil)
	text.ContentType = "text/plain"
	_, err := provider.Invoke(ctx, text)
	assertAnthropicFailure(t, err, core.ProviderErrorInvalidRequest, core.ProviderErrorClassification{})
	for name, credential := range map[string]*core.Credential{
		"service account": {Token: `{"type":"service_account","private_key":"fixture-secret"}`, TokenType: core.TokenTypeGCPServiceAccount},
		"oauth token":     {Token: "fixture-secret", TokenType: "Bearer"},
	} {
		_, invoked := provider.Invoke(ctx, anthropicMessagesRequest(`{"messages":[]}`, credential))
		_, listed := provider.ListModels(ctx, credential)
		for operation, err := range map[string]error{"invoke": invoked, "models": listed} {
			failure := assertAnthropicFailure(t, err, core.ProviderErrorConfiguration, core.ProviderErrorClassification{FailoverEligible: true})
			if strings.Contains(failure.Error(), "fixture-secret") || strings.Contains(fmt.Sprint(failure.Cause), "fixture-secret") {
				t.Fatalf("%s %s: refusal quoted the credential: %v", name, operation, failure)
			}
		}
	}
	if calls := backend.take(); len(calls) != 0 {
		t.Fatalf("refused requests reached Anthropic: %+v", calls)
	}
}

func TestNewAnthropicValidatesTheBaseURL(t *testing.T) {
	t.Parallel()
	for configured, want := range map[string]string{
		"":                                 "https://api.anthropic.com",
		" https://api.anthropic.com/ ":     "https://api.anthropic.com",
		"http://127.0.0.1:8080/proxy/n//":  "http://127.0.0.1:8080/proxy/n",
		"https://gateway.example.test/n":   "https://gateway.example.test/n",
		"api.anthropic.com":                "",
		"ftp://api.anthropic.com":          "",
		"https://api.anthropic.com/?key=1": "",
		"https://api.anthropic.com/#part":  "",
		"https://":                         "",
	} {
		provider, err := NewAnthropic(AnthropicConfig{BaseURL: configured})
		switch {
		case want == "" && err == nil:
			t.Fatalf("NewAnthropic(%q) accepted the base URL", configured)
		case want == "":
			assertAnthropicFailure(t, err, core.ProviderErrorConfiguration, core.ProviderErrorClassification{FailoverEligible: true})
		case err != nil || provider.baseURL != want:
			t.Fatalf("NewAnthropic(%q) = %v, %v; want base %q", configured, provider, err, want)
		}
	}
}

// Ported from the gateway's token count path: the body with its model set
// and no preamble, the protocol headers the client chose when their values
// are clean, and nothing else the client sent.
func TestAnthropicCountsTokens(t *testing.T) {
	t.Parallel()
	backend, server := newAnthropicBackend(t, nil)
	provider := newTestAnthropic(t, server, func(config *AnthropicConfig) {
		config.Preamble = func(context.Context) string { return "policy" }
	})
	for name, check := range map[string]struct {
		header  http.Header
		version string
		beta    []string
	}{
		"clean": {
			header: http.Header{
				"Anthropic-Version": {" 2024-01-01 "}, "Anthropic-Beta": {"tokens-2024", "bad\x01value", "  spaced  "},
				"X-Api-Key": {"client-key"},
			},
			version: "2024-01-01", beta: []string{"tokens-2024", "spaced"},
		},
		"unclean version": {header: http.Header{"Anthropic-Version": {"2024\x7f"}}, version: "2023-06-01"},
		"none":            {version: "2023-06-01"},
	} {
		request := core.TokenCountRequest{Header: check.header, Request: anthropicMessagesRequest(
			`{"model":"alias","system":"client","messages":[{"role":"user","content":"hi"}]}`, anthropicAPIKey("test-key"))}
		count, err := core.CountTokens(context.Background(), provider, request)
		if err != nil || count.InputTokens != 12 {
			t.Fatalf("%s: count = %+v, err = %v", name, count, err)
		}
		calls := backend.take()
		if len(calls) != 1 || calls[0].path != "/v1/messages/count_tokens" ||
			calls[0].body != `{"messages":[{"content":"hi","role":"user"}],"model":"claude-fixture","system":"client"}` {
			t.Fatalf("%s: upstream = %+v", name, calls)
		}
		header := calls[0].header
		if header.Get("anthropic-version") != check.version || !reflect.DeepEqual(header.Values("anthropic-beta"), check.beta) ||
			header.Get("x-api-key") != "test-key" || header.Get("Authorization") != "" {
			t.Fatalf("%s: headers = %v", name, header)
		}
	}
}

// Ported from the gateway's invalid token count handling: only decimal
// digits a count can hold are a count.
func TestAnthropicRefusesAnInvalidTokenCount(t *testing.T) {
	t.Parallel()
	for _, answer := range []string{
		`{"input_tokens":-1}`, `{"input_tokens":1.5}`, `{"input_tokens":1e3}`, `{"input_tokens":"x"}`,
		`{"input_tokens":99999999999999999999}`, `{}`, `null`, `[1]`, `not json`,
	} {
		t.Run(answer, func(t *testing.T) {
			t.Parallel()
			_, server := newAnthropicBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
				_, _ = io.WriteString(w, answer)
			})
			_, err := newTestAnthropic(t, server).CountTokens(context.Background(), core.TokenCountRequest{
				Request: anthropicMessagesRequest(`{"messages":[]}`, nil),
			})
			failure := assertAnthropicFailure(t, err, core.ProviderErrorUpstream, core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true})
			if failure.Message != "anthropic returned an invalid token count" {
				t.Fatalf("message = %q", failure.Message)
			}
		})
	}
}
