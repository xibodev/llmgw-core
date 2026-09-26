package providers

import (
	"context"
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

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

const azureFixtureCompletion = `{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-5.6-sol-2026-07-09","prompt_filter_results":[],"service_tier":"default",` +
	`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],"usage":{"total_tokens":2,"latency_checkpoint":{"ttft_ms":1}}}`

const azureFixtureStream = "data: {\"choices\":[],\"prompt_filter_results\":[]}\n\n" +
	"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n" +
	"data: [DONE]\n\n"

// azureRecorded is one request a fixture Azure resource received.
type azureRecorded struct {
	method, path, query string
	header              http.Header
	body                string
}

// azureBackend is a synthetic Azure OpenAI resource that records every
// request.
type azureBackend struct {
	mu     sync.Mutex
	calls  []azureRecorded
	answer func(w http.ResponseWriter, r *http.Request, body []byte)
}

func (b *azureBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	b.mu.Lock()
	b.calls = append(b.calls, azureRecorded{method: r.Method, path: r.URL.Path, query: r.URL.RawQuery, header: r.Header.Clone(), body: string(body)})
	b.mu.Unlock()
	b.answer(w, r, body)
}

func (b *azureBackend) take() []azureRecorded {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

// answerAzure answers a completion, a stream, or one page of deployments.
func answerAzure(w http.ResponseWriter, r *http.Request, body []byte) {
	w.Header().Set("Content-Type", core.ContentTypeJSON)
	switch {
	case r.URL.Path == "/openai/deployments":
		_, _ = io.WriteString(w, `{"data":[{"id":"gpt-5.6-sol","model":"gpt-5.6","status":"succeeded","object":"deployment"}],"object":"list"}`)
	case r.URL.Path == "/openai/v1/chat/completions" && strings.Contains(string(body), `"stream":true`):
		w.Header().Set("Content-Type", core.ContentTypeEventStream)
		_, _ = io.WriteString(w, azureFixtureStream)
	case r.URL.Path == "/openai/v1/chat/completions":
		_, _ = io.WriteString(w, azureFixtureCompletion)
	default:
		http.NotFound(w, r)
	}
}

func newAzureBackend(t *testing.T, answer func(http.ResponseWriter, *http.Request, []byte)) (*azureBackend, *httptest.Server) {
	t.Helper()
	if answer == nil {
		answer = answerAzure
	}
	backend := &azureBackend{answer: answer}
	server := httptest.NewServer(backend)
	t.Cleanup(server.Close)
	return backend, server
}

// newTestAzure points a provider at the resource's origin, the value the
// Azure portal shows.
func newTestAzure(t *testing.T, server *httptest.Server) *AzureOpenAI {
	t.Helper()
	provider, err := NewAzureOpenAI(AzureOpenAIConfig{BaseURL: server.URL, Client: server.Client(), CatalogClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func azureKey(key string) *core.Credential {
	return &core.Credential{APIKey: key, TokenType: core.TokenTypeAPIKey}
}

func azureChatRequest(body string, credential *core.Credential) core.Request {
	return core.Request{
		Surface: core.ModelSurfaceChatCompletions, Model: "gpt-5.6-sol", ContentType: core.ContentTypeJSON,
		Body: []byte(body), Credential: credential,
	}
}

func assertAzureFailure(t *testing.T, err error, class core.ProviderErrorClass, want core.ProviderErrorClassification) *core.ProviderError {
	t.Helper()
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Class != class || failure.Classification != want {
		t.Fatalf("error = %#v, want class %s and %+v", err, class, want)
	}
	return failure
}

// The gateway labels Azure Chat translated: its provider never declared
// wire preservation. Azure OpenAI declares nothing either, so a label read
// from core.PreservesWire stays translated for every surface.
func TestAzureServesChatWithoutPreservingIt(t *testing.T) {
	t.Parallel()
	_, server := newAzureBackend(t, nil)
	provider := newTestAzure(t, server)
	if surfaces := provider.NativeSurfaces("gpt-5.6-sol"); !reflect.DeepEqual(surfaces, []core.ModelSurface{core.ModelSurfaceChatCompletions}) {
		t.Fatalf("native surfaces = %v", surfaces)
	}
	if _, declares := any(provider).(core.WirePreserver); declares {
		t.Fatal("Azure OpenAI declares wire preservation")
	}
	for _, surface := range []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceMessages, core.ModelSurfaceResponses} {
		if core.PreservesWire(provider, "gpt-5.6-sol", surface) {
			t.Fatalf("%s reads as preserved", surface)
		}
	}
}

// Ported from the gateway's TestAzureCompletionsPostToTheOpenAIV1Route,
// TestAzureAuthenticatesWithApiKeyHeader and
// TestAzureDoesNotRoundTripTheResponseModelID: a portal origin reaches the
// /openai/v1 route with the api-key and no api-version, and the answer is
// returned as Azure sent it, model and Azure-only fields included.
func TestAzureCompletionsPostToTheOpenAIV1Route(t *testing.T) {
	t.Parallel()
	backend, server := newAzureBackend(t, nil)
	response, err := newTestAzure(t, server).Invoke(context.Background(), azureChatRequest(`{"messages":[{"role":"user","content":"hi"}]}`, azureKey("fixture-key")))
	if err != nil {
		t.Fatal(err)
	}
	if string(response.Body) != azureFixtureCompletion || response.ContentType != core.ContentTypeJSON || len(response.Losses) != 0 {
		t.Fatalf("response = %+v, body %s", response, response.Body)
	}
	calls := backend.take()
	if len(calls) != 1 || calls[0].method != http.MethodPost || calls[0].path != "/openai/v1/chat/completions" || calls[0].query != "" {
		t.Fatalf("upstream = %+v", calls)
	}
	header := calls[0].header
	if header.Get("api-key") != "fixture-key" || header.Get("Authorization") != "" || header.Get("Content-Type") != core.ContentTypeJSON {
		t.Fatalf("headers = %v", header)
	}
}

// A Chat body is rebuilt as the gateway's transport builds it, byte for
// byte: the deployment as the model, the forwarded fields, keys sorted and
// HTML escaped, numbers as written. Other fields are reported as dropped.
func TestAzureShapesTheChatBodyAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	for name, check := range map[string]struct {
		body, upstream string
		dropped        []string
	}{
		"forwarded fields": {
			body: `{"model":"alias","stream":true,"messages":[{"role":"user","content":"hi <b>"}],"temperature":0.20,"max_tokens":64,` +
				`"metadata":{"trace":"t"},"response_format":{"type":"json_object"},"user":"u","_affinity_key":"a","tool_choice":null}`,
			upstream: `{"max_tokens":64,"messages":[{"content":"hi \u003cb\u003e","role":"user"}],"metadata":{"trace":"t"},"model":"gpt-5.6-sol","stream":false,"temperature":0.20}`,
			dropped:  []string{"response_format", "user"},
		},
		"a translated output limit": {
			body:     `{"messages":[],"_max_output_tokens":100}`,
			upstream: `{"max_completion_tokens":100,"messages":[],"model":"gpt-5.6-sol","stream":false}`,
		},
		"an output limit of its own": {
			body:     `{"messages":[],"_max_output_tokens":100,"max_tokens":8}`,
			upstream: `{"max_tokens":8,"messages":[],"model":"gpt-5.6-sol","stream":false}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			backend, server := newAzureBackend(t, nil)
			response, err := newTestAzure(t, server).Invoke(context.Background(), azureChatRequest(check.body, azureKey("fixture-key")))
			if err != nil {
				t.Fatal(err)
			}
			if calls := backend.take(); len(calls) != 1 || calls[0].body != check.upstream {
				t.Fatalf("upstream = %+v", calls)
			}
			var dropped []string
			for _, loss := range response.Losses {
				if loss.Class != translate.LossDropped || loss.Severity != translate.LossAdvisory {
					t.Fatalf("loss = %+v", loss)
				}
				dropped = append(dropped, loss.Path)
			}
			if !reflect.DeepEqual(dropped, check.dropped) {
				t.Fatalf("dropped = %v, want %v", dropped, check.dropped)
			}
		})
	}
}

// Ported from the gateway's Chat answer checks: an answer that is not a
// completion with an object as its first choice cannot be used.
func TestAzureInvokeRefusesUnusableAnswers(t *testing.T) {
	t.Parallel()
	for _, answer := range []string{"null", "{}", "[]", `{"choices":[]}`, `{"choices":["text"]}`, `{"choices":{}}`, "not json"} {
		t.Run(answer, func(t *testing.T) {
			t.Parallel()
			_, server := newAzureBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) { _, _ = io.WriteString(w, answer) })
			_, err := newTestAzure(t, server).Invoke(context.Background(), azureChatRequest(`{"messages":[]}`, azureKey("fixture-key")))
			assertAzureFailure(t, err, core.ProviderErrorUpstream, core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true})
		})
	}
}

func TestAzureStreamRelaysRecordsAndReportsLosses(t *testing.T) {
	t.Parallel()
	backend, server := newAzureBackend(t, nil)
	stream, err := newTestAzure(t, server).Stream(context.Background(), azureChatRequest(`{"messages":[{"role":"user","content":"hi"}],"user":"u"}`, azureKey("fixture-key")))
	if err != nil {
		t.Fatal(err)
	}
	losses := core.StreamLosses(stream)
	var frames strings.Builder
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		frames.Write(frame)
	}
	_ = stream.Close()
	if frames.String() != azureFixtureStream || len(losses) != 1 || losses[0].Path != "user" {
		t.Fatalf("frames = %q, losses = %+v", frames.String(), losses)
	}
	if calls := backend.take(); len(calls) != 1 || calls[0].body != `{"messages":[{"content":"hi","role":"user"}],"model":"gpt-5.6-sol","stream":true}` {
		t.Fatalf("upstream = %+v", calls)
	}
}

// Ported from the gateway's TestProviderNon2xxErrorsAreSanitizedAndKeepStatus
// for Azure: a refusal keeps its status, and now its Retry-After, and the
// upstream's text reaches only the redacted diagnostic.
func TestAzureRefusalsKeepStatusAndRetryAfter(t *testing.T) {
	t.Parallel()
	gatewayToken, email := syntheticGatewayToken(), "provider-owner@example.test"
	_, server := newAzureBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprintf(w, `{"error":{"code":"429","message":"account %s used %s"}}`, email, gatewayToken)
	})
	provider := newTestAzure(t, server)
	request := azureChatRequest(`{"messages":[]}`, azureKey("fixture-key"))
	_, invoked := provider.Invoke(context.Background(), request)
	_, streamed := provider.Stream(context.Background(), request)
	for _, err := range []error{invoked, streamed} {
		failure := assertAzureFailure(t, err, core.ProviderErrorRateLimited, core.ProviderErrorClassification{
			StatusCode: http.StatusTooManyRequests, Retryable: true, FailoverEligible: true, CircuitFailure: true, RetryAfter: 7 * time.Second,
		})
		var cause *InvocationError
		if failure.Message != "azure_openai returned HTTP 429 (code=429)" || !errors.As(failure, &cause) ||
			!strings.HasPrefix(cause.Msg, "azure_openai: upstream returned 429: account "+redactedDiagnostic) {
			t.Fatalf("message = %q, cause = %#v", failure.Message, cause)
		}
		for _, text := range []string{failure.Message, cause.Msg} {
			if strings.Contains(text, gatewayToken) || strings.Contains(text, email) {
				t.Fatalf("error exposed diagnostic data: %q", text)
			}
		}
	}
}

// What Azure OpenAI cannot serve is refused before anything is sent: another
// surface, a request without an API key, a credential of another kind, and
// a body that is not a Chat request.
func TestAzureRefusesBeforeSending(t *testing.T) {
	t.Parallel()
	backend, server := newAzureBackend(t, nil)
	provider := newTestAzure(t, server)
	ctx := context.Background()
	messages := azureChatRequest(`{"messages":[]}`, azureKey("fixture-key"))
	messages.Surface = core.ModelSurfaceMessages
	var surface *core.SurfaceError
	if _, err := provider.Invoke(ctx, messages); !errors.As(err, &surface) {
		t.Fatalf("messages invoke = %v", err)
	}
	if _, err := provider.Stream(ctx, messages); !errors.As(err, &surface) {
		t.Fatalf("messages stream = %v", err)
	}
	for name, credential := range map[string]*core.Credential{
		"none":            nil,
		"blank key":       azureKey("  "),
		"service account": {Token: `{"private_key":"fixture-secret"}`, TokenType: core.TokenTypeGCPServiceAccount},
		"oauth token":     {Token: "fixture-secret", TokenType: "Bearer"},
	} {
		_, invoked := provider.Invoke(ctx, azureChatRequest(`{"messages":[]}`, credential))
		_, listed := provider.ListModels(ctx, credential)
		for _, err := range []error{invoked, listed} {
			failure := assertAzureFailure(t, err, core.ProviderErrorConfiguration, core.ProviderErrorClassification{FailoverEligible: true})
			if strings.Contains(failure.Error(), "fixture-secret") {
				t.Fatalf("%s: refusal quoted the credential: %v", name, failure)
			}
		}
	}
	for _, body := range []string{`[]`, `{"messages":[]} {}`, `{"messages":"hi"}`, `{"messages":["hi"]}`, `{}`} {
		_, err := provider.Invoke(ctx, azureChatRequest(body, azureKey("fixture-key")))
		assertAzureFailure(t, err, core.ProviderErrorInvalidRequest, core.ProviderErrorClassification{})
	}
	if calls := backend.take(); len(calls) != 0 {
		t.Fatalf("refused requests reached Azure: %+v", calls)
	}
}

// A request that gets no answer may be repeated, unless the caller gave up.
func TestAzureTransportFailures(t *testing.T) {
	t.Parallel()
	_, server := newAzureBackend(t, nil)
	provider := newTestAzure(t, server)
	server.Close()
	request := azureChatRequest(`{"messages":[]}`, azureKey("fixture-key"))
	_, err := provider.Invoke(context.Background(), request)
	assertAzureFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = provider.Stream(ctx, request)
	assertAzureFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{})
}

// The api-key never follows a redirect off the resource, for inference or
// the catalog, while a redirect on the resource is followed.
func TestAzureDoesNotFollowARedirectOffTheResource(t *testing.T) {
	t.Parallel()
	foreign, foreignServer := newAzureBackend(t, nil)
	backend, server := newAzureBackend(t, func(w http.ResponseWriter, r *http.Request, body []byte) {
		switch {
		case r.URL.Query().Get("moved") == "here":
			answerAzure(w, r, body)
		case strings.Contains(string(body), `"model":"local"`):
			http.Redirect(w, r, r.URL.Path+"?moved=here", http.StatusTemporaryRedirect)
		default:
			http.Redirect(w, r, foreignServer.URL+r.URL.RequestURI(), http.StatusTemporaryRedirect)
		}
	})
	provider := newTestAzure(t, server)
	ctx := context.Background()
	_, err := provider.Invoke(ctx, azureChatRequest(`{"messages":[]}`, azureKey("fixture-key")))
	assertAzureFailure(t, err, core.ProviderErrorUpstream, core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true})
	_, err = provider.ListModels(ctx, azureKey("fixture-key"))
	if catalog := catalogCode(t, err); catalog.Code != CatalogCodeNotDiscoverable {
		t.Fatalf("catalog code = %q", catalog.Code)
	}
	if calls := foreign.take(); len(calls) != 0 {
		t.Fatalf("the api-key followed a redirect off the resource: %+v", calls)
	}
	local := azureChatRequest(`{"messages":[]}`, azureKey("fixture-key"))
	local.Model = "local"
	if _, err := provider.Invoke(ctx, local); err != nil {
		t.Fatalf("a redirect on the resource was not followed: %v", err)
	}
	if calls := backend.take(); len(calls) != 4 || calls[3].query != "moved=here" || calls[3].header.Get("api-key") != "fixture-key" {
		t.Fatalf("resource calls = %+v", calls)
	}
}
