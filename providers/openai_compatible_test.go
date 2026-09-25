package providers

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"sync"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

const openAIChatAnswer = `{"id":"chatcmpl_fixture","object":"chat.completion","created":1700000000,"model":"chat-fixture",` +
	`"choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"provider_extra":{"kept":true}}`

// openAICall is one request a synthetic OpenAI-compatible upstream got.
type openAICall struct {
	method, path, authorization, accept, vision, body string
	header                                            http.Header
}

// openAIBackend records every request and answers with reply, which gets
// the request's 1-based arrival number.
type openAIBackend struct {
	mu    sync.Mutex
	calls []openAICall
	reply func(w http.ResponseWriter, r *http.Request, attempt int)
}

func (b *openAIBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	b.mu.Lock()
	b.calls = append(b.calls, openAICall{
		method: r.Method, path: r.URL.Path, authorization: r.Header.Get("Authorization"), accept: r.Header.Get("Accept"),
		vision: r.Header.Get("Copilot-Vision-Request"), body: string(body), header: r.Header.Clone(),
	})
	attempt := len(b.calls)
	b.mu.Unlock()
	b.reply(w, r, attempt)
}

func (b *openAIBackend) take() []openAICall {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

func newOpenAIBackend(t *testing.T, reply func(w http.ResponseWriter, r *http.Request, attempt int)) (*openAIBackend, *httptest.Server) {
	t.Helper()
	backend := &openAIBackend{reply: reply}
	server := httptest.NewServer(backend)
	t.Cleanup(server.Close)
	return backend, server
}

func answerOpenAIChat(w http.ResponseWriter, _ *http.Request, _ int) {
	_, _ = io.WriteString(w, openAIChatAnswer)
}

func newTestOpenAICompatible(t *testing.T, server *httptest.Server, adjust func(*OpenAICompatibleConfig)) *OpenAICompatible {
	t.Helper()
	config := OpenAICompatibleConfig{BaseURL: server.URL + "/v1/", Client: server.Client(), CatalogClient: server.Client()}
	if adjust != nil {
		adjust(&config)
	}
	provider, err := NewOpenAICompatible(config)
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func openAIRequest(surface core.ModelSurface, model, body string, credential *core.Credential) core.Request {
	return core.Request{Surface: surface, Model: model, Body: []byte(body), ContentType: core.ContentTypeJSON, Credential: credential}
}

func openAIRows(rows map[string]core.ModelInfo) func(string) (core.ModelInfo, bool) {
	return func(model string) (core.ModelInfo, bool) {
		row, ok := rows[model]
		return row, ok
	}
}

// The gateway's transport builds the Chat body from its own field list,
// forcing the model and the stream flag, and drops every other field.
func TestOpenAICompatibleShapesChatAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, answerOpenAIChat)
	provider := newTestOpenAICompatible(t, server, nil)
	body := `{"model":"ignored","stream":true,"messages":[{"role":"user","content":"a<b"}],"temperature":0.2,"max_tokens":64,` +
		`"response_format":{"type":"json_object"},"n":2,"user":"fixture-user","force_api_support":false,"_affinity_key":"k",` +
		`"logprobs":null,"parallel_tool_calls":false,"stream_options":{"include_usage":true},"thinking":{"type":"disabled"}}`
	response, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "chat-fixture", body, &core.Credential{APIKey: "fixture-key"}))
	if err != nil {
		t.Fatal(err)
	}
	if string(response.Body) != openAIChatAnswer || response.ContentType != core.ContentTypeJSON {
		t.Fatalf("response = %s", response.Body)
	}
	if got, want := lossPaths(response.Losses), []string{"advisory dropped n", "advisory dropped response_format", "advisory dropped user"}; !slices.Equal(got, want) {
		t.Fatalf("losses = %v, want %v", got, want)
	}
	want := `{"max_tokens":64,"messages":[{"content":"a<b","role":"user"}],"model":"chat-fixture","parallel_tool_calls":false,` +
		`"stream":false,"stream_options":{"include_usage":true},"temperature":0.2,"thinking":{"type":"disabled"}}`
	calls := backend.take()
	if len(calls) != 1 || calls[0].method != http.MethodPost || calls[0].path != "/v1/chat/completions" || calls[0].body != want ||
		calls[0].authorization != "Bearer fixture-key" || calls[0].accept != "" || calls[0].vision != "" ||
		calls[0].header.Get("Content-Type") != core.ContentTypeJSON {
		t.Fatalf("upstream = %+v", calls)
	}
}

// A Responses request a translation.Adapter serves over Chat keeps its
// output limit, unless the request limits its output itself.
func TestOpenAICompatibleSendsTheOutputLimitAsMaxCompletionTokens(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, answerOpenAIChat)
	provider := newTestOpenAICompatible(t, server, nil)
	for body, want := range map[string]string{
		`{"messages":[],"_max_output_tokens":128}`:                 `{"max_completion_tokens":128,"messages":[],"model":"chat-fixture","stream":false}`,
		`{"messages":[],"_max_output_tokens":128,"max_tokens":16}`: `{"max_tokens":16,"messages":[],"model":"chat-fixture","stream":false}`,
		`{}`: `{"messages":[],"model":"chat-fixture","stream":false}`,
	} {
		response, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "chat-fixture", body, nil))
		if calls := backend.take(); err != nil || len(response.Losses) != 0 || len(calls) != 1 || calls[0].body != want || calls[0].authorization != "" {
			t.Fatalf("%s: calls = %+v, losses = %v, err = %v", body, calls, response.Losses, err)
		}
	}
}

func TestOpenAICompatibleForwardsAllFieldsWhenAsked(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, answerOpenAIChat)
	provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) { config.ForwardAllFields = true })
	body := `{"messages":[],"n":2,"user":"fixture-user","seed":7,"logprobs":null,"_gateway":1,"force_api_support":false}`
	response, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "chat-fixture", body, nil))
	want := `{"messages":[],"model":"chat-fixture","n":2,"seed":7,"stream":false,"user":"fixture-user"}`
	if calls := backend.take(); err != nil || len(response.Losses) != 0 || len(calls) != 1 || calls[0].body != want {
		t.Fatalf("calls = %+v, losses = %v, err = %v", calls, response.Losses, err)
	}
}

// Keys are normalized as the gateway normalizes them. Headers configured
// and those a credential carries are sent, and never replace the
// transport's own.
func TestOpenAICompatibleAuthenticatesWithTheCredential(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, answerOpenAIChat)
	provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) {
		config.Headers = map[string]string{"X-Fixture": "static", "Content-Type": "text/plain"}
	})
	for name, test := range map[string]struct {
		credential    *core.Credential
		authorization string
	}{
		"none":          {nil, ""},
		"free":          {&core.Credential{APIKey: "free"}, ""},
		"NONE":          {&core.Credential{APIKey: " NONE "}, ""},
		"bearer prefix": {&core.Credential{APIKey: "  bearer  fixture-key "}, "Bearer fixture-key"},
		"token":         {&core.Credential{Token: "fixture-token"}, "Bearer fixture-token"},
		"public":        {&core.Credential{APIKey: "public"}, "Bearer public"},
		"headers": {&core.Credential{APIKey: "fixture-key", Headers: map[string]string{
			"X-Fixture": "credential", "OpenAI-Organization": "org-fixture", "Authorization": "Bearer other",
		}}, "Bearer fixture-key"},
	} {
		if _, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "m", `{}`, test.credential)); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		calls := backend.take()
		if len(calls) != 1 || calls[0].authorization != test.authorization || calls[0].header.Get("Content-Type") != core.ContentTypeJSON {
			t.Fatalf("%s: upstream = %+v", name, calls)
		}
		wantFixture := "static"
		if name == "headers" {
			wantFixture = "credential"
			if calls[0].header.Get("OpenAI-Organization") != "org-fixture" {
				t.Fatalf("credential headers were not sent: %v", calls[0].header)
			}
		}
		if calls[0].header.Get("X-Fixture") != wantFixture {
			t.Fatalf("%s: X-Fixture = %q", name, calls[0].header.Get("X-Fixture"))
		}
	}
}

func TestOpenAICompatibleMarksImageRequests(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, answerOpenAIChat)
	provider := newTestOpenAICompatible(t, server, nil)
	body := `{"messages":[{"role":"user","content":[{"type":"text","text":"what"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}`
	if _, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "m", body, nil)); err != nil {
		t.Fatal(err)
	}
	if calls := backend.take(); len(calls) != 1 || calls[0].vision != "true" {
		t.Fatalf("upstream = %+v", calls)
	}
}

func TestOpenAICompatibleSurfacesFollowTheCatalogRow(t *testing.T) {
	t.Parallel()
	rows := openAIRows(map[string]core.ModelInfo{
		"responses": {SupportedAPIs: []string{"/responses"}}, "both": {SupportedAPIs: []string{"/chat/completions", "/v1/responses"}},
		"ws": {SupportedAPIs: []string{" WS:/responses "}}, "chat": {SupportedAPIs: []string{"/chat/completions"}},
	})
	chat, responses, messages := core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses, core.ModelSurfaceMessages
	for _, test := range []struct {
		config OpenAICompatibleConfig
		model  string
		native []core.ModelSurface
		chat   bool
	}{
		{OpenAICompatibleConfig{Models: rows}, "responses", []core.ModelSurface{chat, responses}, true},
		{OpenAICompatibleConfig{Models: rows}, "both", []core.ModelSurface{chat, responses}, true},
		{OpenAICompatibleConfig{Models: rows}, "ws", []core.ModelSurface{chat, responses}, true},
		{OpenAICompatibleConfig{Models: rows}, "chat", []core.ModelSurface{chat}, true},
		{OpenAICompatibleConfig{Models: rows}, "unlisted", []core.ModelSurface{chat}, true},
		{OpenAICompatibleConfig{}, "responses", []core.ModelSurface{chat}, true},
		{OpenAICompatibleConfig{RegistryID: " OpenAI "}, "future-model", []core.ModelSurface{chat, responses}, true},
		{OpenAICompatibleConfig{Models: rows, ForceAPISupport: true}, "responses", []core.ModelSurface{chat, responses}, false},
		{OpenAICompatibleConfig{Models: rows, ForceAPISupport: true}, "both", []core.ModelSurface{chat, responses}, true},
	} {
		test.config.BaseURL = "https://upstream.example.test/v1"
		provider, err := NewOpenAICompatible(test.config)
		if err != nil {
			t.Fatal(err)
		}
		native := provider.NativeSurfaces(test.model)
		if !reflect.DeepEqual(native, test.native) || core.PreservesWire(provider, test.model, chat) != test.chat ||
			core.PreservesWire(provider, test.model, responses) != slices.Contains(test.native, responses) ||
			core.PreservesWire(provider, test.model, messages) {
			t.Fatalf("%s %+v: surfaces = %v", test.model, test.config, native)
		}
	}
	for _, base := range []string{"", "upstream.example.test/v1", "/v1"} {
		if _, err := NewOpenAICompatible(OpenAICompatibleConfig{BaseURL: base}); err == nil {
			t.Fatalf("base URL %q was accepted", base)
		}
	}
}
