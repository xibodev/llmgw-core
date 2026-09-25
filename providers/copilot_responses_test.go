package providers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	copilotauth "github.com/xibodev/llm-provider-auth/copilot"
	core "github.com/xibodev/llmgw-core"
)

const (
	copilotResponsesReply = `{"id":"resp_fixture","object":"response","status":"completed","model":"copilot-fixture-codex",` +
		`"output":[{"type":"message","id":"msg_fixture","role":"assistant","content":[{"type":"output_text","text":"Hello from responses"}]}],` +
		`"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}`
	copilotResponsesReplyEvents = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_fixture\",\"output\":[]}}\n\n" +
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":" + copilotResponsesReply + "}\n\n"
)

// answerCopilot serves the catalog fixture, Chat and Responses.
func answerCopilot(t *testing.T) func(http.ResponseWriter, *http.Request, []byte) {
	catalog := readCopilotFixture(t, "catalog-current.json")
	return func(w http.ResponseWriter, r *http.Request, body []byte) {
		switch {
		case r.URL.Path == "/api/models":
			_, _ = w.Write(catalog)
		case r.URL.Path == "/api/responses" && r.Header.Get("Accept") == core.ContentTypeEventStream:
			w.Header().Set("Content-Type", core.ContentTypeEventStream)
			_, _ = io.WriteString(w, copilotResponsesReplyEvents)
		case r.URL.Path == "/api/responses":
			_, _ = io.WriteString(w, copilotResponsesReply)
		default:
			answerCopilotChat(w, r, body)
		}
	}
}

// listedCopilot returns a Copilot that has listed the catalog fixture.
func listedCopilot(t *testing.T, backend *copilotBackend, adjust ...func(*CopilotConfig, *copilotauth.Config)) *Copilot {
	t.Helper()
	provider := newFixtureCopilot(t, backend, adjust...)
	if _, err := provider.ListModels(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	backend.take()
	return provider
}

func TestCopilotListsTheCatalogAsTheGatewayDerivesIt(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilot(t))
	backend.update(func(b *copilotBackend) { b.expiresAt = copilotDiscoveredAt.Add(30 * time.Minute).Unix() })
	provider := newFixtureCopilot(t, backend, func(config *CopilotConfig, _ *copilotauth.Config) {
		config.Now = func() time.Time { return copilotDiscoveredAt }
	})
	if surfaces := provider.NativeSurfaces("copilot-fixture-reasoning"); !slices.Equal(surfaces, []core.ModelSurface{core.ModelSurfaceChatCompletions}) {
		t.Fatalf("surfaces before listing = %v, want Chat alone", surfaces)
	}
	models, err := provider.ListModels(context.Background(), copilotCredential("oauth-a"))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(models)
	assertSameJSON(t, got, readCopilotFixture(t, "catalog-current.golden.json"))
	_, calls := backend.take()
	if len(calls) != 1 || calls[0].method != http.MethodGet || calls[0].path != "/api/models" || calls[0].body != "" {
		t.Fatalf("catalog upstream = %+v", calls)
	}
	assertCopilotHeaders(t, calls[0].header, "session-1", "application/json", false)
	chat, both := []core.ModelSurface{core.ModelSurfaceChatCompletions}, []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses}
	for model, want := range map[string][]core.ModelSurface{
		"copilot-fixture-chat": chat, "copilot-fixture-reasoning": both, "copilot-fixture-codex": both,
		"copilot-fixture-claude": chat, "copilot-fixture-embedding": chat, "unlisted": chat,
	} {
		if surfaces := provider.NativeSurfaces(model); !slices.Equal(surfaces, want) {
			t.Errorf("%s surfaces = %v, want %v", model, surfaces, want)
		}
	}
}

func TestCopilotCatalogFailuresAreProviderErrors(t *testing.T) {
	t.Parallel()
	for name, testCase := range map[string]struct {
		status int
		body   string
		class  core.ProviderErrorClass
		want   core.ProviderErrorClassification
	}{
		"server error": {503, ``, core.ProviderErrorUpstream, core.ProviderErrorClassification{StatusCode: 503, Retryable: true, FailoverEligible: true, CircuitFailure: true}},
		"malformed":    {200, `{"data":[{"id":7}]}`, core.ProviderErrorUpstream, core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}},
	} {
		t.Run(name, func(t *testing.T) {
			backend := newCopilotBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
				w.WriteHeader(testCase.status)
				_, _ = io.WriteString(w, testCase.body)
			})
			_, err := newFixtureCopilot(t, backend).ListModels(context.Background(), nil)
			assertCopilotFailure(t, err, testCase.class, testCase.want)
		})
	}
}

func TestCopilotServesResponsesNativelyOnlyForListedModels(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilot(t))
	unlisted := newFixtureCopilot(t, backend)
	request := copilotRequest(core.ModelSurfaceResponses, "copilot-fixture-reasoning",
		`{"model":"x","input":[{"role":"user","content":[{"type":"input_image","image_url":"data:,"}]}],"stream":true,"force_api_support":true,"temperature":0.20,"store":false}`,
		copilotCredential("oauth-a"))
	var surface *core.SurfaceError
	if _, err := unlisted.Invoke(context.Background(), request); !errors.As(err, &surface) {
		t.Fatalf("unlisted Responses: err = %v, want a *core.SurfaceError", err)
	}
	if _, err := unlisted.Stream(context.Background(), request); !errors.As(err, &surface) {
		t.Fatalf("unlisted Responses stream: err = %v, want a *core.SurfaceError", err)
	}
	provider := listedCopilot(t, backend)
	response, err := provider.Invoke(context.Background(), request)
	if err != nil || string(response.Body) != copilotResponsesReply || response.Losses != nil {
		t.Fatalf("response = %s %v, err = %v", response.Body, response.Losses, err)
	}
	stream, err := provider.Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if frames := drainCopilot(t, stream); len(frames) != 2 || frames[0]+frames[1] != copilotResponsesReplyEvents {
		t.Fatalf("frames = %q", frames)
	}
	_, calls := backend.take()
	input := `{"input":[{"content":[{"image_url":"data:,","type":"input_image"}],"role":"user"}],"model":"copilot-fixture-reasoning","store":false,`
	if len(calls) != 2 || calls[0].body != input+`"stream":false,"temperature":0.2}` || calls[1].body != input+`"stream":true,"temperature":0.2}` {
		t.Fatalf("upstream = %+v", calls)
	}
	assertCopilotHeaders(t, calls[0].header, "session-2", "application/json", true)
	assertCopilotHeaders(t, calls[1].header, "session-2", core.ContentTypeEventStream, true)
	backend.update(func(b *copilotBackend) {
		b.answer = func(w http.ResponseWriter, _ *http.Request, _ []byte) { w.WriteHeader(http.StatusNotFound) }
	})
	if _, err := provider.Invoke(context.Background(), request); !errors.As(err, &surface) {
		t.Fatalf("404: err = %v, want a *core.SurfaceError", err)
	}
	if _, err := provider.Stream(context.Background(), request); !errors.As(err, &surface) {
		t.Fatalf("404 stream: err = %v, want a *core.SurfaceError", err)
	}
}

// copilotAdaptedChat is a Chat request to a Responses-only model, and
// copilotAdaptedUpstream the Responses request the gateway's adaptation,
// ChatToResponsesWithReport over its Chat facade's fields, sends for it.
const (
	copilotAdaptedChat = `{"model":"x","stream":true,"messages":[{"role":"system","content":"Be <brief>."},` +
		`{"role":"user","content":[{"type":"text","text":"Say hello"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}],` +
		`"max_tokens":64,"temperature":0.2,"stream_options":{"include_usage":true},"n":1}`
	copilotAdaptedUpstream = `{"input":[{"content":[{"text":"Say hello","type":"input_text"},{"image_url":"data:image/png;base64,AA==","type":"input_image"}],"role":"user"}],` +
		`"instructions":"Be <brief>.","max_output_tokens":64,"model":"copilot-fixture-codex","temperature":0.2}`
)

func lossPaths(losses []core.Loss) []string {
	paths := make([]string, len(losses))
	for index, loss := range losses {
		paths[index] = string(loss.Severity) + " " + string(loss.Class) + " " + loss.Path
	}
	return paths
}

// With adaptation on, as it is in the gateway, Chat for a model that lists
// only Responses is served over Responses, streamed from one complete
// answer; force_api_support or the configuration turn it off.
func TestCopilotServesChatOverResponsesForResponsesOnlyModels(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilot(t))
	provider := listedCopilot(t, backend)
	disabled := listedCopilot(t, backend, func(config *CopilotConfig, _ *copilotauth.Config) { config.DisableAdaptation = true })
	chat := copilotRequest(core.ModelSurfaceChatCompletions, "copilot-fixture-codex", copilotAdaptedChat, copilotCredential("oauth-a"))
	response, err := provider.Invoke(context.Background(), chat)
	if err != nil {
		t.Fatal(err)
	}
	var answer struct {
		Object  string         `json:"object"`
		Forced  map[string]any `json:"forced_support"`
		Choices []struct {
			Message struct{ Content string } `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(response.Body, &answer) != nil || answer.Object != "chat.completion" || answer.Forced != nil ||
		len(answer.Choices) != 1 || answer.Choices[0].Message.Content != "Hello from responses" {
		t.Fatalf("answer = %s", response.Body)
	}
	if want := []string{"advisory renamed max_tokens", "advisory dropped n", "advisory dropped stream_options"}; !reflect.DeepEqual(lossPaths(response.Losses), want) {
		t.Fatalf("losses = %v, want %v", lossPaths(response.Losses), want)
	}
	stream, err := provider.Stream(context.Background(), chat)
	if err != nil {
		t.Fatal(err)
	}
	frames := drainCopilot(t, stream)
	if len(frames) != 4 || frames[3] != "data: [DONE]\n\n" || !slices.Contains(lossPaths(core.StreamLosses(stream)), "advisory approximated stream") {
		t.Fatalf("frames = %q losses = %v", frames, lossPaths(core.StreamLosses(stream)))
	}
	_, calls := backend.take()
	if len(calls) != 2 {
		t.Fatalf("calls = %+v", calls)
	}
	for _, call := range calls {
		if call.path != "/api/responses" || call.body != copilotAdaptedUpstream {
			t.Fatalf("upstream = %s %s\nwant %s", call.path, call.body, copilotAdaptedUpstream)
		}
		assertCopilotHeaders(t, call.header, "session-3", "application/json", true)
	}

	marked := chat
	marked.Body = []byte(`{"messages":[{"role":"user","content":"Say hello"}],"force_api_support":true}`)
	if response, err := provider.Invoke(context.Background(), marked); err != nil || json.Unmarshal(response.Body, &answer) != nil ||
		!reflect.DeepEqual(answer.Forced, map[string]any{"req_api": "chat", "resp_api": "responses"}) {
		t.Fatalf("marked answer = %s err = %v", response.Body, err)
	}
	direct := chat
	direct.Body = []byte(`{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":64,"force_api_support":false}`)
	if _, err := provider.Invoke(context.Background(), direct); err != nil {
		t.Fatal(err)
	}
	direct.Body = []byte(`{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":64}`)
	if _, err := disabled.Invoke(context.Background(), direct); err != nil {
		t.Fatal(err)
	}
	_, calls = backend.take()
	chatBody := `{"max_tokens":64,"messages":[{"content":"Say hello","role":"user"}],"model":"copilot-fixture-codex","stream":false}`
	if len(calls) != 3 || calls[0].path != "/api/responses" || calls[1].path != "/api/chat/completions" || calls[1].body != chatBody ||
		calls[2].path != "/api/chat/completions" || calls[2].body != chatBody {
		t.Fatalf("upstream = %+v", calls)
	}
}

func TestCopilotRefusesChatItCannotServeOverResponses(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilot(t))
	provider := listedCopilot(t, backend)
	_, err := provider.Invoke(context.Background(), copilotRequest(core.ModelSurfaceChatCompletions, "copilot-fixture-codex",
		`{"messages":[{"role":"user","content":"Say hello"}],"stop":["END"]}`, nil))
	assertCopilotFailure(t, err, core.ProviderErrorUnsupported, core.ProviderErrorClassification{FailoverEligible: true})
	if exchanges, calls := backend.take(); len(exchanges) != 0 || len(calls) != 0 {
		t.Fatalf("sent %v %+v", exchanges, calls)
	}
}

// A 400 that names a sampling parameter is sent once more without it, with
// the same session, as the gateway retries its adapted requests.
func TestCopilotRetriesWithoutTheSamplingParametersCopilotRejects(t *testing.T) {
	t.Parallel()
	answer := answerCopilot(t)
	backend := newCopilotBackend(t, answer)
	provider := listedCopilot(t, backend)
	backend.update(func(b *copilotBackend) {
		b.answer = func(w http.ResponseWriter, r *http.Request, body []byte) {
			if r.URL.Path == "/api/responses" && strings.Contains(string(body), `"temperature"`) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(w, `{"error":{"message":"Unsupported parameter: 'temperature' is not supported with this model."}}`)
				return
			}
			answer(w, r, body)
		}
	})
	response, err := provider.Invoke(context.Background(), copilotRequest(core.ModelSurfaceChatCompletions, "copilot-fixture-codex",
		`{"messages":[{"role":"user","content":"Say hello"}],"temperature":0.2,"top_p":0.9}`, copilotCredential("oauth-a")))
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"advisory dropped temperature"}; !reflect.DeepEqual(lossPaths(response.Losses), want) {
		t.Fatalf("losses = %v, want %v", lossPaths(response.Losses), want)
	}
	_, calls := backend.take()
	if len(calls) != 2 || calls[0].authorization != calls[1].authorization ||
		calls[0].body != `{"input":[{"content":"Say hello","role":"user"}],"model":"copilot-fixture-codex","temperature":0.2,"top_p":0.9}` {
		t.Fatalf("upstream = %+v", calls)
	}
}

// With adaptation on, a Chat model that lists reasoning efforts takes
// max_tokens as max_completion_tokens, overwriting one sent besides.
func TestCopilotRenamesMaxTokensForReasoningModels(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilot(t))
	provider := listedCopilot(t, backend)
	disabled := listedCopilot(t, backend, func(config *CopilotConfig, _ *copilotauth.Config) { config.DisableAdaptation = true })
	body := `{"messages":[],"max_tokens":64,"max_completion_tokens":32}`
	for _, testCase := range []struct {
		provider *Copilot
		model    string
		want     string
	}{
		{provider, "copilot-fixture-reasoning", `{"max_completion_tokens":64,"messages":[],"model":"copilot-fixture-reasoning","stream":false}`},
		{disabled, "copilot-fixture-reasoning", `{"max_completion_tokens":32,"max_tokens":64,"messages":[],"model":"copilot-fixture-reasoning","stream":false}`},
		{provider, "copilot-fixture-claude", `{"max_completion_tokens":32,"max_tokens":64,"messages":[],"model":"copilot-fixture-claude","stream":false}`},
		{provider, "unlisted", `{"max_completion_tokens":32,"max_tokens":64,"messages":[],"model":"unlisted","stream":false}`},
	} {
		response, err := testCase.provider.Invoke(context.Background(), copilotRequest(core.ModelSurfaceChatCompletions, testCase.model, body, nil))
		if err != nil {
			t.Fatal(err)
		}
		renamed := slices.Contains(lossPaths(response.Losses), "advisory renamed max_tokens")
		if _, calls := backend.take(); len(calls) != 1 || calls[0].body != testCase.want || renamed != (testCase.provider == provider && testCase.model == "copilot-fixture-reasoning") {
			t.Fatalf("%s: upstream = %+v losses = %v", testCase.model, calls, response.Losses)
		}
	}
}
