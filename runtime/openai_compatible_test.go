package runtime_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
	"github.com/xibodev/llmgw-core/translation"
)

// gatewayGolden is one of the gateway's characterization goldens, copied
// unchanged into testdata/gateway: what its client received and every
// request it sent upstream.
type gatewayGolden struct {
	upstream      []string // "METHOD path", then its body when it had one
	transportMode string
	body          string
}

func readGatewayGolden(t *testing.T, name string) gatewayGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "gateway", name+".golden"))
	if err != nil {
		t.Fatal(err)
	}
	head, body, found := strings.Cut(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\nbody:\n")
	if !found {
		t.Fatalf("%s has no body", name)
	}
	golden := gatewayGolden{body: strings.TrimSuffix(body, "\n")}
	for _, line := range strings.Split(head, "\n") {
		if call, ok := strings.CutPrefix(line, "upstream: "); ok {
			golden.upstream = append(golden.upstream, call)
		} else if sent, ok := strings.CutPrefix(line, "upstream body: "); ok {
			golden.upstream = append(golden.upstream, sent)
		} else if mode, ok := strings.CutPrefix(line, "header X-Llmgw-Transport-Mode: "); ok {
			golden.transportMode = mode
		}
	}
	return golden
}

// characterized renders JSON as the gateway's characterization does: keys
// sorted, times and generated identifiers replaced, and fixture
// identifiers kept.
func characterized(t *testing.T, raw []byte, indent bool) string {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return strings.TrimSpace(string(raw))
	}
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	if indent {
		encoder.SetIndent("", "  ")
	}
	if err := encoder.Encode(characterizedValue("", value)); err != nil {
		t.Fatal(err)
	}
	return strings.TrimRight(out.String(), "\n")
}

func characterizedValue(key string, value any) any {
	volatile := false
	switch key {
	case "created", "created_at", "discovered_at", "verified_at", "expires_at", "observed_at", "refreshed_at":
		volatile = true
	}
	switch typed := value.(type) {
	case map[string]any:
		for name, item := range typed {
			typed[name] = characterizedValue(name, item)
		}
	case []any:
		for index, item := range typed {
			typed[index] = characterizedValue(key, item)
		}
	case json.Number:
		if volatile {
			return "<time>"
		}
	case string:
		if volatile && typed != "" {
			return "<time>"
		}
		if (key == "id" || strings.HasSuffix(key, "_id")) && typed != "" && !strings.Contains(typed, "fixture") {
			if separator := strings.IndexAny(typed, "-_"); separator > 0 {
				return "<generated " + typed[:separator+1] + ">"
			}
			return "<generated>"
		}
	}
	return value
}

// characterizedFrames renders SSE frames as the gateway's characterization
// does, each data object characterized.
func characterizedFrames(t *testing.T, frames []string) string {
	t.Helper()
	rendered := make([]string, 0, len(frames))
	for _, frame := range frames {
		lines := strings.Split(strings.TrimRight(strings.ReplaceAll(frame, "\r\n", "\n"), "\n"), "\n")
		for index, line := range lines {
			if data, ok := strings.CutPrefix(line, "data: "); ok && strings.HasPrefix(data, "{") {
				lines[index] = "data: " + characterized(t, []byte(data), false)
			}
		}
		rendered = append(rendered, strings.Join(lines, "\n"))
	}
	return strings.Join(rendered, "\n\n")
}

// drainFrames reads a stream to its end, or to the error it fails with.
func drainFrames(stream core.StreamIter) ([]string, error) {
	defer stream.Close()
	var frames []string
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			return frames, nil
		}
		if err != nil {
			return frames, err
		}
		frames = append(frames, string(frame))
	}
}

// gatewayUpstream is the upstream the gateway's characterization test
// serves, answer for answer: its catalogs, and the Chat and Responses
// answers it writes for each request. It records calls as the goldens do.
type gatewayUpstream struct {
	mu       sync.Mutex
	calls    []string
	failover string
}

func (u *gatewayUpstream) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	u.mu.Lock()
	u.calls = append(u.calls, r.Method+" "+r.URL.Path)
	if len(bytes.TrimSpace(body)) > 0 {
		u.calls = append(u.calls, string(body))
	}
	failover := u.failover
	u.mu.Unlock()
	// The gateway's instances carry this key, and so does the owner's
	// credential here.
	if r.Header.Get("Authorization") != "Bearer fixture-upstream-token" {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	stream := payload["stream"] == true
	model, _ := payload["model"].(string)
	switch r.URL.Path {
	case "/f/models":
		writeGatewayJSON(w, http.StatusOK, map[string]any{"data": []any{
			map[string]any{"id": "chat-model", "supported_endpoints": []string{"/chat/completions"}},
			map[string]any{"id": "responses-model", "supported_endpoints": []string{"/chat/completions", "/responses"}},
		}})
	case "/a/models", "/b/models":
		id := "model-" + r.URL.Path[1:2]
		writeGatewayJSON(w, http.StatusOK, map[string]any{"data": []any{map[string]any{"id": id, "supported_endpoints": []string{"/chat/completions"}}}})
	case "/f/chat/completions", "/b/chat/completions":
		writeGatewayChat(w, model, stream)
	case "/a/chat/completions":
		if failover == "before-output" {
			writeGatewayJSON(w, http.StatusServiceUnavailable, map[string]any{"error": map[string]any{"message": "fixture unavailable", "type": "server_error"}})
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: {\"id\":\"chatcmpl_fixture\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":%q,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Partial\"},\"finish_reason\":null}]}\n\n", model)
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler)
	case "/f/responses":
		writeGatewayResponses(w, model, stream)
	default:
		http.NotFound(w, r)
	}
}

func (u *gatewayUpstream) take() []string {
	u.mu.Lock()
	defer u.mu.Unlock()
	calls := u.calls
	u.calls = nil
	return calls
}

func writeGatewayJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeGatewayChat(w http.ResponseWriter, model string, stream bool) {
	if !stream {
		writeGatewayJSON(w, http.StatusOK, map[string]any{
			"id": "chatcmpl_fixture", "object": "chat.completion", "created": 1700000000, "model": model,
			"choices": []any{map[string]any{"index": 0, "finish_reason": "stop", "message": map[string]any{"role": "assistant", "content": "Hello from chat"}}},
			"usage":   map[string]any{"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
		})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	chunk := func(delta, finish, usage string) string {
		return fmt.Sprintf("data: {\"id\":\"chatcmpl_fixture\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":%q,\"choices\":[{\"index\":0,\"delta\":%s,\"finish_reason\":%s}]%s}\n\n", model, delta, finish, usage)
	}
	_, _ = fmt.Fprint(w, chunk(`{"role":"assistant","content":"Hello"}`, "null", "")+chunk(`{"content":" from chat"}`, "null", "")+
		chunk(`{}`, `"stop"`, `,"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}`)+"data: [DONE]\n\n")
}

func writeGatewayResponses(w http.ResponseWriter, model string, stream bool) {
	message := map[string]any{
		"id": "msg_fixture", "type": "message", "status": "completed", "role": "assistant",
		"content": []any{map[string]any{"type": "output_text", "text": "Hello from responses", "annotations": []any{}}},
	}
	completed := map[string]any{
		"id": "resp_fixture", "object": "response", "created_at": 1700000000, "status": "completed", "model": model,
		"output": []any{message}, "usage": map[string]any{"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
	}
	if !stream {
		writeGatewayJSON(w, http.StatusOK, completed)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	for index, event := range []map[string]any{
		{"type": "response.created", "response": map[string]any{"id": "resp_fixture", "object": "response", "created_at": 1700000000, "status": "in_progress", "model": model, "output": []any{}}},
		{"type": "response.output_item.added", "output_index": 0, "item": map[string]any{"id": "msg_fixture", "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}}},
		{"type": "response.output_text.delta", "item_id": "msg_fixture", "output_index": 0, "content_index": 0, "delta": "Hello"},
		{"type": "response.output_text.delta", "item_id": "msg_fixture", "output_index": 0, "content_index": 0, "delta": " from responses"},
		{"type": "response.output_text.done", "item_id": "msg_fixture", "output_index": 0, "content_index": 0, "text": "Hello from responses"},
		{"type": "response.output_item.done", "output_index": 0, "item": message},
		{"type": "response.completed", "response": completed},
	} {
		event["sequence_number"] = index
		data, _ := json.Marshal(event)
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event["type"], data)
	}
}

type openAISettings struct{ BaseURL string }

// openAIFixture is a Runtime over the gateway's characterization instances,
// each an OpenAICompatible behind a translation.Adapter that reads the
// catalog the Runtime keeps, keyed by the owner's credential.
type openAIFixture struct {
	runtime  *coreruntime.Runtime[openAISettings]
	owner    core.Caller
	upstream *gatewayUpstream

	mu    sync.Mutex
	built map[string]core.Provider
}

func newOpenAIFixture(t *testing.T) *openAIFixture {
	t.Helper()
	ctx := context.Background()
	fixture := &openAIFixture{owner: core.Caller{ID: "owner", Kind: core.CallerHuman}, upstream: &gatewayUpstream{}, built: map[string]core.Provider{}}
	server := httptest.NewServer(fixture.upstream)
	t.Cleanup(server.Close)
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(ctx, "fixture-upstream", core.APIKeyRecord("fixture-upstream-token")); err != nil {
		t.Fatal(err)
	}
	paths := map[string]string{"fixture": "/f", "fixture-a": "/a", "fixture-b": "/b"}
	for instance := range paths {
		store.Bind(fixture.owner, instance, "fixture-upstream")
	}
	catalogs := core.NewMemoryCatalogStore()
	runtime, err := coreruntime.New(coreruntime.Options[openAISettings]{
		Settings: coreruntime.NewMemorySettings(openAISettings{BaseURL: server.URL}),
		Providers: func(settings openAISettings, instance string) (core.Provider, error) {
			provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{
				BaseURL: settings.BaseURL + paths[instance], Client: server.Client(), CatalogClient: server.Client(),
				Models: func(model string) (core.ModelInfo, bool) {
					record, err := catalogs.Load(context.Background(), core.CatalogKey{Instance: instance, CredentialKey: "fixture-upstream"})
					if err != nil {
						return core.ModelInfo{}, false
					}
					index := slices.IndexFunc(record.Evidence.Models, func(row core.ModelInfo) bool { return row.ID == model })
					if index < 0 {
						return core.ModelInfo{}, false
					}
					return record.Evidence.Models[index], true
				},
			})
			if err != nil {
				return nil, err
			}
			adapter := translation.Adapter{Provider: provider}
			fixture.mu.Lock()
			fixture.built[instance] = adapter
			fixture.mu.Unlock()
			return adapter, nil
		},
		Credentials: store, Catalogs: catalogs,
	})
	if err != nil {
		t.Fatal(err)
	}
	fixture.runtime = runtime
	// The gateway's characterization lists every catalog before it records.
	for instance := range paths {
		if record, err := runtime.ListModels(ctx, fixture.owner, instance); err != nil || record.Evidence.Status != core.CatalogDiscovered {
			t.Fatalf("%s catalog = %+v, err = %v", instance, record, err)
		}
	}
	fixture.upstream.take()
	return fixture
}

// transportMode is the label a product reads for a response, as the
// gateway's X-Llmgw-Transport-Mode header labels it.
func (f *openAIFixture) transportMode(instance, model string, surface core.ModelSurface) string {
	f.mu.Lock()
	provider := f.built[instance]
	f.mu.Unlock()
	if core.PreservesWire(provider, model, surface) {
		return "native"
	}
	return "translated"
}

// withCallerInstructions sets the instructions of every Responses object
// in raw to null. The gateway's Responses facade writes the caller's own
// instructions into each Responses object it relays, so that a preamble it
// added never shows; with none, that is null. The rest is the upstream's
// answer, which the provider returns as sent.
func withCallerInstructions(t *testing.T, raw []byte) []byte {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	if response, ok := value["response"].(map[string]any); ok {
		response["instructions"] = nil
	} else if value["object"] == "response" {
		value["instructions"] = nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func gatewayRequest(surface core.ModelSurface, model string, body map[string]any) core.Request {
	encoded, _ := json.Marshal(body)
	return core.Request{Surface: surface, Model: model, Body: encoded, ContentType: core.ContentTypeJSON}
}

// The OpenAI-compatible vertical replays the gateway's characterization of
// its generic OpenAI transport: each request reaches the upstream byte for
// byte as the gateway sent it, and the answer, native or through the
// translation.Adapter, is what the gateway's client received, less what
// the gateway's facades add.
func TestRuntimeReplaysTheGatewayOpenAICompatibleGoldens(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	fixture := newOpenAIFixture(t)
	say := []any{map[string]any{"role": "user", "content": "Say hello"}}
	chat := func(model string, stream bool) map[string]any {
		return map[string]any{"model": model, "stream": stream, "messages": say}
	}
	responses := func(model string, stream bool) map[string]any {
		return map[string]any{"model": model, "stream": stream, "input": "Say hello"}
	}
	messages := func(model string, stream bool) map[string]any {
		return map[string]any{"model": model, "stream": stream, "max_tokens": 64, "messages": say}
	}
	for _, test := range []struct {
		golden       string
		surface      core.ModelSurface
		model        string
		body         func(string, bool) map[string]any
		stream       bool
		instructions bool
	}{
		{"openai-chat-native", core.ModelSurfaceChatCompletions, "chat-model", chat, false, false},
		{"openai-chat-native-stream", core.ModelSurfaceChatCompletions, "chat-model", chat, true, false},
		{"openai-responses-native", core.ModelSurfaceResponses, "responses-model", responses, false, true},
		{"openai-responses-native-stream", core.ModelSurfaceResponses, "responses-model", responses, true, true},
		{"openai-responses-via-chat", core.ModelSurfaceResponses, "chat-model", responses, false, false},
		{"openai-messages-via-chat", core.ModelSurfaceMessages, "chat-model", messages, false, false},
		{"openai-messages-via-chat-stream", core.ModelSurfaceMessages, "chat-model", messages, true, false},
	} {
		t.Run(test.golden, func(t *testing.T) {
			golden := readGatewayGolden(t, test.golden)
			request := gatewayRequest(test.surface, test.model, test.body(test.model, test.stream))
			var got string
			if test.stream {
				stream, err := fixture.runtime.Stream(ctx, fixture.owner, "fixture", request)
				if err != nil {
					t.Fatal(err)
				}
				frames, err := drainFrames(stream)
				if err != nil {
					t.Fatal(err)
				}
				if test.instructions {
					for index, frame := range frames {
						lines := strings.Split(frame, "\n")
						for line, text := range lines {
							if data, ok := strings.CutPrefix(text, "data: "); ok && strings.HasPrefix(data, "{") {
								lines[line] = "data: " + string(withCallerInstructions(t, []byte(data)))
							}
						}
						frames[index] = strings.Join(lines, "\n")
					}
				}
				got = characterizedFrames(t, frames)
			} else {
				response, err := fixture.runtime.Invoke(ctx, fixture.owner, "fixture", request)
				if err != nil {
					t.Fatal(err)
				}
				if test.instructions {
					response.Body = withCallerInstructions(t, response.Body)
				}
				got = characterized(t, response.Body, true)
			}
			if got != golden.body {
				t.Errorf("the client's answer drifted from the gateway's\n--- gateway\n%s\n--- core\n%s", golden.body, got)
			}
			if calls := fixture.upstream.take(); !slices.Equal(calls, golden.upstream) {
				t.Errorf("upstream = %q, the gateway sent %q", calls, golden.upstream)
			}
			if golden.transportMode != "" && fixture.transportMode("fixture", test.model, test.surface) != golden.transportMode {
				t.Errorf("transport mode = %s, the gateway reports %s", fixture.transportMode("fixture", test.model, test.surface), golden.transportMode)
			}
		})
	}
}

// translation.Adapter does not stream Responses over Chat: llm-translate
// has no converter from a Chat stream to Responses events, which the
// gateway's Responses facade renders itself. The Adapter refuses before
// anything is sent, and a product that renders the events converts the
// request and streams Chat, which reaches the upstream as the gateway's
// request did.
func TestRuntimeStreamsResponsesOverChatAsTheGatewayRequestsIt(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	fixture := newOpenAIFixture(t)
	golden := readGatewayGolden(t, "openai-responses-via-chat-stream")
	payload := map[string]any{"model": "chat-model", "stream": true, "input": "Say hello"}
	var surface *core.SurfaceError
	if _, err := fixture.runtime.Stream(ctx, fixture.owner, "fixture", gatewayRequest(core.ModelSurfaceResponses, "chat-model", payload)); !errors.As(err, &surface) {
		t.Fatalf("err = %v, want a *core.SurfaceError", err)
	}
	if calls := fixture.upstream.take(); len(calls) != 0 {
		t.Fatalf("a refused stream reached the upstream: %q", calls)
	}
	converted, err := translate.ResponsesRequestToChatWithReport(payload)
	if err != nil || len(converted.Report.Losses) != 0 {
		t.Fatalf("conversion = %+v, err = %v", converted, err)
	}
	body := map[string]any{"model": "chat-model", "messages": converted.Value.Messages}
	for key, value := range converted.Value.Keywords {
		body[key] = value
	}
	stream, err := fixture.runtime.Stream(ctx, fixture.owner, "fixture", gatewayRequest(core.ModelSurfaceChatCompletions, "chat-model", body))
	if err != nil {
		t.Fatal(err)
	}
	frames, err := drainFrames(stream)
	if err != nil || len(frames) != 4 || frames[3] != "data: [DONE]\n\n" {
		t.Fatalf("frames = %q, err = %v", frames, err)
	}
	if calls := fixture.upstream.take(); !slices.Equal(calls, golden.upstream) {
		t.Fatalf("upstream = %q, the gateway sent %q", calls, golden.upstream)
	}
}

// The failover goldens, through execution.ExecuteStream: a member that
// fails before output yields to the next unseen, and one that fails after
// output reached the caller ends the stream with an *AfterOutputError that
// fails over to nothing. The gateway renders that error as a final event.
func TestRuntimeFailsStreamsOverAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	targets := []core.Target{{Provider: "fixture-a", Model: "model-a"}, {Provider: "fixture-b", Model: "model-b"}}
	carriesOutput := func(frame []byte) bool { return bytes.Contains(frame, []byte(`"content":"`)) }
	for _, mode := range []string{"before-output", "after-output"} {
		t.Run(mode, func(t *testing.T) {
			fixture := newOpenAIFixture(t)
			fixture.upstream.mu.Lock()
			fixture.upstream.failover = mode
			fixture.upstream.mu.Unlock()
			golden := readGatewayGolden(t, "failover-"+mode+"-stream")
			open := func(ctx context.Context, target core.Target) (core.StreamIter, error) {
				return fixture.runtime.Stream(ctx, fixture.owner, target.Provider, gatewayRequest(core.ModelSurfaceChatCompletions, target.Model, map[string]any{
					"model": target.Model, "stream": true, "messages": []any{map[string]any{"role": "user", "content": "Say hello"}},
				}))
			}
			executor := execution.Executor[core.Target]{
				Health: execution.NewHealthTracker(execution.HealthOptions{}), Key: func(target core.Target) string { return target.Provider },
			}
			result, err := execution.ExecuteStream(ctx, executor, targets, open, carriesOutput)
			if err != nil {
				t.Fatal(err)
			}
			frames, err := drainFrames(result.Value)
			wantFrames := golden.body
			if mode == "after-output" {
				var afterOutput *execution.AfterOutputError
				if !errors.As(err, &afterOutput) || core.ClassifyError(err) != (core.ProviderErrorClassification{}) || result.Candidate != targets[0] {
					t.Fatalf("err = %#v, served by %+v", err, result.Candidate)
				}
				// The gateway's own error event and [DONE] follow the output.
				wantFrames, _, _ = strings.Cut(golden.body, "\n\ndata: {\"error\"")
			} else if err != nil || result.Candidate != targets[1] || len(result.Attempts) != 2 ||
				result.Attempts[0].Classification.StatusCode != http.StatusServiceUnavailable || result.Attempts[0].Disposition != core.DispositionRetryable {
				t.Fatalf("err = %v, result = %+v", err, result)
			}
			if got := characterizedFrames(t, frames); got != wantFrames {
				t.Errorf("the client's frames drifted from the gateway's\n--- gateway\n%s\n--- core\n%s", wantFrames, got)
			}
			if calls := fixture.upstream.take(); !slices.Equal(calls, golden.upstream) {
				t.Errorf("upstream = %q, the gateway sent %q", calls, golden.upstream)
			}
		})
	}
}
