package providers

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers/zen"
	"github.com/xibodev/llmgw-core/translation"
)

func TestZenShapesKeyedChatAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	const completion = `{"id":"chatcmpl-keyed","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}]}`
	backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, _ int) { _, _ = io.WriteString(w, completion) }}
	server := httptest.NewServer(backend)
	defer server.Close()
	provider := newTestZen(t, server, nil)
	identity := zen.InvocationIdentity{Project: "project-fixture", Session: "session-fixture", Request: "request-fixture", Client: "desktop", UserAgent: zen.AnonymousUserAgent}
	ctx := zen.WithInvocationIdentity(context.Background(), identity)
	body := `{"model":"big-pickle","temperature":0.2,"seed":7,"stop":"END","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}`

	response, err := provider.Invoke(ctx, chatRequest("big-pickle", body, &core.Credential{APIKey: " Bearer fixture-key "}))
	if err != nil {
		t.Fatal(err)
	}
	if string(response.Body) != completion || len(response.Losses) != 1 || response.Losses[0].Path != "seed" {
		t.Fatalf("response = %s, losses = %+v", response.Body, response.Losses)
	}
	calls := backend.take()
	const upstream = `{"messages":[{"content":[{"image_url":{"url":"data:image/png;base64,AA=="},"type":"image_url"}],"role":"user"}],"model":"big-pickle","stop":"END","stream":false,"temperature":0.2}`
	if len(calls) != 1 || calls[0].path != "/zen/v1/chat/completions" || calls[0].body != upstream {
		t.Fatalf("upstream = %+v", calls)
	}
	if calls[0].header.Get("Copilot-Vision-Request") != "true" {
		t.Fatalf("an image request lacks the gateway's vision header: %v", calls[0].header)
	}
	calls[0].header.Del("Copilot-Vision-Request")
	assertZenHeaders(t, calls[0].header, "Bearer fixture-key", "", identity)
}

// A Responses request an Adapter serves over Chat carries its output limit
// as _max_output_tokens, which the gateway sends as max_completion_tokens.
func TestZenSendsATranslatedOutputLimit(t *testing.T) {
	t.Parallel()
	const completion = `{"id":"chatcmpl-keyed","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}]}`
	backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, _ int) { _, _ = io.WriteString(w, completion) }}
	server := httptest.NewServer(backend)
	defer server.Close()
	provider := newTestZen(t, server, nil)
	key := &core.Credential{APIKey: "fixture-key"}
	for body, want := range map[string]string{
		`{"messages":[{"role":"user","content":"Say hello"}],"_max_output_tokens":64,"temperature":0.2}`: `{"max_completion_tokens":64,"messages":[{"content":"Say hello","role":"user"}],"model":"big-pickle","stream":false,"temperature":0.2}`,
		`{"messages":[{"role":"user","content":"Say hello"}],"_max_output_tokens":64,"max_tokens":32}`:   `{"max_tokens":32,"messages":[{"content":"Say hello","role":"user"}],"model":"big-pickle","stream":false}`,
	} {
		if _, err := provider.Invoke(context.Background(), chatRequest("big-pickle", body, key)); err != nil {
			t.Fatal(err)
		}
		if calls := backend.take(); len(calls) != 1 || calls[0].body != want {
			t.Fatalf("%s: upstream = %+v", body, calls)
		}
	}
	adapter := translation.Adapter{Provider: provider}
	response, err := adapter.Invoke(context.Background(), responsesRequest("big-pickle", `{"input":"Say hello","max_output_tokens":64}`, key))
	if err != nil || !strings.Contains(string(response.Body), `"object":"response"`) {
		t.Fatalf("response = %s, err = %v", response.Body, err)
	}
	if calls := backend.take(); len(calls) != 1 || !strings.Contains(calls[0].body, `"max_completion_tokens":64`) || strings.Contains(calls[0].body, "_max_output_tokens") {
		t.Fatalf("translated upstream = %+v", calls)
	}
}

// zenMuseEvents is the gateway's Muse regression stream: reasoning and tool
// arguments the answer must not carry, then the answer.
const zenMuseEvents = "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"private reasoning\"}\n\n" +
	"data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"secret\\\":true}\"}\n\n" +
	"data: {\"type\":\"response.output_text.delta\",\"delta\":\"Muse answer\"}\n\n" +
	"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_muse\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"muse-spark-fixture\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Muse answer\"}]}]}}\n\n"

func TestZenServesAnonymousMuseChatOverResponses(t *testing.T) {
	t.Parallel()
	backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, _ int) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, zenMuseEvents)
	}}
	server := httptest.NewServer(backend)
	defer server.Close()
	provider := newTestZen(t, server, nil)
	body := `{"messages":[{"role":"user","content":"turn one"},{"role":"assistant","content":"turn two"},{"role":"user","content":"turn three"}],"max_tokens":2048}`

	response, err := provider.Invoke(context.Background(), chatRequest("muse-spark-fixture", body, nil))
	if err != nil {
		t.Fatal(err)
	}
	if text := string(response.Body); !strings.Contains(text, `"content":"Muse answer"`) || strings.Contains(text, "private reasoning") || strings.Contains(text, "secret") {
		t.Fatalf("response = %s", text)
	}
	calls := backend.take()
	if len(calls) != 1 || calls[0].path != "/zen/v1/responses" {
		t.Fatalf("upstream = %+v", calls)
	}
	assertZenHeaders(t, calls[0].header, "Bearer public", "application/json", freshIdentity)
	// The upstream body the gateway's regression test pins.
	tools := zenJSON(t, []any{
		map[string]any{"type": "function", "name": "bash", "description": "Executes a given bash/powershell command.", "parameters": map[string]any{
			"type": "object", "properties": map[string]any{"command": map[string]any{"type": "string", "description": "The command to execute"}}, "required": []any{"command"}}},
		map[string]any{"type": "function", "name": "read", "description": "Read a file from the local filesystem.", "parameters": map[string]any{
			"type": "object", "properties": map[string]any{"filePath": map[string]any{"type": "string", "description": "The absolute path to the file to read"}}, "required": []any{"filePath"}}},
	})
	want := `{"input":[{"content":"turn one","role":"user"},{"content":"turn two","role":"assistant"},{"content":"turn three","role":"user"}],` +
		`"max_output_tokens":2048,"model":"muse-spark-fixture","reasoning":{"effort":"minimal"},"stream":true,"tool_choice":"auto","tools":` + tools + `}`
	if calls[0].body != want {
		t.Fatalf("upstream body\n got %s\nwant %s", calls[0].body, want)
	}

	// A stream is converted from the same response, without the minimal
	// reasoning the gateway gives only a completion.
	stream, err := provider.Stream(context.Background(), chatRequest("muse-spark-fixture", body, nil))
	if err != nil {
		t.Fatal(err)
	}
	frames, err := readZenFrames(t, stream)
	if err != nil || !strings.Contains(frames, `"content":"Muse answer"`) || !strings.HasSuffix(frames, "data: [DONE]\n\n") {
		t.Fatalf("frames = %q, err = %v", frames, err)
	}
	if calls := backend.take(); len(calls) != 1 || calls[0].body != strings.Replace(want, `"reasoning":{"effort":"minimal"},`, "", 1) {
		t.Fatalf("stream upstream = %+v", calls)
	}
}

func TestZenServesKeyedChatOverResponsesAndRetriesARejectedSampling(t *testing.T) {
	t.Parallel()
	backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, call int) {
		if call == 1 {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"message":"Unsupported parameter: temperature"}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"resp_keyed","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi"}]}]}`)
	}}
	server := httptest.NewServer(backend)
	defer server.Close()
	row := core.ModelInfo{ID: "gpt-fixture", SupportedAPIs: []string{"/v1/responses"}}
	provider := newTestZen(t, server, func(model string) (core.ModelInfo, bool) { return row, model == row.ID })
	if got := provider.NativeSurfaces("gpt-fixture"); !reflect.DeepEqual(got, []core.ModelSurface{core.ModelSurfaceResponses, core.ModelSurfaceChatCompletions}) {
		t.Fatalf("surfaces = %v", got)
	}
	body := `{"messages":[{"role":"system","content":"Be brief"},{"role":"user","content":"Say hello"}],"max_tokens":64,"temperature":0.2}`
	response, err := provider.Invoke(context.Background(), chatRequest("gpt-fixture", body, &core.Credential{APIKey: "fixture-key"}))
	if err != nil || !strings.Contains(string(response.Body), `"content":"Hi"`) {
		t.Fatalf("response = %s, err = %v", response.Body, err)
	}
	calls := backend.take()
	const first = `{"input":[{"content":"Say hello","role":"user"}],"instructions":"Be brief","max_output_tokens":64,"model":"gpt-fixture","temperature":0.2}`
	if len(calls) != 2 || calls[0].body != first || calls[1].body != strings.Replace(first, `,"temperature":0.2`, "", 1) {
		t.Fatalf("upstream = %+v", calls)
	}
	for _, key := range []string{"X-Opencode-Session", "X-Opencode-Request"} {
		if calls[0].header.Get(key) == "" || calls[0].header.Get(key) != calls[1].header.Get(key) {
			t.Fatalf("%s changed across the retry", key)
		}
	}
}
