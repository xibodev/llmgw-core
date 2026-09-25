package providers

import (
	"context"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	copilotauth "github.com/xibodev/llm-provider-auth/copilot"
	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// copilotRichChat carries every field the gateway's Chat facade forwards to
// Copilot and several its transport drops. copilotRichChatUpstream is what
// the gateway's own chatKwargs and buildOpenAIPayload send for it: sorted
// keys, numbers as the gateway renders them, HTML unescaped. copilotRoutedChat
// carries the fields the transport forwards besides.
const (
	copilotRichChat = `{"model":"ignored","stream":true,"messages":[{"role":"user","content":"Say <b>hello</b> & more"}],` +
		`"temperature":1.0,"max_tokens":64,"top_p":0.5,"stop":["END"],` +
		`"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"n":{"type":"integer","maximum":1e3}}}}}],` +
		`"tool_choice":"auto","metadata":{"client":"fixture"},"reasoning_effort":"low","n":1,"response_format":{"type":"json_object"},` +
		`"user":"fixture-user","fallback_timeout_ms":5000,` +
		`"force_api_support":false,"seed":null}`
	copilotRichChatUpstream = `{"max_tokens":64,"messages":[{"content":"Say <b>hello</b> & more","role":"user"}],"metadata":{"client":"fixture"},` +
		`"model":"gpt-fixture","reasoning_effort":"low","stop":["END"],"stream":false,"temperature":1,"tool_choice":"auto",` +
		`"tools":[{"function":{"name":"lookup","parameters":{"properties":{"n":{"maximum":1000,"type":"integer"}},"type":"object"}},"type":"function"}],"top_p":0.5}`
	copilotChatAnswer = `{"id":"chatcmpl-fixture","object":"chat.completion","model":"gpt-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"Hello <b>there</b>"},"finish_reason":"stop"}]}`
	// copilotRoutedChat is the Chat body the gateway's Copilot facade hands
	// core for a request its router serves over Chat from Responses or
	// Messages: its transport's payload, with parallel_tool_calls,
	// stream_options and thinking, and output_config, which that payload
	// never carried. copilotRoutedChatUpstream is what the gateway's
	// transport sent for it, recorded from the gateway's own payload builder.
	copilotRoutedChat = `{"model":"chat-model","stream":false,"messages":[{"role":"user","content":[{"type":"text","text":"Say <b>hi</b> & more"},` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}],"temperature":0.5,"top_p":1.0,"max_tokens":64,"stop":["END"],` +
		`"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"tool_choice":"auto","metadata":{"client":"fixture"},` +
		`"reasoning_effort":"low","parallel_tool_calls":false,"stream_options":{"include_usage":true},` +
		`"thinking":{"type":"enabled","budget_tokens":1024},"output_config":{"effort":"low"},"force_api_support":true}`
	copilotRoutedChatUpstream = `{"max_tokens":64,"messages":[{"content":[{"text":"Say <b>hi</b> & more","type":"text"},` +
		`{"image_url":{"url":"data:image/png;base64,AA=="},"type":"image_url"}],"role":"user"}],"metadata":{"client":"fixture"},` +
		`"model":"chat-model","parallel_tool_calls":false,"reasoning_effort":"low","stop":["END"],"stream":false,` +
		`"stream_options":{"include_usage":true},"temperature":0.5,"thinking":{"budget_tokens":1024,"type":"enabled"},` +
		`"tool_choice":"auto","tools":[{"function":{"name":"lookup","parameters":{"type":"object"}},"type":"function"}],"top_p":1}`
)

func answerCopilotChat(w http.ResponseWriter, _ *http.Request, _ []byte) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, copilotChatAnswer)
}

func TestNewCopilotRequiresAuthAndTheProductsIdentity(t *testing.T) {
	t.Parallel()
	auth := copilotauth.New(copilotauth.Config{AllowProxy: true})
	for want, config := range map[string]CopilotConfig{
		"Copilot auth client is required":           {EditorPluginVersion: "plugin/1", UserAgent: "agent/1"},
		"Copilot editor plugin version is required": {Auth: auth, UserAgent: "agent/1"},
		"Copilot user agent is required":            {Auth: auth, EditorPluginVersion: "plugin/1", UserAgent: " "},
	} {
		if _, err := NewCopilot(config); err == nil || err.Error() != want {
			t.Errorf("err = %v, want %q", err, want)
		}
	}
}

// The gateway's Chat facade forwards nine fields and its transport sends
// each that is not null; everything else is dropped, never refused. The
// body, the headers and the answer must match the gateway byte for byte.
func TestCopilotShapesChatAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilotChat)
	response, err := newFixtureCopilot(t, backend).Invoke(context.Background(),
		copilotRequest(core.ModelSurfaceChatCompletions, "gpt-fixture", copilotRichChat, copilotCredential("caller-oauth")))
	if err != nil {
		t.Fatal(err)
	}
	if string(response.Body) != copilotChatAnswer || response.ContentType != core.ContentTypeJSON {
		t.Fatalf("response = %s %s, want Copilot's answer unchanged", response.ContentType, response.Body)
	}
	exchanges, calls := backend.take()
	if !reflect.DeepEqual(exchanges, []string{"caller-oauth"}) || len(calls) != 1 {
		t.Fatalf("exchanges = %v calls = %+v", exchanges, calls)
	}
	call := calls[0]
	if call.method != http.MethodPost || call.path != "/api/chat/completions" || call.body != copilotRichChatUpstream {
		t.Fatalf("upstream = %s %s %s\nwant body %s", call.method, call.path, call.body, copilotRichChatUpstream)
	}
	assertCopilotHeaders(t, call.header, "session-1", "application/json", false)
	want := []core.Loss{
		{Path: "fallback_timeout_ms", Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "the Copilot transport does not carry this Chat field"},
		{Path: "n", Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "the Copilot transport does not carry this Chat field"},
		{Path: "response_format", Class: translate.LossDropped, Severity: translate.LossMaterial, Detail: "the Copilot transport does not carry this Chat field"},
		{Path: "user", Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "the Copilot transport does not carry this Chat field"},
	}
	if !reflect.DeepEqual(response.Losses, want) {
		t.Fatalf("losses = %+v\nwant %+v", response.Losses, want)
	}
}

// The gateway's Copilot transport forwards what the OpenAI-compatible
// transport forwards, so parallel_tool_calls, stream_options and thinking,
// which the router hands a request it serves over Chat, reach Copilot as that
// transport sent them, streamed or not. A field it drops is still reported.
func TestCopilotForwardsTheTransportsChatFields(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilotChat)
	provider := newFixtureCopilot(t, backend)
	request := copilotRequest(core.ModelSurfaceChatCompletions, "chat-model", copilotRoutedChat, copilotCredential("caller-oauth"))
	response, err := provider.Invoke(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := provider.Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	drainCopilot(t, stream)
	want := []core.Loss{{Path: "output_config", Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "the Copilot transport does not carry this Chat field"}}
	if !reflect.DeepEqual(response.Losses, want) || !reflect.DeepEqual(core.StreamLosses(stream), want) {
		t.Fatalf("losses = %+v and %+v\nwant %+v", response.Losses, core.StreamLosses(stream), want)
	}
	_, calls := backend.take()
	streamed := strings.Replace(copilotRoutedChatUpstream, `"stream":false`, `"stream":true`, 1)
	if len(calls) != 2 || calls[0].body != copilotRoutedChatUpstream || calls[1].body != streamed {
		t.Fatalf("upstream = %+v\nwant %s\nand %s", calls, copilotRoutedChatUpstream, streamed)
	}
	for _, call := range calls {
		assertCopilotHeaders(t, call.header, "session-1", "application/json", true)
	}
}

// assertCopilotHeaders checks the gateway's Copilot headers exactly.
func assertCopilotHeaders(t *testing.T, header http.Header, session, accept string, vision bool) {
	t.Helper()
	want := map[string]string{
		"Authorization": "Bearer " + session, "Content-Type": "application/json", "Accept": accept,
		"Copilot-Integration-Id": "vscode-chat", "Editor-Version": "vscode/1.95.3", "Editor-Plugin-Version": "fixture-plugin/1.0",
		"Openai-Intent": "conversation-panel", "User-Agent": "FixtureCopilotChat/1.0", "Accept-Encoding": "gzip",
	}
	if vision {
		want["Copilot-Vision-Request"] = "true"
	}
	got := map[string]string{}
	for name, values := range header {
		if name != "Content-Length" {
			got[name] = strings.Join(values, ", ")
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("headers = %v\nwant %v", got, want)
	}
}

// A translation.Adapter leaves a Responses request's max_output_tokens in
// the Chat body where llm-translate carries it, and the gateway sends it as
// max_completion_tokens unless the request limits its output already.
func TestCopilotSendsATranslatedOutputLimitAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilot(t))
	provider := listedCopilot(t, backend)
	for _, testCase := range []struct{ model, body, want string }{
		{"gpt-fixture", `{"messages":[],"_max_output_tokens":64}`, `{"max_completion_tokens":64,"messages":[],"model":"gpt-fixture","stream":false}`},
		{"gpt-fixture", `{"messages":[],"_max_output_tokens":64,"max_tokens":32}`, `{"max_tokens":32,"messages":[],"model":"gpt-fixture","stream":false}`},
		{"copilot-fixture-codex", `{"messages":[{"role":"user","content":"hi"}],"_max_output_tokens":64}`,
			`{"input":[{"content":"hi","role":"user"}],"max_output_tokens":64,"model":"copilot-fixture-codex"}`},
	} {
		response, err := provider.Invoke(context.Background(), copilotRequest(core.ModelSurfaceChatCompletions, testCase.model, testCase.body, nil))
		if err != nil {
			t.Fatal(err)
		}
		if _, calls := backend.take(); len(calls) != 1 || calls[0].body != testCase.want {
			t.Fatalf("%s: upstream = %+v, want %s", testCase.body, calls, testCase.want)
		}
		for _, loss := range response.Losses {
			if loss.Path == copilotOutputLimit {
				t.Fatalf("%s: the output limit was reported as a loss", testCase.body)
			}
		}
	}
}
