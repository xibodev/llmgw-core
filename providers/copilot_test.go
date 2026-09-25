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

// copilotRichChat carries every field the gateway forwards to Copilot and
// several it drops. copilotRichChatUpstream is what the gateway's own
// chatKwargs and buildOpenAIPayload send for it: sorted keys, numbers as the
// gateway renders them, HTML unescaped.
const (
	copilotRichChat = `{"model":"ignored","stream":true,"messages":[{"role":"user","content":"Say <b>hello</b> & more"}],` +
		`"temperature":1.0,"max_tokens":64,"top_p":0.5,"stop":["END"],` +
		`"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"n":{"type":"integer","maximum":1e3}}}}}],` +
		`"tool_choice":"auto","metadata":{"client":"fixture"},"reasoning_effort":"low","n":1,"response_format":{"type":"json_object"},` +
		`"stream_options":{"include_usage":true},"parallel_tool_calls":false,"user":"fixture-user","fallback_timeout_ms":5000,` +
		`"force_api_support":false,"seed":null}`
	copilotRichChatUpstream = `{"max_tokens":64,"messages":[{"content":"Say <b>hello</b> & more","role":"user"}],"metadata":{"client":"fixture"},` +
		`"model":"gpt-fixture","reasoning_effort":"low","stop":["END"],"stream":false,"temperature":1,"tool_choice":"auto",` +
		`"tools":[{"function":{"name":"lookup","parameters":{"properties":{"n":{"maximum":1000,"type":"integer"}},"type":"object"}},"type":"function"}],"top_p":0.5}`
	copilotChatAnswer = `{"id":"chatcmpl-fixture","object":"chat.completion","model":"gpt-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"Hello <b>there</b>"},"finish_reason":"stop"}]}`
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
		{Path: "parallel_tool_calls", Class: translate.LossDropped, Severity: translate.LossMaterial, Detail: "the Copilot transport does not carry this Chat field"},
		{Path: "response_format", Class: translate.LossDropped, Severity: translate.LossMaterial, Detail: "the Copilot transport does not carry this Chat field"},
		{Path: "stream_options", Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "the Copilot transport does not carry this Chat field"},
		{Path: "user", Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "the Copilot transport does not carry this Chat field"},
	}
	if !reflect.DeepEqual(response.Losses, want) {
		t.Fatalf("losses = %+v\nwant %+v", response.Losses, want)
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
