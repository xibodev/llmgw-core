package providers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

func codexChatRequest(body string) core.Request {
	request := codexResponsesRequest(body)
	request.Surface = core.ModelSurfaceChatCompletions
	request.Model = "codex-alpha-2026-09"
	return request
}

func droppedChatField(path string, harmless bool) core.Loss {
	detail := "the Codex Responses transport does not carry this Chat field"
	if harmless {
		detail = "Codex already behaves as this value asks"
	}
	return core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: detail}
}

// A Chat request the gateway's facade serves reaches Codex exactly as
// CodexProvider.Complete sends the payload the facade hands it: the fields
// the facade drops never arrive, and Codex reports them instead.
func TestCodexServesChatAsTheGatewayFacadeDoes(t *testing.T) {
	t.Parallel()
	events := readCodexFixture(t, "response-complete.sse")
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
		_, _ = w.Write(events)
	}))
	defer server.Close()
	provider, err := NewCodex(CodexConfig{
		Instructions: "Follow the caller's request.", ClientVersion: fixtureCodexClientVersion,
		ResponsesURL: server.URL + "/responses", ModelsURL: server.URL + "/models", Client: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	response, err := provider.Invoke(context.Background(), codexChatRequest(`{
		"model": "codex-alpha-2026-09", "stream": false,
		"messages": [{"role": "user", "content": "Use the tool"}],
		"tools": [{"type": "function", "function": {"name": "lookup", "description": "Lookup", "parameters": {"type": "object"}}}],
		"prompt_cache_key": "fixture-cache",
		"max_tokens": 64, "temperature": 0.2, "top_p": 0.9, "stop": ["END"], "stream_options": {"include_usage": true},
		"n": 1, "tool_choice": "auto", "parallel_tool_calls": true, "logprobs": false,
		"response_format": {"type": "text"}, "modalities": ["text"]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	request := <-requests
	assertCodexRequestFixture(t, request.request, request.body, "request-chat.json")
	var chat struct {
		Choices []struct {
			Message struct {
				Content   any    `json:"content"`
				Reasoning string `json:"reasoning_content"`
				ToolCalls []any  `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(response.Body, &chat); err != nil || len(chat.Choices) != 1 || chat.Choices[0].Message.Content != nil ||
		chat.Choices[0].Message.Reasoning != "fixture reasoning" || len(chat.Choices[0].Message.ToolCalls) != 1 || chat.Usage.TotalTokens != 18 {
		t.Fatalf("chat response = %s", response.Body)
	}
	want := []core.Loss{
		droppedChatField("logprobs", true), droppedChatField("max_tokens", false), droppedChatField("modalities", true),
		droppedChatField("n", true), droppedChatField("parallel_tool_calls", true), droppedChatField("response_format", true),
		droppedChatField("stop", false), droppedChatField("stream_options", false), droppedChatField("temperature", false),
		droppedChatField("tool_choice", true), droppedChatField("top_p", false),
	}
	if !reflect.DeepEqual(response.Losses, want) {
		t.Fatalf("losses = %+v\nwant %+v", response.Losses, want)
	}
}

// The gateway refuses these rather than drop them; so does Codex, before
// anything is sent. Another target may serve them.
func TestCodexChatRefusesWhatTheGatewayRefuses(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls.Add(1) }))
	defer server.Close()
	provider := newFixtureCodex(t, server)
	unsupported := core.ProviderErrorClassification{FailoverEligible: true}
	for field, value := range map[string]string{
		"response_format": `{"type":"json_schema"}`, "n": `2`, "tool_choice": `"required"`,
		"parallel_tool_calls": `false`, "logprobs": `true`, "top_logprobs": `3`,
		"modalities": `["text","audio"]`, "audio": `{"voice":"fixture"}`, "prediction": `{"type":"content"}`,
		"tools": `[{"type":"computer"}]`,
	} {
		request := codexChatRequest(`{"messages":[{"role":"user","content":"hi"}],"` + field + `":` + value + `}`)
		_, err := provider.Invoke(context.Background(), request)
		assertCodexFailure(t, err, core.ProviderErrorUnsupported, unsupported)
		if field != "tools" && !strings.Contains(err.Error(), field) {
			t.Fatalf("%s: error %q does not name the field", field, err)
		}
		_, err = provider.Stream(context.Background(), request)
		assertCodexFailure(t, err, core.ProviderErrorUnsupported, unsupported)
	}
	_, err := provider.Invoke(context.Background(), codexChatRequest(`{"model":"codex-alpha-2026-09"}`))
	assertCodexFailure(t, err, core.ProviderErrorInvalidRequest, core.ProviderErrorClassification{})
	if calls.Load() != 0 {
		t.Fatalf("upstream received %d requests Codex cannot serve", calls.Load())
	}
}

func TestCodexStreamsChatAndReportsItsLosses(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"type":"response.reasoning_summary_text.delta","delta":"thinking"}`,
			`data: {"type":"response.output_text.delta","delta":"answer"}`,
			`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}`,
		}, "\n\n")+"\n\n")
	}))
	defer server.Close()
	stream, err := newFixtureCodex(t, server).Stream(context.Background(), codexChatRequest(
		`{"messages":[{"role":"user","content":"go"}],"stream":true,"max_tokens":8,"temperature":0.5}`))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	want := []core.Loss{droppedChatField("max_tokens", false), droppedChatField("temperature", false)}
	if losses := core.StreamLosses(stream); !reflect.DeepEqual(losses, want) {
		t.Fatalf("losses before the first frame = %+v, want %+v", losses, want)
	}
	var chunks strings.Builder
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		chunks.Write(frame)
	}
	for _, expected := range []string{`"reasoning_content":"thinking"`, `"content":"answer"`, `"total_tokens":5`} {
		if !strings.Contains(chunks.String(), expected) {
			t.Fatalf("stream missing %s: %s", expected, chunks.String())
		}
	}
	if !strings.HasSuffix(chunks.String(), "data: [DONE]\n\n") {
		t.Fatalf("stream is not terminal Chat SSE: %q", chunks.String())
	}
}

// Codex serves a Gemini history as CodexProvider always has, and reports
// each signature it drops, as advisory.
func TestCodexChatServesThoughtSignatureHistories(t *testing.T) {
	t.Parallel()
	bodies := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodies <- string(body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, codexCompletedStream)
	}))
	defer server.Close()
	response, err := newFixtureCodex(t, server).Invoke(context.Background(), codexChatRequest(`{"messages":[
		{"role":"user","content":"look it up"},
		{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","thought_signature":"fixture-signature-c",
			"function":{"name":"lookup","arguments":"{}","thought_signature":"fixture-signature-b"},
			"extra_content":{"google":{"thought_signature":"fixture-signature-a"}}}]},
		{"role":"tool","tool_call_id":"call_1","content":"found"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if body := <-bodies; strings.Contains(body, "fixture-signature") || !strings.Contains(body, `"function_call"`) {
		t.Fatalf("upstream request = %s, want the tool call without its signatures", body)
	}
	signatures := 0
	for _, loss := range response.Losses {
		if loss.Severity != translate.LossAdvisory {
			t.Fatalf("loss %+v is not advisory", loss)
		}
		if strings.HasSuffix(loss.Path, ".thought_signature") && loss.Class == translate.LossDropped {
			signatures++
		}
	}
	if signatures != 3 {
		t.Fatalf("losses = %+v, want the three dropped signatures", response.Losses)
	}
}
