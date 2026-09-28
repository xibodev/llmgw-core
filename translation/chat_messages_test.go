package translation_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	"github.com/xibodev/llmgw-core/translation"
)

func fixture(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return strings.ReplaceAll(string(raw), "\r\n", "\n")
}

// sseRecords splits an SSE fixture into its records, each ending with its
// blank line, as a provider stream yields them.
func sseRecords(body string) []string {
	var records []string
	for _, record := range strings.Split(strings.TrimRight(body, "\n"), "\n\n") {
		records = append(records, record+"\n\n")
	}
	return records
}

// chatAnswer is what a Chat stream assembles to.
type chatAnswer struct {
	content   string
	calls     map[int]*chatCall
	finish    string
	usage     map[string]any
	done      int
	afterDone int
}

type chatCall struct {
	id, name, arguments string
}

func assembleChat(t *testing.T, frames []string) chatAnswer {
	t.Helper()
	answer := chatAnswer{calls: map[int]*chatCall{}}
	for _, frame := range frames {
		data, ok := strings.CutPrefix(strings.TrimSpace(frame), "data: ")
		if !ok {
			t.Fatalf("frame %q is not one data record", frame)
		}
		if answer.done > 0 {
			answer.afterDone++
		}
		if data == "[DONE]" {
			answer.done++
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   *string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage map[string]any `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("chunk %q is not JSON: %v", data, err)
		}
		if chunk.Usage != nil {
			answer.usage = chunk.Usage
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != nil {
				answer.content += *choice.Delta.Content
			}
			for _, call := range choice.Delta.ToolCalls {
				existing := answer.calls[call.Index]
				if existing == nil {
					existing = &chatCall{}
					answer.calls[call.Index] = existing
				}
				if call.ID != "" {
					existing.id = call.ID
				}
				if call.Function.Name != "" {
					existing.name = call.Function.Name
				}
				existing.arguments += call.Function.Arguments
			}
			if choice.FinishReason != nil {
				answer.finish = *choice.FinishReason
			}
		}
	}
	return answer
}

func hasLoss(losses []core.Loss, path string, class translate.LossClass, severity translate.LossSeverity) bool {
	return slices.ContainsFunc(losses, func(loss core.Loss) bool {
		return loss.Path == path && loss.Class == class && loss.Severity == severity
	})
}

func assertWeatherCall(t *testing.T, arguments string) {
	t.Helper()
	var input map[string]any
	if err := json.Unmarshal([]byte(arguments), &input); err != nil || input["city"] != "Paris" || input["unit"] != "celsius" {
		t.Fatalf("tool arguments = %q (%v), want the city and unit", arguments, err)
	}
}

const weatherChatRequest = `{
	"model": "m",
	"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"}],
	"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}}, "required": ["city"]}}}],
	"tool_choice": "required",
	"parallel_tool_calls": false,
	"max_completion_tokens": 256,
	"temperature": 1.5,
	"stop": "END",
	"user": "user-1",
	"seed": 7,
	"n": 1,
	"response_format": {"type": "text"}
}`

func TestChatOverMessages(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceMessages, response: fixture(t, "messages_tool_use.json")}
	response, err := translation.Adapter{Provider: provider}.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, weatherChatRequest))
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	received, body := provider.last(t)
	if received.Surface != core.ModelSurfaceMessages {
		t.Fatalf("upstream surface = %s, want messages", received.Surface)
	}
	if body["system"] != "Be brief." || body["max_tokens"] != float64(256) || body["temperature"] != float64(1) {
		t.Fatalf("upstream body = %v, want the system prompt, max_tokens and a clamped temperature", body)
	}
	if stops, _ := body["stop_sequences"].([]any); len(stops) != 1 || stops[0] != "END" {
		t.Fatalf("stop_sequences = %v", body["stop_sequences"])
	}
	if metadata, _ := body["metadata"].(map[string]any); metadata["user_id"] != "user-1" {
		t.Fatalf("metadata = %v, want the user", body["metadata"])
	}
	tools, _ := body["tools"].([]any)
	if tool, _ := tools[0].(map[string]any); len(tools) != 1 || tool["name"] != "get_weather" || tool["input_schema"] == nil {
		t.Fatalf("tools = %v", body["tools"])
	}
	if choice, _ := body["tool_choice"].(map[string]any); choice["type"] != "any" || choice["disable_parallel_tool_use"] != true {
		t.Fatalf("tool_choice = %v, want any without parallel tool use", body["tool_choice"])
	}
	for _, field := range []string{"seed", "n", "response_format", "user", "max_completion_tokens", "parallel_tool_calls", "stop"} {
		if _, ok := body[field]; ok {
			t.Fatalf("the Chat field %s reached the Messages request: %v", field, body)
		}
	}

	if !hasLoss(response.Losses, "max_completion_tokens", translate.LossRenamed, translate.LossAdvisory) ||
		!hasLoss(response.Losses, "temperature", translate.LossApproximated, translate.LossAdvisory) ||
		!hasLoss(response.Losses, "seed", translate.LossDropped, translate.LossAdvisory) {
		t.Fatalf("losses = %v, want the rename, the clamp and the dropped seed", response.Losses)
	}
	if hasLoss(response.Losses, "n", translate.LossDropped, translate.LossMaterial) || hasLoss(response.Losses, "response_format", translate.LossDropped, translate.LossMaterial) {
		t.Fatalf("losses = %v, want none for n=1 and a text response format", response.Losses)
	}

	completion := decode(t, response.Body)
	choices, _ := completion["choices"].([]any)
	choice, _ := choices[0].(map[string]any)
	message, _ := choice["message"].(map[string]any)
	calls, _ := message["tool_calls"].([]any)
	if message["content"] != "Checking the weather." || choice["finish_reason"] != "tool_calls" || len(calls) != 1 {
		t.Fatalf("completion = %v", completion)
	}
	call, _ := calls[0].(map[string]any)
	function, _ := call["function"].(map[string]any)
	if call["id"] != "toolu_01WEATHER" || function["name"] != "get_weather" {
		t.Fatalf("tool call = %v", call)
	}
	assertWeatherCall(t, function["arguments"].(string))
	assertChatUsage(t, completion["usage"])
}

// assertChatUsage checks the fixtures' usage as Chat counts it: 412 input
// tokens and 128 read from the cache make 540 prompt tokens.
func assertChatUsage(t *testing.T, raw any) {
	t.Helper()
	usage, _ := raw.(map[string]any)
	details, _ := usage["prompt_tokens_details"].(map[string]any)
	if usage["prompt_tokens"] != float64(540) || usage["completion_tokens"] != float64(58) || usage["total_tokens"] != float64(598) || details["cached_tokens"] != float64(128) {
		t.Fatalf("usage = %v, want 540 prompt tokens with 128 cached, 58 completion tokens", raw)
	}
}

func TestChatOverMessagesReportsWhatItDrops(t *testing.T) {
	t.Parallel()
	allowAll := core.LossPolicy{Rules: []core.LossRule{{Action: core.LossAllow}}}
	for _, test := range []struct {
		field, value string
		severity     translate.LossSeverity // "" when nothing is lost
	}{
		{field: "n", value: `1`},
		{field: "n", value: `2`, severity: translate.LossMaterial},
		{field: "presence_penalty", value: `0`},
		{field: "presence_penalty", value: `0.5`, severity: translate.LossMaterial},
		{field: "logprobs", value: `false`},
		{field: "logprobs", value: `true`, severity: translate.LossMaterial},
		{field: "modalities", value: `["text"]`},
		{field: "modalities", value: `["text","audio"]`, severity: translate.LossMaterial},
		{field: "response_format", value: `{"type":"text"}`},
		{field: "response_format", value: `{"type":"json_object"}`, severity: translate.LossMaterial},
		{field: "web_search_options", value: `{}`, severity: translate.LossMaterial},
		{field: "seed", value: `0`, severity: translate.LossAdvisory},
		{field: "store", value: `true`, severity: translate.LossAdvisory},
		{field: "reasoning_effort", value: `"low"`, severity: translate.LossAdvisory},
	} {
		provider := &fakeProvider{native: core.ModelSurfaceMessages, response: fixture(t, "messages_tool_use.json")}
		body := `{"messages":[{"role":"user","content":"hi"}],"` + test.field + `":` + test.value + `}`
		response, err := translation.Adapter{Provider: provider, Policy: allowAll}.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, body))
		if err != nil {
			t.Fatalf("%s=%s: %v", test.field, test.value, err)
		}
		if _, sent := provider.last(t); sent[test.field] != nil {
			t.Fatalf("%s=%s reached the Messages request", test.field, test.value)
		}
		lost := slices.IndexFunc(response.Losses, func(loss core.Loss) bool { return loss.Path == test.field })
		switch {
		case test.severity == "" && lost >= 0:
			t.Errorf("%s=%s asks for nothing, but lost %v", test.field, test.value, response.Losses[lost])
		case test.severity != "" && (lost < 0 || response.Losses[lost].Severity != test.severity):
			t.Errorf("%s=%s: losses = %v, want a %s loss", test.field, test.value, response.Losses, test.severity)
		}
	}
}

func TestChatOverMessagesDefaultsMaxTokensAndRejectsMaterialLosses(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceMessages, response: fixture(t, "messages_tool_use.json")}
	adapter := translation.Adapter{Provider: provider}
	if _, err := adapter.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, `{"messages":[{"role":"user","content":"hi"}]}`)); err != nil {
		t.Fatal(err)
	}
	if _, body := provider.last(t); body["max_tokens"] != float64(8192) {
		t.Fatalf("max_tokens = %v, want the default", body["max_tokens"])
	}

	calls := provider.calls.Load()
	_, err := adapter.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, `{"messages":[{"role":"user","content":"hi"}],"n":2}`))
	if err == nil || provider.calls.Load() != calls {
		t.Fatalf("n=2 was sent (err=%v): a material loss must be rejected before the provider is called", err)
	}
	allowed := translation.Adapter{Provider: provider, Policy: core.LossPolicy{Rules: []core.LossRule{{Path: "n", Action: core.LossAllow}}}}
	response, err := allowed.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, `{"messages":[{"role":"user","content":"hi"}],"n":2}`))
	if err != nil || !hasLoss(response.Losses, "n", translate.LossDropped, translate.LossMaterial) {
		t.Fatalf("allowed n=2: err=%v losses=%v", err, response.Losses)
	}
}

func TestChatOverMessagesConvertsAToolResultTurn(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceMessages, response: fixture(t, "messages_tool_use.json")}
	_, err := translation.Adapter{Provider: provider}.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, `{
		"messages": [
			{"role": "user", "content": "Weather in Paris?"},
			{"role": "assistant", "content": null, "tool_calls": [{"id": "toolu_01WEATHER", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
			{"role": "tool", "tool_call_id": "toolu_01WEATHER", "content": "18 C, clear"}
		],
		"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	_, body := provider.last(t)
	messages, _ := body["messages"].([]any)
	encoded, _ := json.Marshal(messages)
	for _, want := range []string{`"type":"tool_use"`, `"id":"toolu_01WEATHER"`, `"type":"tool_result"`, `"tool_use_id":"toolu_01WEATHER"`, `18 C, clear`} {
		if !strings.Contains(string(encoded), want) {
			t.Fatalf("messages = %s, want %s", encoded, want)
		}
	}
	if _, set := body["tool_choice"]; set {
		t.Fatalf("tool_choice = %v, want none when the Chat request set none", body["tool_choice"])
	}
}

func TestChatOverMessagesDropsReasoningWithALoss(t *testing.T) {
	t.Parallel()
	const request = `{"messages":[{"role":"user","content":"Weather?"}],"reasoning_effort":"high"}`
	answer := `{"id":"msg_01THINK","type":"message","role":"assistant","content":[` +
		`{"type":"thinking","thinking":"Paris is in France.","signature":"c2ln"},{"type":"text","text":"It is sunny."}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":20}}`
	provider := &fakeProvider{native: core.ModelSurfaceMessages, response: answer}
	adapter := translation.Adapter{Provider: provider}
	response, err := adapter.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, request))
	if err != nil {
		t.Fatal(err)
	}
	if _, body := provider.last(t); body["reasoning_effort"] != nil || body["thinking"] != nil || body["output_config"] != nil {
		t.Fatalf("upstream body = %v, want no reasoning setting", body)
	}
	if strings.Contains(string(response.Body), "Paris is in France") || !strings.Contains(string(response.Body), "It is sunny.") {
		t.Fatalf("completion = %s, want the text without the thinking", response.Body)
	}
	for _, path := range []string{"reasoning_effort", "content.0.reasoning"} {
		if !hasLoss(response.Losses, path, translate.LossDropped, translate.LossAdvisory) {
			t.Fatalf("losses = %v, want the advisory %s loss", response.Losses, path)
		}
	}
	if hasLoss(response.Losses, "content.0", translate.LossDropped, translate.LossMaterial) {
		t.Fatalf("losses = %v, want the thinking block reported once, as reasoning", response.Losses)
	}

	// A block Chat has no place for, other than thinking, is still material.
	provider.response = strings.Replace(answer, `{"type":"thinking","thinking":"Paris is in France.","signature":"c2ln"}`,
		`{"type":"web_search_tool_result","tool_use_id":"srvtoolu_01","content":[]}`, 1)
	_, err = adapter.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, request))
	var rejected *core.LossPolicyError
	if !errors.As(err, &rejected) || len(rejected.Losses) != 1 || rejected.Losses[0].Path != "content.0" {
		t.Fatalf("err = %v, want the lost search result rejected under the zero policy", err)
	}

	provider.frames = []string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01THINK\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Paris is in France.\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"c2ln\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"It is sunny.\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":20}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}
	stream, err := adapter.Stream(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, strings.Replace(request, "{", `{"stream":true,`, 1)))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	frames, err := collect(t, stream)
	if err != nil {
		t.Fatal(err)
	}
	streamed := assembleChat(t, frames)
	if streamed.content != "It is sunny." || streamed.finish != "stop" || strings.Contains(strings.Join(frames, ""), "Paris is in France") {
		t.Fatalf("streamed = %+v from %q, want the text without the thinking", streamed, frames)
	}
	if losses := core.StreamLosses(stream); !hasLoss(losses, "content.0.reasoning", translate.LossDropped, translate.LossAdvisory) {
		t.Fatalf("stream losses = %v, want the advisory reasoning loss", losses)
	}
}

func chatStreamRequest(includeUsage bool) core.Request {
	body := `{"model":"m","stream":true,"messages":[{"role":"user","content":"Weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]`
	if includeUsage {
		body += `,"stream_options":{"include_usage":true}`
	}
	return jsonRequest(core.ModelSurfaceChatCompletions, body+"}")
}

func TestChatStreamOverMessages(t *testing.T) {
	t.Parallel()
	for _, includeUsage := range []bool{true, false} {
		provider := &fakeProvider{native: core.ModelSurfaceMessages, frames: sseRecords(fixture(t, "messages_tool_use.sse"))}
		stream, err := translation.Adapter{Provider: provider}.Stream(context.Background(), chatStreamRequest(includeUsage))
		if err != nil {
			t.Fatal(err)
		}
		frames, err := collect(t, stream)
		if err != nil {
			t.Fatalf("stream error = %v", err)
		}
		_, body := provider.last(t)
		if body["stream"] != true {
			t.Fatalf("upstream body = %v, want a Messages stream", body)
		}
		answer := assembleChat(t, frames)
		call := answer.calls[0]
		if answer.content != "Checking the weather." || answer.finish != "tool_calls" || call == nil ||
			call.id != "toolu_01WEATHER" || call.name != "get_weather" || answer.done != 1 || answer.afterDone != 0 {
			t.Fatalf("include_usage=%t: answer = %+v, call = %+v", includeUsage, answer, call)
		}
		assertWeatherCall(t, call.arguments)

		losses := core.StreamLosses(stream)
		usageLoss := hasLoss(losses, "usage", translate.LossDropped, translate.LossMaterial)
		if includeUsage {
			if usageLoss {
				t.Fatal("the usage loss is reported, but the stream carries the usage")
			}
			assertChatUsage(t, answer.usage)
		} else if answer.usage != nil || !usageLoss {
			t.Fatalf("usage = %v, usage loss = %t; want no usage chunk and the usage loss reported", answer.usage, usageLoss)
		}
		if err := stream.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestChatStreamOverMessagesFailsOnAnErrorEvent(t *testing.T) {
	t.Parallel()
	records := sseRecords(fixture(t, "messages_tool_use.sse"))
	for _, test := range []struct {
		kind      string
		class     core.ProviderErrorClass
		transient bool
	}{
		{kind: "overloaded_error", class: core.ProviderErrorUpstream, transient: true},
		{kind: "api_error", class: core.ProviderErrorUpstream, transient: true},
		{kind: "rate_limit_error", class: core.ProviderErrorRateLimited, transient: true},
		{kind: "invalid_request_error", class: core.ProviderErrorUpstream},
		{kind: "permission_error", class: core.ProviderErrorForbidden},
	} {
		frames := append(append([]string(nil), records[:4]...),
			"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\""+test.kind+"\",\"message\":\"Overloaded\"}}\n\n")
		provider := &fakeProvider{native: core.ModelSurfaceMessages, frames: frames}
		stream, err := translation.Adapter{Provider: provider}.Stream(context.Background(), chatStreamRequest(false))
		if err != nil {
			t.Fatal(err)
		}
		delivered, err := collect(t, stream)
		_ = stream.Close()
		var providerErr *core.ProviderError
		if !errors.As(err, &providerErr) || providerErr.Class != test.class || !strings.Contains(providerErr.Message, test.kind+": Overloaded") {
			t.Fatalf("%s: stream error = %v, want a %s provider error", test.kind, err, test.class)
		}
		if got := providerErr.Classification; got.Retryable != test.transient || got.FailoverEligible != test.transient || got.CircuitFailure != test.transient {
			t.Fatalf("%s: classification = %+v, want transient=%t", test.kind, got, test.transient)
		}
		answer := assembleChat(t, delivered)
		if answer.content != "Checking the" || answer.finish != "" || answer.done != 0 {
			t.Fatalf("%s: delivered = %q; want the text before the error, no finish and no [DONE]", test.kind, delivered)
		}
	}
}

func TestChatStreamOverMessagesFailsWhenCutShort(t *testing.T) {
	t.Parallel()
	records := sseRecords(fixture(t, "messages_tool_use.sse"))
	provider := &fakeProvider{native: core.ModelSurfaceMessages, frames: records[:len(records)-1]}
	stream, err := translation.Adapter{Provider: provider}.Stream(context.Background(), chatStreamRequest(true))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	delivered, err := collect(t, stream)
	var providerErr *core.ProviderError
	if !errors.As(err, &providerErr) || !strings.Contains(providerErr.Message, "message_stop") || providerErr.Classification.Retryable || !providerErr.Classification.FailoverEligible {
		t.Fatalf("stream error = %v, want a failover-eligible error for a stream cut short", err)
	}
	if answer := assembleChat(t, delivered); answer.finish != "" || answer.usage != nil || answer.done != 0 {
		t.Fatalf("delivered = %q; a stream cut short must not look complete", delivered)
	}
}

func TestChatStreamOverMessagesSurfacesAnUpstreamFailure(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceMessages, frames: sseRecords(fixture(t, "messages_tool_use.sse")), failAt: 5}
	stream, err := translation.Adapter{Provider: provider}.Stream(context.Background(), chatStreamRequest(false))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	delivered, err := collect(t, stream)
	if !errors.Is(err, errUpstream) {
		t.Fatalf("stream error = %v, want the upstream's", err)
	}
	if answer := assembleChat(t, delivered); answer.finish != "" || answer.done != 0 {
		t.Fatalf("delivered = %q; a failed stream must not look complete", delivered)
	}
}

func TestChatStreamOverMessagesClosesWithoutDraining(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceMessages, frames: sseRecords(fixture(t, "messages_tool_use.sse"))}
	stream, err := translation.Adapter{Provider: provider}.Stream(context.Background(), chatStreamRequest(true))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Next(); err != nil {
		t.Fatal(err)
	}
	closed := make(chan error, 1)
	go func() { closed <- stream.Close() }()
	select {
	case err := <-closed:
		if err != nil || !provider.closed.Load() {
			t.Fatalf("Close() = %v, upstream closed = %t", err, provider.closed.Load())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not return: the converter goroutine is stuck")
	}
	if _, err := stream.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("Next after Close = %v, want io.EOF", err)
	}
}

// Serves and ServesStream must agree with what an Adapter actually does: it
// reaches the provider for a surface they report, natively or translated,
// and refuses any other with a SurfaceError before the provider is called.
func TestServesAgreesWithTheAdapter(t *testing.T) {
	t.Parallel()
	chat, messages, responses, embeddings := core.ModelSurfaceChatCompletions, core.ModelSurfaceMessages, core.ModelSurfaceResponses, core.ModelSurfaceEmbeddings
	bodies := map[core.ModelSurface]string{
		chat:       `{"messages":[{"role":"user","content":"hi"}]}`,
		messages:   `{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}`,
		responses:  `{"input":"hi"}`,
		embeddings: `{"input":"hi"}`,
	}
	for _, test := range []struct {
		native            []core.ModelSurface
		chat, chatStreams bool
	}{
		{native: []core.ModelSurface{chat}, chat: true, chatStreams: true},
		{native: []core.ModelSurface{responses}, chat: true, chatStreams: false},
		{native: []core.ModelSurface{messages}, chat: true, chatStreams: true},
		{native: []core.ModelSurface{responses, messages}, chat: true, chatStreams: true},
		{native: []core.ModelSurface{embeddings}},
		{native: []core.ModelSurface{core.ModelSurfaceAudioSpeech, core.ModelSurfaceAudioTranscriptions}},
		{native: nil},
	} {
		if got := translation.ServesChat(test.native...); got != test.chat {
			t.Errorf("ServesChat(%v) = %t, want %t", test.native, got, test.chat)
		}
		if got := translation.ServesStream(chat, test.native...); got != test.chatStreams {
			t.Errorf("ServesStream(chat, %v) = %t, want %t", test.native, got, test.chatStreams)
		}
		for surface, body := range bodies {
			for _, stream := range []bool{false, true} {
				provider := &surfacesProvider{native: test.native}
				adapter := translation.Adapter{Provider: provider}
				serves := translation.Serves(surface, test.native...)
				var err error
				if stream {
					serves = translation.ServesStream(surface, test.native...)
					_, err = adapter.Stream(context.Background(), jsonRequest(surface, body))
				} else {
					_, err = adapter.Invoke(context.Background(), jsonRequest(surface, body))
				}
				var refused *core.SurfaceError
				if reached := provider.calls.Load() > 0; reached != serves || (!serves && !errors.As(err, &refused)) {
					t.Errorf("native %v, %s, stream=%t: reached the provider = %t (err %v), but Serves says %t", test.native, surface, stream, reached, err, serves)
				}
			}
			adapter := translation.Adapter{Provider: &surfacesProvider{native: test.native}}
			if listed := slices.Contains(adapter.Surfaces("m"), surface); listed != translation.Serves(surface, test.native...) {
				t.Errorf("native %v: Adapter.Surfaces lists %s = %t, Serves says %t", test.native, surface, listed, !listed)
			}
		}
	}
}

// surfacesProvider serves a fixed set of native surfaces. It counts the
// calls that reach it, answers each with an empty object, and fails each
// stream.
type surfacesProvider struct {
	native []core.ModelSurface
	calls  atomic.Int32
}

func (p *surfacesProvider) NativeSurfaces(string) []core.ModelSurface { return p.native }

func (p *surfacesProvider) Invoke(context.Context, core.Request) (core.Response, error) {
	p.calls.Add(1)
	return core.Response{Body: []byte(`{}`), ContentType: core.ContentTypeJSON}, nil
}

func (p *surfacesProvider) Stream(context.Context, core.Request) (core.StreamIter, error) {
	p.calls.Add(1)
	return nil, errUpstream
}

func (p *surfacesProvider) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	return nil, nil
}

// Chat Completions round-trip through core's Anthropic provider, which
// serves only Messages, on recorded Messages answers.
func TestChatOverAnthropicMessagesRoundTrip(t *testing.T) {
	t.Parallel()
	answer, stream := fixture(t, "messages_tool_use.json"), fixture(t, "messages_tool_use.sse")
	var received []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" || r.Header.Get("x-api-key") != "test-key" {
			http.Error(w, `{"type":"error","error":{"type":"not_found_error","message":"unexpected request"}}`, http.StatusNotFound)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		received = append(received, body)
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, stream)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, answer)
	}))
	t.Cleanup(server.Close)
	anthropic, err := providers.NewAnthropic(providers.AnthropicConfig{BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	adapter := translation.Adapter{Provider: anthropic}
	credential := &core.Credential{APIKey: "test-key"}

	request := jsonRequest(core.ModelSurfaceChatCompletions, weatherChatRequest)
	request.Credential = credential
	response, err := adapter.Invoke(context.Background(), request)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	completion := decode(t, response.Body)
	choices, _ := completion["choices"].([]any)
	choice, _ := choices[0].(map[string]any)
	message, _ := choice["message"].(map[string]any)
	calls, _ := message["tool_calls"].([]any)
	if choice["finish_reason"] != "tool_calls" || len(calls) != 1 {
		t.Fatalf("completion = %v", completion)
	}
	function, _ := calls[0].(map[string]any)["function"].(map[string]any)
	assertWeatherCall(t, function["arguments"].(string))
	assertChatUsage(t, completion["usage"])

	streamRequest := chatStreamRequest(true)
	streamRequest.Credential = credential
	chunks, err := adapter.Stream(context.Background(), streamRequest)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer chunks.Close()
	frames, err := collect(t, chunks)
	if err != nil {
		t.Fatalf("stream error = %v", err)
	}
	streamed := assembleChat(t, frames)
	if streamed.content != "Checking the weather." || streamed.finish != "tool_calls" || streamed.calls[0] == nil || streamed.done != 1 {
		t.Fatalf("streamed = %+v", streamed)
	}
	assertWeatherCall(t, streamed.calls[0].arguments)
	assertChatUsage(t, streamed.usage)

	if len(received) != 2 || received[0]["stream"] == true || received[1]["stream"] != true {
		t.Fatalf("Anthropic received %d requests (%v), want one completion and one stream", len(received), received)
	}
	if tools, _ := received[0]["tools"].([]any); len(tools) != 1 {
		t.Fatalf("Anthropic request tools = %v", received[0]["tools"])
	}
}
