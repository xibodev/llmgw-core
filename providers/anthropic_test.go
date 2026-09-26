package providers_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/xibodev/llmgw-core/providers"
)

// sendAnthropic serves a Chat payload through Complete or Stream and returns
// the Messages request body the upstream received.
func sendAnthropic(t *testing.T, stream bool, payload map[string]any) map[string]any {
	t.Helper()
	bodies := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("the upstream request is not JSON: %v", err)
		}
		bodies <- body
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()
	provider := providers.NewAnthropicProvider("anthropic", server.URL, "fixture-key", server.Client())
	if !stream {
		if _, err := provider.Complete(context.Background(), "claude-fixture", payload, nil); err != nil {
			t.Fatalf("complete: %v", err)
		}
		return <-bodies
	}
	iter, err := provider.Stream(context.Background(), "claude-fixture", payload, nil)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer iter.Close()
	for {
		if _, err := iter.Next(); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatalf("stream frame: %v", err)
		}
	}
	return <-bodies
}

func TestAnthropicSendsSystemCacheBreakpointsAsBlocks(t *testing.T) {
	ephemeral := map[string]any{"type": "ephemeral"}
	for _, stream := range []bool{false, true} {
		body := sendAnthropic(t, stream, map[string]any{"messages": []map[string]any{
			{"role": "system", "content": []any{
				map[string]any{"type": "text", "text": "stable rules", "cache_control": ephemeral},
				map[string]any{"type": "text", "text": "per-request context"},
			}},
			{"role": "user", "content": "hi"},
		}})
		want := []any{
			map[string]any{"type": "text", "text": "stable rules", "cache_control": ephemeral},
			map[string]any{"type": "text", "text": "per-request context"},
		}
		if !reflect.DeepEqual(body["system"], want) {
			t.Fatalf("stream=%v: system=%#v, want %#v", stream, body["system"], want)
		}

		body = sendAnthropic(t, stream, map[string]any{"messages": []map[string]any{
			{"role": "system", "content": "be brief"},
			{"role": "user", "content": "hi"},
		}})
		if body["system"] != "be brief" {
			t.Fatalf("stream=%v: system=%#v, want the plain string without breakpoints", stream, body["system"])
		}
	}
}

// A Gemini thought signature has no place in a Messages request, and
// llm-translate reports dropping it as material. This provider has no way to
// report a loss and served such histories before, so it still serves them.
func TestAnthropicServesHistoriesWithThoughtSignatures(t *testing.T) {
	for _, stream := range []bool{false, true} {
		body := sendAnthropic(t, stream, map[string]any{"messages": []map[string]any{
			{"role": "user", "content": "look it up"},
			{"role": "assistant", "tool_calls": []any{map[string]any{
				"id": "call_1", "type": "function",
				"function":      map[string]any{"name": "lookup", "arguments": `{"q":"fixture"}`},
				"extra_content": map[string]any{"google": map[string]any{"thought_signature": "fixture-signature"}},
			}}},
			{"role": "tool", "tool_call_id": "call_1", "content": "found"},
		}})
		messages, _ := json.Marshal(body["messages"])
		if !strings.Contains(string(messages), `"tool_use"`) || strings.Contains(string(messages), "fixture-signature") {
			t.Fatalf("stream=%v: messages=%s, want the tool call without its signature", stream, messages)
		}
	}
}
