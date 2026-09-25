package providers

import (
	"encoding/json"
	"fmt"
)

// ollamaCompletion converts an /api/chat answer to a Chat completion as the
// gateway does. The completion's ID is always chatcmpl-ollama, and its usage
// comes from Ollama's evaluation counts. An answer that is not a chat reply,
// or one with neither content nor tool calls, cannot be used.
func ollamaCompletion(model string, raw []byte) ([]byte, error) {
	var data map[string]any
	err := json.Unmarshal(raw, &data)
	message, ok := data["message"].(map[string]any)
	if err != nil || !ok {
		return nil, unusableResponse("the Ollama response is not a chat reply", err)
	}
	content, _ := message["content"].(string)
	toolCalls := ollamaToolCalls(message["tool_calls"])
	if content == "" && toolCalls == nil {
		return nil, unusableResponse("Ollama returned an empty response", nil)
	}
	reply := map[string]any{"role": "assistant", "content": content}
	finish := "stop"
	if toolCalls != nil {
		reply["tool_calls"] = toolCalls
		finish = "tool_calls"
	}
	prompt, completion := codexChatInt(data["prompt_eval_count"]), codexChatInt(data["eval_count"])
	body, err := json.Marshal(map[string]any{
		"id": "chatcmpl-ollama", "object": "chat.completion", "model": model,
		"choices": []any{map[string]any{"index": 0, "message": reply, "finish_reason": finish}},
		"usage":   map[string]any{"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
	})
	if err != nil {
		return nil, unusableResponse("the Ollama response could not be converted", err)
	}
	return body, nil
}

// ollamaToolCalls converts Ollama's tool calls to Chat's as the gateway
// does: arguments that are not a string are encoded as JSON, and a call
// without an ID is named call_<index>. No calls is nil.
func ollamaToolCalls(raw any) []any {
	list, ok := raw.([]any)
	if !ok || len(list) == 0 {
		return nil
	}
	var mapped []any
	for index, item := range list {
		call, _ := item.(map[string]any)
		function, _ := call["function"].(map[string]any)
		arguments, ok := function["arguments"].(string)
		if !ok {
			encoded, _ := json.Marshal(function["arguments"])
			arguments = string(encoded)
		}
		id, _ := call["id"].(string)
		if id == "" {
			id = fmt.Sprintf("call_%d", index)
		}
		name, _ := function["name"].(string)
		mapped = append(mapped, map[string]any{
			"index": index, "id": id, "type": "function",
			"function": map[string]any{"name": name, "arguments": arguments},
		})
	}
	return mapped
}
