package zen

import (
	"reflect"
	"testing"
)

func TestRequestClassificationRequiresExplicitTitleInstruction(t *testing.T) {
	ordinaryChat := []map[string]any{{"role": "user", "content": "Suggest a title generator"}}
	if got := ClassifyChat(ordinaryChat); got != RequestOrdinary {
		t.Fatalf("ordinary chat=%q", got)
	}
	if got := ClassifyChat([]map[string]any{{"role": "system", "content": "You help with titles"}}); got != RequestOrdinary {
		t.Fatalf("non-canonical system message=%q", got)
	}
	if got := ClassifyChat([]map[string]any{{"role": "developer", "content": "You are a title generator. Output one title."}}); got != RequestTitle {
		t.Fatalf("title chat=%q", got)
	}
	if got := ClassifyResponses(map[string]any{"instructions": "Be concise", "input": "hello"}); got != RequestOrdinary {
		t.Fatalf("ordinary responses=%q", got)
	}
	if got := ClassifyResponses(map[string]any{"instructions": " You are a title generator "}); got != RequestTitle {
		t.Fatalf("title responses=%q", got)
	}
}

func TestAdmitChatAddsCompatibilityToMultiTurnAndPreservesCallerChoice(t *testing.T) {
	messages := []map[string]any{{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "continue"}}
	custom := map[string]any{"type": "function", "function": map[string]any{"name": "search"}}
	payload := map[string]any{"tools": []any{custom}, "tool_choice": map[string]any{"type": "function", "function": map[string]any{"name": "search"}}, "temperature": 0.2}
	beforeMessages := copyMessages(messages)
	admitted := AdmitChat(messages, payload)
	if !reflect.DeepEqual(messages, beforeMessages) || admitted["temperature"] != payload["temperature"] || !reflect.DeepEqual(admitted["tool_choice"], payload["tool_choice"]) {
		t.Fatalf("messages=%v admitted=%v", messages, admitted)
	}
	tools, _ := admitted["tools"].([]any)
	if got := toolNames(tools); !reflect.DeepEqual(got, []string{"search", "bash", "read"}) {
		t.Fatalf("tools=%v", got)
	}
	if original, _ := payload["tools"].([]any); len(original) != 1 {
		t.Fatalf("caller payload mutated: %v", payload)
	}
}

func TestAdmissionPreservesMapToolSlices(t *testing.T) {
	chatSearch := map[string]any{"type": "function", "function": map[string]any{"name": "search"}}
	chatPayload := map[string]any{"tools": []map[string]any{chatSearch}}
	chat := AdmitChat([]map[string]any{{"role": "user", "content": "hello"}}, chatPayload)
	chatTools, ok := chat["tools"].([]map[string]any)
	if !ok || !reflect.DeepEqual(toolMapNames(chatTools), []string{"search", "bash", "read"}) {
		t.Fatalf("chat tools=%T %v", chat["tools"], chat["tools"])
	}
	if len(chatPayload["tools"].([]map[string]any)) != 1 {
		t.Fatalf("chat caller payload mutated: %v", chatPayload)
	}

	responsesSearch := map[string]any{"type": "function", "name": "search"}
	responsesPayload := map[string]any{"input": "hello", "tools": []map[string]any{responsesSearch}, "tool_choice": "required"}
	responses := AdmitResponses(responsesPayload)
	responsesTools, ok := responses["tools"].([]map[string]any)
	if !ok || !reflect.DeepEqual(toolMapNames(responsesTools), []string{"search", "bash", "read"}) || responses["tool_choice"] != "required" {
		t.Fatalf("responses tools=%T %v", responses["tools"], responses["tools"])
	}
	if len(responsesPayload["tools"].([]map[string]any)) != 1 {
		t.Fatalf("responses caller payload mutated: %v", responsesPayload)
	}
}

func TestAdmitOrdinaryNoToolRequestsWithoutPromptRewrite(t *testing.T) {
	messages := []map[string]any{{"role": "user", "content": "Explain this failure"}}
	admitted := AdmitChat(messages, map[string]any{"max_tokens": 32})
	admittedMessages := admitted["messages"].([]map[string]any)
	if messages[0]["content"] != "Explain this failure" || len(admittedMessages) != 2 || admittedMessages[0]["content"] != AnonymousAssistantPreamble || admitted["tool_choice"] != nil || admitted["tools"] != nil {
		t.Fatalf("messages=%v admitted=%v", messages, admitted)
	}
	responses := AdmitResponses(map[string]any{"instructions": "Be concise", "input": "Explain this failure"})
	if responses["instructions"] != AnonymousAssistantPreamble+"\n\nBe concise" || responses["input"] != "Explain this failure" || responses["tool_choice"] != nil || responses["tools"] != nil {
		t.Fatalf("responses=%v", responses)
	}
}

func TestAdmitMultiTurnDefaultsCompatibilityToolsAndAutoChoice(t *testing.T) {
	messages := []map[string]any{{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}, {"role": "user", "content": "three"}}
	chat := AdmitChat(messages, map[string]any{"max_tokens": 2048})
	if got := toolNames(chat["tools"].([]any)); !reflect.DeepEqual(got, []string{"bash", "read"}) || chat["tool_choice"] != "auto" {
		t.Fatalf("chat=%v", chat)
	}
	responses := AdmitResponses(map[string]any{"input": []any{map[string]any{"role": "user", "content": "one"}, map[string]any{"role": "assistant", "content": "two"}, map[string]any{"role": "user", "content": "three"}}})
	if got := toolNames(responses["tools"].([]any)); !reflect.DeepEqual(got, []string{"bash", "read"}) || responses["tool_choice"] != "auto" {
		t.Fatalf("responses=%v", responses)
	}
}

// TestAdmittedCompatibilityToolsAreNotShared pins that each admitted request
// gets its own tool definitions: mutating one request must not change the
// next.
func TestAdmittedCompatibilityToolsAreNotShared(t *testing.T) {
	messages := []map[string]any{{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}, {"role": "user", "content": "three"}}
	first := AdmitChat(messages, map[string]any{})
	tool := first["tools"].([]any)[0].(map[string]any)
	tool["type"] = "mutated"
	tool["function"].(map[string]any)["name"] = "mutated"
	second := AdmitChat(messages, map[string]any{})
	if got := toolNames(second["tools"].([]any)); !reflect.DeepEqual(got, []string{"bash", "read"}) || second["tools"].([]any)[0].(map[string]any)["type"] != "function" {
		t.Fatalf("a mutation of one admitted request leaked into the next: %v", second["tools"])
	}

	input := []any{map[string]any{"role": "user", "content": "one"}, map[string]any{"role": "user", "content": "two"}}
	responses := AdmitResponses(map[string]any{"input": input})
	responses["tools"].([]any)[0].(map[string]any)["name"] = "mutated"
	again := AdmitResponses(map[string]any{"input": input})
	if got := toolNames(again["tools"].([]any)); !reflect.DeepEqual(got, []string{"bash", "read"}) {
		t.Fatalf("a mutation of one admitted Responses request leaked into the next: %v", again["tools"])
	}
}

func TestExplicitTitleRequestsRemainToolFreeAndUnchanged(t *testing.T) {
	chatPayload := map[string]any{"max_tokens": 16}
	titleMessages := []map[string]any{{"role": "system", "content": "You are a title generator. Output one title."}, {"role": "user", "content": "hello"}}
	chatPayload["tools"] = []any{map[string]any{"type": "function", "function": map[string]any{"name": "search"}}}
	chatPayload["tool_choice"] = "auto"
	if got := AdmitChat(titleMessages, chatPayload); got["tools"] != nil || got["tool_choice"] != nil {
		t.Fatalf("chat title=%v", got)
	}
	responsesPayload := map[string]any{"instructions": "You are a title generator.", "input": "hello", "tools": []any{map[string]any{"type": "function", "name": "search"}}, "tool_choice": "auto"}
	if got := AdmitResponses(responsesPayload); got["tools"] != nil || got["tool_choice"] != nil {
		t.Fatalf("responses title=%v", got)
	}
}

func copyMessages(input []map[string]any) []map[string]any {
	output := make([]map[string]any, len(input))
	for index, message := range input {
		output[index] = cloneMap(message)
	}
	return output
}

func toolNames(tools []any) []string {
	names := make([]string, 0, len(tools))
	for _, raw := range tools {
		tool, _ := raw.(map[string]any)
		name, _ := tool["name"].(string)
		if function, ok := tool["function"].(map[string]any); ok {
			name, _ = function["name"].(string)
		}
		names = append(names, name)
	}
	return names
}

func toolMapNames(tools []map[string]any) []string {
	values := make([]any, len(tools))
	for index := range tools {
		values[index] = tools[index]
	}
	return toolNames(values)
}
