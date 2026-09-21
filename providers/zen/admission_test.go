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

func TestAdmitChatPreservesMessagesAndAddsOnlyMissingTools(t *testing.T) {
	messages := []map[string]any{{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "hello"}}
	custom := map[string]any{"type": "function", "function": map[string]any{"name": "search"}}
	payload := map[string]any{"tools": []any{custom}, "temperature": 0.2}
	beforeMessages := copyMessages(messages)
	admitted := AdmitChat(messages, payload)
	if !reflect.DeepEqual(messages, beforeMessages) || admitted["temperature"] != 0.2 || admitted["tool_choice"] != nil {
		t.Fatalf("messages=%v admitted=%v", messages, admitted)
	}
	tools, _ := admitted["tools"].([]any)
	if got := toolNames(tools); !reflect.DeepEqual(got, []string{"search", "bash", "read"}) {
		t.Fatalf("tools=%v", got)
	}
	if original, _ := payload["tools"].([]any); len(original) != 1 {
		t.Fatalf("caller payload mutated: %v", payload)
	}

	existing := map[string]any{"tools": []any{chatTool("bash"), responsesTool("read")}}
	admitted = AdmitChat(messages, existing)
	if got := toolNames(admitted["tools"].([]any)); !reflect.DeepEqual(got, []string{"bash", "read"}) {
		t.Fatalf("duplicate tools=%v", got)
	}
}

func TestAdmissionPreservesMapToolSlicesAndAddsOnlyMissingTools(t *testing.T) {
	chatSearch := map[string]any{"type": "function", "function": map[string]any{"name": "search"}}
	chatBash := chatTool("bash")
	chatPayload := map[string]any{"tools": []map[string]any{chatSearch, chatBash}}
	chat := AdmitChat([]map[string]any{{"role": "user", "content": "hello"}}, chatPayload)
	chatTools, ok := chat["tools"].([]map[string]any)
	if !ok || !reflect.DeepEqual(toolMapNames(chatTools), []string{"search", "bash", "read"}) {
		t.Fatalf("chat tools=%T %v", chat["tools"], chat["tools"])
	}
	if len(chatPayload["tools"].([]map[string]any)) != 2 {
		t.Fatalf("chat caller payload mutated: %v", chatPayload)
	}

	responsesSearch := map[string]any{"type": "function", "name": "search"}
	responsesRead := responsesTool("read")
	responsesPayload := map[string]any{"input": "hello", "tools": []map[string]any{responsesSearch, responsesRead}}
	responses := AdmitResponses(responsesPayload)
	responsesTools, ok := responses["tools"].([]map[string]any)
	if !ok || !reflect.DeepEqual(toolMapNames(responsesTools), []string{"search", "read", "bash"}) {
		t.Fatalf("responses tools=%T %v", responses["tools"], responses["tools"])
	}
	if len(responsesPayload["tools"].([]map[string]any)) != 2 {
		t.Fatalf("responses caller payload mutated: %v", responsesPayload)
	}
}

func TestAdmitOrdinaryNoToolRequestsWithoutPromptRewrite(t *testing.T) {
	messages := []map[string]any{{"role": "user", "content": "Explain this failure"}}
	admitted := AdmitChat(messages, map[string]any{"max_tokens": 32})
	if messages[0]["content"] != "Explain this failure" || admitted["tool_choice"] != "auto" || len(admitted["tools"].([]any)) != 2 {
		t.Fatalf("messages=%v admitted=%v", messages, admitted)
	}
	responses := AdmitResponses(map[string]any{"instructions": "Be concise", "input": "Explain this failure"})
	if responses["instructions"] != "Be concise" || responses["input"] != "Explain this failure" || responses["tool_choice"] != "auto" || len(responses["tools"].([]any)) != 2 {
		t.Fatalf("responses=%v", responses)
	}
}

func TestExplicitTitleRequestsRemainToolFreeAndUnchanged(t *testing.T) {
	chatPayload := map[string]any{"max_tokens": 16}
	titleMessages := []map[string]any{{"role": "system", "content": "You are a title generator. Output one title."}, {"role": "user", "content": "hello"}}
	if got := AdmitChat(titleMessages, chatPayload); !reflect.DeepEqual(got, chatPayload) || got["tools"] != nil {
		t.Fatalf("chat title=%v", got)
	}
	responsesPayload := map[string]any{"instructions": "You are a title generator.", "input": "hello"}
	if got := AdmitResponses(responsesPayload); !reflect.DeepEqual(got, responsesPayload) || got["tools"] != nil {
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
