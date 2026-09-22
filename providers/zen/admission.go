package zen

import (
	"reflect"
	"strings"
)

type RequestKind string

const (
	RequestOrdinary RequestKind = "ordinary_assistant"
	RequestTitle    RequestKind = "title_generation"
)

// ClassifyChat recognizes only an explicit title-generator instruction in a
// system/developer message. Turn count and absence of tools never imply title
// intent, so ordinary first turns are preserved.
func ClassifyChat(messages []map[string]any) RequestKind {
	for _, message := range messages {
		role, _ := message["role"].(string)
		content, _ := message["content"].(string)
		if (role == "system" || role == "developer") && explicitTitleInstruction(content) {
			return RequestTitle
		}
	}
	return RequestOrdinary
}

func ClassifyResponses(payload map[string]any) RequestKind {
	instructions, _ := payload["instructions"].(string)
	if explicitTitleInstruction(instructions) {
		return RequestTitle
	}
	return RequestOrdinary
}

func explicitTitleInstruction(value string) bool {
	value = strings.ToLower(strings.TrimSpace(value))
	return value == "you are a title generator" || strings.HasPrefix(value, "you are a title generator.")
}

var compatibilityChatTools = []map[string]any{
	{
		"type": "function",
		"function": map[string]any{
			"name": "bash", "description": "Executes a given bash/powershell command.",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{
				"command": map[string]any{"type": "string", "description": "The command to execute"},
			}, "required": []any{"command"}},
		},
	},
	{
		"type": "function",
		"function": map[string]any{
			"name": "read", "description": "Read a file from the local filesystem.",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{
				"filePath": map[string]any{"type": "string", "description": "The absolute path to the file to read"},
			}, "required": []any{"filePath"}},
		},
	},
}

var compatibilityResponsesTools = []map[string]any{
	{
		"type": "function", "name": "bash", "description": "Executes a given bash/powershell command.",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{
			"command": map[string]any{"type": "string", "description": "The command to execute"},
		}, "required": []any{"command"}},
	},
	{
		"type": "function", "name": "read", "description": "Read a file from the local filesystem.",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{
			"filePath": map[string]any{"type": "string", "description": "The absolute path to the file to read"},
		}, "required": []any{"filePath"}},
	},
}

// AdmitChat restores the anonymous CLI's multi-turn compatibility contract
// without changing first-turn assistant requests or explicit title requests.
func AdmitChat(messages []map[string]any, payload map[string]any) map[string]any {
	out := cloneMap(payload)
	if ClassifyChat(messages) == RequestTitle {
		delete(out, "tools")
		delete(out, "tool_choice")
		return out
	}
	if conversationTurns(messages) <= 1 && !hasTools(out["tools"]) {
		return out
	}
	out["tools"] = ensureTools(out["tools"], compatibilityChatTools)
	if out["tool_choice"] == nil {
		out["tool_choice"] = "auto"
	}
	return out
}

// AdmitResponses is the Responses-wire equivalent of AdmitChat.
func AdmitResponses(payload map[string]any) map[string]any {
	out := cloneMap(payload)
	if ClassifyResponses(payload) == RequestTitle {
		delete(out, "tools")
		delete(out, "tool_choice")
		return out
	}
	input, _ := payload["input"].([]any)
	if len(input) <= 1 && !hasTools(out["tools"]) {
		return out
	}
	out["tools"] = ensureTools(out["tools"], compatibilityResponsesTools)
	if out["tool_choice"] == nil {
		out["tool_choice"] = "auto"
	}
	return out
}

func hasTools(value any) bool {
	if value == nil {
		return false
	}
	raw := reflect.ValueOf(value)
	return raw.Kind() == reflect.Slice && raw.Len() > 0
}

func conversationTurns(messages []map[string]any) int {
	turns := 0
	for _, message := range messages {
		role, _ := message["role"].(string)
		if role != "system" && role != "developer" {
			turns++
		}
	}
	return turns
}

func ensureTools(value any, compatibility []map[string]any) any {
	tools := toolSlice(value)
	names := map[string]bool{}
	for _, tool := range tools {
		names[toolName(tool)] = true
	}
	for _, tool := range compatibility {
		if !names[toolName(tool)] {
			tools = append(tools, tool)
		}
	}
	if _, ok := value.([]map[string]any); ok {
		mapped := make([]map[string]any, 0, len(tools))
		for _, raw := range tools {
			if tool, ok := raw.(map[string]any); ok {
				mapped = append(mapped, tool)
			}
		}
		return mapped
	}
	return tools
}

func toolSlice(value any) []any {
	if value == nil {
		return nil
	}
	raw := reflect.ValueOf(value)
	if raw.Kind() != reflect.Slice {
		return nil
	}
	tools := make([]any, raw.Len())
	for index := range tools {
		tools[index] = raw.Index(index).Interface()
	}
	return tools
}

func toolName(value any) string {
	tool, _ := value.(map[string]any)
	name, _ := tool["name"].(string)
	if function, ok := tool["function"].(map[string]any); ok {
		name, _ = function["name"].(string)
	}
	return name
}
