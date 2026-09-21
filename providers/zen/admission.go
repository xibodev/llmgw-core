package zen

import "strings"

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

// AdmitChat adds Zen's minimum agent tools to an ordinary request without
// changing any message or replacing caller tools. Explicit title requests are
// returned unchanged.
func AdmitChat(messages []map[string]any, payload map[string]any) map[string]any {
	if ClassifyChat(messages) == RequestTitle {
		return cloneMap(payload)
	}
	return admitTools(payload, chatTool)
}

// AdmitResponses is the Responses-wire equivalent of AdmitChat.
func AdmitResponses(payload map[string]any) map[string]any {
	if ClassifyResponses(payload) == RequestTitle {
		return cloneMap(payload)
	}
	return admitTools(payload, responsesTool)
}

func admitTools(payload map[string]any, tool func(string) map[string]any) map[string]any {
	output := cloneMap(payload)
	rawTools := output["tools"]
	callerTools := make([]map[string]any, 0)
	callerToolCount := 0
	switch tools := rawTools.(type) {
	case []any:
		callerToolCount = len(tools)
		for _, raw := range tools {
			if entry, ok := raw.(map[string]any); ok {
				callerTools = append(callerTools, entry)
			}
		}
	case []map[string]any:
		callerToolCount = len(tools)
		callerTools = append(callerTools, tools...)
	}
	hasBash, hasRead := false, false
	for _, entry := range callerTools {
		name, _ := entry["name"].(string)
		if function, ok := entry["function"].(map[string]any); ok {
			if nested, ok := function["name"].(string); ok {
				name = nested
			}
		}
		hasBash = hasBash || name == "bash"
		hasRead = hasRead || name == "read"
	}
	if tools, ok := rawTools.([]map[string]any); ok {
		admitted := append([]map[string]any(nil), tools...)
		if !hasBash {
			admitted = append(admitted, tool("bash"))
		}
		if !hasRead {
			admitted = append(admitted, tool("read"))
		}
		output["tools"] = admitted
	} else {
		tools, _ := rawTools.([]any)
		admitted := append([]any(nil), tools...)
		if !hasBash {
			admitted = append(admitted, tool("bash"))
		}
		if !hasRead {
			admitted = append(admitted, tool("read"))
		}
		output["tools"] = admitted
	}
	if callerToolCount == 0 {
		if _, supplied := output["tool_choice"]; !supplied {
			output["tool_choice"] = "auto"
		}
	}
	return output
}

func chatTool(name string) map[string]any {
	return map[string]any{"type": "function", "function": functionTool(name)}
}

func responsesTool(name string) map[string]any {
	tool := functionTool(name)
	tool["type"] = "function"
	return tool
}

func functionTool(name string) map[string]any {
	property, description := "command", "Executes a given bash or PowerShell command."
	if name == "read" {
		property, description = "filePath", "Reads a file from the local filesystem."
	}
	return map[string]any{
		"name": name, "description": description,
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{property: map[string]any{"type": "string"}},
			"required":   []any{property},
		},
	}
}
