package providers

import (
	"fmt"
	"strconv"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// ollamaMessages normalizes Chat messages as the gateway does for Ollama and
// reports what that loses at each message's path. A developer message is
// sent as a system one, and a message without a role as a user one.
// Content that is not a string is sent as fmt prints it, which is how the
// gateway sends an image: inside text, never as an image. A message with
// no content, tool calls or tool call ID is dropped, and of the others only
// the role, content, tool_calls, tool_call_id and name are sent.
func ollamaMessages(messages []map[string]any) ([]map[string]any, []core.Loss) {
	out := []map[string]any{}
	var losses []core.Loss
	for index, message := range messages {
		path := "messages." + strconv.Itoa(index)
		role, _ := message["role"].(string)
		if role == "" {
			role = "user"
		}
		content := ""
		if text, ok := message["content"].(string); ok {
			content = text
		} else if message["content"] != nil {
			content = fmt.Sprintf("%v", message["content"])
		}
		toolCalls := message["tool_calls"]
		toolCallID, _ := message["tool_call_id"].(string)
		if strings.TrimSpace(content) == "" && toolCalls == nil && toolCallID == "" {
			losses = append(losses, core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Ollama does not send a message without content, tool calls or a tool call ID"})
			continue
		}
		if role == "developer" {
			role = "system"
			losses = append(losses, core.Loss{Path: path + ".role", Class: translate.LossApproximated, Severity: translate.LossAdvisory, Detail: "Ollama receives a developer message as a system message"})
		}
		if _, ok := message["content"].(string); !ok && message["content"] != nil {
			losses = append(losses, ollamaContentLosses(path+".content", message["content"])...)
		}
		sent := map[string]any{"role": role, "content": content}
		if toolCalls != nil {
			sent["tool_calls"] = toolCalls
		}
		if toolCallID != "" {
			sent["tool_call_id"] = toolCallID
		}
		if name, ok := message["name"].(string); ok && name != "" {
			sent["name"] = name
		}
		for field, value := range message {
			switch field {
			case "role", "content", "tool_calls", "tool_call_id", "name":
				continue
			}
			if value != nil {
				losses = append(losses, core.Loss{Path: path + "." + field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Ollama does not send this message field"})
			}
		}
		out = append(out, sent)
	}
	return out, losses
}

// ollamaContentLosses reports content that is not a string. Ollama receives
// it printed as text, and each part of it that is not text, such as an
// image, only as the text that prints it.
func ollamaContentLosses(path string, content any) []core.Loss {
	losses := []core.Loss{{Path: path, Class: translate.LossApproximated, Severity: translate.LossMaterial, Detail: "Ollama receives content that is not a string printed as text"}}
	parts, _ := content.([]any)
	for index, raw := range parts {
		if part, ok := raw.(map[string]any); ok && part["type"] != "text" {
			losses = append(losses, core.Loss{Path: path + "." + strconv.Itoa(index), Class: translate.LossDropped, Severity: translate.LossMaterial, Detail: "Ollama receives this part only as the text that prints it"})
		}
	}
	return losses
}
