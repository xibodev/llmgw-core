package providers

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

func exemptThoughtSignatures(losses []core.Loss) []core.Loss {
	exempted := make([]core.Loss, len(losses))
	for i, loss := range losses {
		if loss.Class == translate.LossDropped && strings.HasSuffix(loss.Path, ".thought_signature") {
			loss.Severity = translate.LossAdvisory
		}
		exempted[i] = loss
	}
	return exempted
}

var codexErrorIdentifier = regexp.MustCompile("^[a-zA-Z0-9_.-]+$")

func codexMessages(value any) ([]map[string]any, error) {
	switch messages := value.(type) {
	case []map[string]any:
		return messages, nil
	case []any:
		out := make([]map[string]any, len(messages))
		for i, value := range messages {
			message, ok := value.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("message %d is not an object", i)
			}
			out[i] = message
		}
		return out, nil
	default:
		return nil, fmt.Errorf("messages must be an array")
	}
}

func anySlice(value any) []any {
	values, _ := value.([]any)
	return values
}

func codexErrorDiagnostic(raw []byte) string {
	var envelope struct {
		Error struct {
			Code  any    `json:"code"`
			Type  string `json:"type"`
			Param string `json:"param"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &envelope) != nil {
		return ""
	}
	values := []struct {
		name  string
		value string
	}{
		{name: "code", value: fmt.Sprint(envelope.Error.Code)},
		{name: "type", value: envelope.Error.Type},
		{name: "param", value: envelope.Error.Param},
	}
	parts := make([]string, 0, len(values))
	for _, value := range values {
		if value.value != "" && value.value != "<nil>" && codexErrorIdentifier.MatchString(value.value) {
			parts = append(parts, value.name+"="+value.value)
		}
	}
	return strings.Join(parts, ", ")
}

// structuralChatField reports a Chat field that changes the structure of
// the answer, which neither provider drops silently.
func structuralChatField(field string) bool {
	switch field {
	case "response_format", "n", "logprobs", "top_logprobs", "audio", "modalities", "prediction", "tool_choice", "parallel_tool_calls":
		return true
	}
	return false
}

// chatFieldDefault reports a structural field set to what providers do anyway:
// one choice, no log probabilities, parallel and automatic tool calls, text only.
func chatFieldDefault(field string, value any) bool {
	switch field {
	case "n":
		return codexChatInt(value) == 1
	case "logprobs", "parallel_tool_calls":
		enabled, ok := value.(bool)
		return ok && enabled == (field == "parallel_tool_calls")
	case "tool_choice":
		return value == "auto"
	case "modalities":
		list, ok := value.([]any)
		return ok && len(list) == 1 && list[0] == "text"
	case "response_format":
		format, ok := value.(map[string]any)
		return ok && format["type"] == "text"
	}
	return false
}

func codexChatInt(value any) int {
	switch number := value.(type) {
	case float64:
		return int(number)
	case int:
		return number
	case int64:
		return int(number)
	}
	return 0
}
