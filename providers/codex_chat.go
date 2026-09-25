package providers

import (
	"encoding/json"
	"maps"
	"slices"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// codexChatFields applies the gateway's Chat field policy to a Chat
// Completions body. It returns the payload the Codex conversion takes, which
// is the messages, the tools and a prompt cache key, and the fields it
// dropped as advisory losses at their paths.
//
// The official Codex client sends no length or sampling hints, so
// max_tokens, temperature, top_p, stop and every other field the Codex
// Responses transport cannot carry are dropped rather than refused. A field
// that changes the structure of the answer is refused, unless its value is
// what Codex does anyway, and then it is dropped too. Fields are read in
// order, so the refusal and the losses do not depend on map order.
func codexChatFields(body map[string]any) (map[string]any, []core.Loss, error) {
	chat := make(map[string]any, 3)
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(body)) {
		value := body[field]
		switch field {
		case "model", "stream":
			// The request's model and the operation decide these.
			continue
		case "messages":
			chat[field] = value
			continue
		case "tools":
			if value != nil {
				chat[field] = value
				continue
			}
		case "prompt_cache_key":
			if key, _ := value.(string); key != "" {
				chat[field] = key
				continue
			}
		}
		if value == nil {
			continue
		}
		detail := "the Codex Responses transport does not carry this Chat field"
		if codexStructuralChatField(field) {
			if !codexChatFieldDefault(field, value) {
				return nil, nil, &core.ProviderError{
					Message: "Chat field " + field + " is not supported by the Codex Responses transport",
					Class:   core.ProviderErrorUnsupported, Classification: core.ProviderErrorClassification{FailoverEligible: true},
				}
			}
			detail = "Codex already behaves as this value asks"
		}
		losses = append(losses, core.Loss{Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: detail})
	}
	return chat, losses, nil
}

// codexStructuralChatField reports a Chat field that changes the structure
// of the answer, which Codex must not drop silently.
func codexStructuralChatField(field string) bool {
	switch field {
	case "response_format", "n", "logprobs", "top_logprobs", "audio", "modalities", "prediction", "tool_choice", "parallel_tool_calls":
		return true
	}
	return false
}

// codexChatFieldDefault reports a structural field set to what the Codex
// transport does anyway: one choice, no log probabilities, parallel and
// automatic tool calls, text only.
func codexChatFieldDefault(field string, value any) bool {
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

// codexChatInt reads a JSON number as the gateway reads it, truncating a
// fraction.
func codexChatInt(value any) int {
	switch number := value.(type) {
	case float64:
		return int(number)
	case int:
		return number
	case int64:
		return int(number)
	case json.Number:
		if parsed, err := number.Int64(); err == nil {
			return int(parsed)
		}
		if parsed, err := number.Float64(); err == nil {
			return int(parsed)
		}
	}
	return 0
}
