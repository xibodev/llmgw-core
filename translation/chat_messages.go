package translation

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"

	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
)

// defaultMessagesMaxTokens is the max_tokens a Messages request carries when
// the Chat request sets none, since Messages requires one.
const defaultMessagesMaxTokens = 8192

// chatMessages returns the messages of a Chat Completions request.
func chatMessages(payload map[string]any) ([]map[string]any, error) {
	rawMessages, _ := payload["messages"].([]any)
	messages := make([]map[string]any, 0, len(rawMessages))
	for _, raw := range rawMessages {
		message, ok := raw.(map[string]any)
		if !ok {
			return nil, errors.New("every Chat message must be an object")
		}
		messages = append(messages, message)
	}
	return messages, nil
}

// chatToMessages converts a Chat Completions request to an Anthropic
// Messages request. System and developer messages become the system prompt,
// tools and the tool choice convert, max_tokens or max_completion_tokens
// becomes max_tokens (defaultMessagesMaxTokens when neither is set), and
// temperature, top_p, top_k, stop and user map to their Messages fields.
// Every other field is dropped with a loss; see droppedChatField.
func chatToMessages(model string, payload map[string]any, stream bool) (map[string]any, []core.Loss, error) {
	messages, err := chatMessages(payload)
	if err != nil {
		return nil, nil, err
	}
	converted := translate.OpenAIMessagesToAnthropicWithReport(messages)
	losses := append([]core.Loss(nil), converted.Report.Losses...)
	body := map[string]any{"model": model, "messages": converted.Value.Messages, "max_tokens": defaultMessagesMaxTokens}
	switch {
	case len(converted.Value.SystemBlocks) > 0:
		body["system"] = converted.Value.SystemBlocks
	case converted.Value.System != "":
		body["system"] = converted.Value.System
	}
	if stream {
		body["stream"] = true
	}

	fields := make([]string, 0, len(payload))
	for field := range payload {
		fields = append(fields, field)
	}
	slices.Sort(fields)
	for _, field := range fields {
		value := payload[field]
		switch field {
		case "model", "messages", "stream", "stream_options", "tools", "tool_choice", "parallel_tool_calls":
			// Set above, or below with the fields they depend on. A
			// stream's include_usage is honoured by the stream itself.
		case "max_tokens":
			if limit, ok := value.(float64); ok && limit > 0 && payload["max_completion_tokens"] == nil {
				body["max_tokens"] = limit
			}
		case "max_completion_tokens":
			if limit, ok := value.(float64); ok && limit > 0 {
				body["max_tokens"] = limit
				losses = append(losses, core.Loss{Path: field, Class: translate.LossRenamed, Severity: translate.LossAdvisory, Detail: "converted to max_tokens"})
			}
		case "temperature":
			if temperature, ok := value.(float64); ok {
				if temperature > 1 {
					temperature = 1
					losses = append(losses, core.Loss{Path: field, Class: translate.LossApproximated, Severity: translate.LossAdvisory, Detail: "Messages temperature ranges from 0 to 1; clamped to 1"})
				}
				body["temperature"] = temperature
			}
		case "top_p", "top_k":
			if value != nil {
				body[field] = value
			}
		case "stop":
			switch stop := value.(type) {
			case string:
				if stop != "" {
					body["stop_sequences"] = []any{stop}
				}
			case []any:
				if len(stop) > 0 {
					body["stop_sequences"] = stop
				}
			case nil:
			default:
				losses = append(losses, core.Loss{Path: field, Class: translate.LossUnsupported, Severity: translate.LossMaterial, Detail: "stop must be a string or a list of strings"})
			}
		case "user":
			if user, ok := value.(string); ok && user != "" {
				body["metadata"] = map[string]any{"user_id": user}
			}
		default:
			if loss, dropped := droppedChatField(field, value); dropped {
				losses = append(losses, loss)
			}
		}
	}

	tools, toolLosses := messagesTools(payload["tools"])
	losses = append(losses, toolLosses...)
	if len(tools) > 0 {
		body["tools"] = tools
		choice, choiceLosses := messagesToolChoice(payload["tool_choice"])
		losses = append(losses, choiceLosses...)
		if parallel, ok := payload["parallel_tool_calls"].(bool); ok && !parallel {
			if choice == nil {
				choice = map[string]any{"type": "auto"}
			}
			if choice["type"] != "none" {
				choice["disable_parallel_tool_use"] = true
			}
		}
		if choice != nil {
			body["tool_choice"] = choice
		}
	}
	return body, losses, nil
}

// messagesTools converts Chat tools to Messages tools. A tool that is not a
// named function cannot be converted and is dropped with a material loss.
func messagesTools(raw any) ([]any, []core.Loss) {
	tools, _ := raw.([]any)
	var converted []any
	var losses []core.Loss
	for index, tool := range tools {
		one := translate.OpenAIToolsToAnthropic([]any{tool})
		if len(one) == 0 {
			losses = append(losses, core.Loss{Path: "tools." + strconv.Itoa(index), Class: translate.LossUnsupported, Severity: translate.LossMaterial, Detail: "only named function tools convert to Messages tools"})
			continue
		}
		converted = append(converted, one...)
	}
	return converted, losses
}

// messagesToolChoice converts a Chat tool_choice to a Messages tool_choice:
// auto, required (any), none, or one named function (tool). nil means none
// was set.
func messagesToolChoice(raw any) (map[string]any, []core.Loss) {
	switch choice := raw.(type) {
	case nil:
		return nil, nil
	case string:
		switch choice {
		case "auto":
			return map[string]any{"type": "auto"}, nil
		case "required":
			return map[string]any{"type": "any"}, nil
		case "none":
			return map[string]any{"type": "none"}, nil
		}
	case map[string]any:
		if function, ok := choice["function"].(map[string]any); ok {
			if name, _ := function["name"].(string); name != "" {
				return map[string]any{"type": "tool", "name": name}, nil
			}
		}
	}
	return nil, []core.Loss{{Path: "tool_choice", Class: translate.LossUnsupported, Severity: translate.LossMaterial, Detail: "tool_choice is not representable by Messages"}}
}

// materialChatField returns why Messages cannot carry a Chat field whose
// loss changes what the answer is or how it is shaped. Dropping any other
// field only changes how the answer is produced, billed or recorded, such as
// seed, store, metadata, service_tier, prompt_cache_key or reasoning_effort,
// so its loss is advisory.
func materialChatField(field string) (string, bool) {
	switch field {
	case "n":
		return "Messages returns one choice", true
	case "response_format":
		return "Messages has no response format", true
	case "logprobs", "top_logprobs":
		return "Messages returns no log probabilities", true
	case "logit_bias":
		return "Messages has no logit bias", true
	case "presence_penalty":
		return "Messages has no presence penalty", true
	case "frequency_penalty":
		return "Messages has no frequency penalty", true
	case "modalities", "audio":
		return "Messages answers with text only", true
	case "prediction":
		return "Messages has no predicted output", true
	case "functions":
		return "legacy functions do not convert; send tools", true
	case "function_call":
		return "legacy function_call does not convert; send tool_choice", true
	case "web_search_options":
		return "Messages has no built-in web search option", true
	}
	return "", false
}

// droppedChatField returns the loss of dropping a Chat field Messages cannot
// carry, and false for a value that asks for nothing, such as n=1, a text
// response format, text-only modalities, a zero penalty or false.
func droppedChatField(field string, value any) (core.Loss, bool) {
	if chatFieldIsNoop(field, value) {
		return core.Loss{}, false
	}
	if reason, material := materialChatField(field); material {
		return core.Loss{Path: field, Class: translate.LossDropped, Severity: translate.LossMaterial, Detail: reason}, true
	}
	return core.Loss{Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: fmt.Sprintf("Messages does not carry %s", field)}, true
}

// chatFieldIsNoop reports a value that asks for nothing. Zero is nothing
// only where it is the default that turns the feature off, so a seed of 0
// still asks for a seed; and an empty web_search_options turns web search on.
func chatFieldIsNoop(field string, value any) bool {
	switch typed := value.(type) {
	case nil:
		return true
	case bool:
		return !typed
	case float64:
		switch field {
		case "n":
			return typed == 1
		case "presence_penalty", "frequency_penalty", "top_logprobs":
			return typed == 0
		}
	case string:
		return typed == ""
	case []any:
		if field == "modalities" {
			return len(typed) == 0 || (len(typed) == 1 && typed[0] == "text")
		}
		return len(typed) == 0
	case map[string]any:
		switch field {
		case "response_format":
			return typed["type"] == "text"
		case "web_search_options":
			return false
		}
		return len(typed) == 0
	}
	return false
}

// messagesToChatResponse converts a Messages answer to a Chat completion,
// with its usage counted as a stream's is.
func messagesToChatResponse(model string, response map[string]any) (map[string]any, []core.Loss) {
	converted := translate.AnthropicResponseToOpenAIWithReport(response, model)
	if reported, ok := response["usage"].(map[string]any); ok && converted.Value != nil {
		var usage messagesUsage
		usage.update(reported)
		converted.Value["usage"] = usage.chat()
	}
	return converted.Value, withoutThinkingBlockLosses(response, converted.Report.Losses)
}

// withoutThinkingBlockLosses keeps one loss per thinking block of an answer.
// llm-translate reports a dropped thinking block twice: as its advisory
// reasoning loss, and as a content block Chat cannot represent, which is
// material. A stream reports only the reasoning loss, so without this an
// answer that thought would fail under a policy its stream passes.
func withoutThinkingBlockLosses(response map[string]any, losses []core.Loss) []core.Loss {
	blocks, _ := response["content"].([]any)
	thinking := map[string]bool{}
	for index, raw := range blocks {
		block, _ := raw.(map[string]any)
		if kind := block["type"]; kind == "thinking" || kind == "redacted_thinking" {
			thinking["content."+strconv.Itoa(index)] = true
		}
	}
	if len(thinking) == 0 {
		return losses
	}
	kept := losses[:0:0]
	for _, loss := range losses {
		if thinking[loss.Path] && loss.Severity == translate.LossMaterial {
			continue
		}
		kept = append(kept, loss)
	}
	return kept
}

// messagesUsage is the usage a Messages answer reports. Its input tokens
// leave out the input written to and read from the cache, which it counts
// apart.
type messagesUsage struct {
	input, cacheCreation, cacheRead, output int
}

// update reads the counts usage carries and keeps the others. A stream's
// counts are cumulative, and its message_delta often carries only the
// output tokens.
func (u *messagesUsage) update(usage map[string]any) {
	for key, count := range map[string]*int{
		"input_tokens":                &u.input,
		"cache_creation_input_tokens": &u.cacheCreation,
		"cache_read_input_tokens":     &u.cacheRead,
		"output_tokens":               &u.output,
	} {
		if value, ok := usage[key]; ok {
			*count = usageInt(value)
		}
	}
}

// chat is the usage as Chat reports it: the prompt tokens count the cached
// input, and cached_tokens is the part read from the cache.
func (u messagesUsage) chat() map[string]any {
	prompt := u.input + u.cacheCreation + u.cacheRead
	return map[string]any{
		"prompt_tokens":         prompt,
		"completion_tokens":     u.output,
		"total_tokens":          prompt + u.output,
		"prompt_tokens_details": map[string]any{"cached_tokens": u.cacheRead},
	}
}

func usageInt(value any) int {
	switch number := value.(type) {
	case float64:
		return int(number)
	case json.Number:
		parsed, _ := number.Int64()
		return int(parsed)
	case int:
		return number
	}
	return 0
}
