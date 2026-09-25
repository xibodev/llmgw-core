package providers

import (
	"context"
	"encoding/json"
	"io"
	"maps"
	"slices"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers/zen"
)

// zenChatForwarded reports a Chat field the gateway's OpenAI-compatible
// transport forwards to Zen, besides the model, messages and stream flag.
func zenChatForwarded(field string) bool {
	switch field {
	case "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "tools", "tool_choice",
		"reasoning_effort", "stream_options", "metadata", "parallel_tool_calls", "thinking":
		return true
	}
	return false
}

// zenChat splits a Chat body into its messages and options, as the gateway
// holds a Chat request. The request and the operation decide the model and
// the stream flag, so the body's are dropped.
//
// llm-translate carries a Responses request's max_output_tokens into Chat
// as _max_output_tokens, and the gateway's transport sends it as
// max_completion_tokens unless the request sets a limit of its own, so a
// Responses request a translation.Adapter serves over Chat keeps its limit.
func zenChat(payload map[string]any) ([]map[string]any, map[string]any, error) {
	messages, err := codexMessages(payload["messages"])
	if err != nil {
		return nil, nil, &core.ProviderError{Message: "the OpenCode Zen Chat messages are invalid: " + err.Error(), Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	options := make(map[string]any, len(payload))
	for key, value := range payload {
		if key != "model" && key != "messages" && key != "stream" {
			options[key] = value
		}
	}
	if limit := options["_max_output_tokens"]; limit != nil {
		delete(options, "_max_output_tokens")
		if options["max_tokens"] == nil && options["max_completion_tokens"] == nil {
			options["max_completion_tokens"] = limit
		}
	}
	return messages, options, nil
}

// zenAdmitChat applies the anonymous admission to a Chat request.
func zenAdmitChat(messages []map[string]any, options map[string]any) ([]map[string]any, map[string]any) {
	admitted := zen.AdmitChat(messages, options)
	if admittedMessages, ok := admitted["messages"].([]map[string]any); ok {
		messages = admittedMessages
		delete(admitted, "messages")
	}
	return messages, admitted
}

// zenChatBody is the body the gateway's transport sends to the Chat
// endpoint: the model, the messages, the stream flag and the forwarded
// fields that are set. Any other field is dropped and reported as an
// advisory loss, except the gateway's internal fields, prefixed "_", which
// never reach a provider.
func zenChatBody(model string, messages []map[string]any, options map[string]any, stream bool) (map[string]any, []core.Loss) {
	body := map[string]any{"model": model, "messages": messages, "stream": stream}
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(options)) {
		value := options[field]
		switch {
		case value == nil || strings.HasPrefix(field, "_"):
		case zenChatForwarded(field):
			body[field] = value
		default:
			losses = append(losses, core.Loss{
				Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory,
				Detail: "the OpenCode Zen Chat transport does not forward this field",
			})
		}
	}
	return body, losses
}

// postChat sends a Chat body and returns the open response of a request
// Zen accepted.
func (p *Zen) postChat(ctx context.Context, call zenCall, messages []map[string]any, body map[string]any) (io.ReadCloser, error) {
	header := zenHeaders(call.access, call.identity)
	zenVisionHeader(header, zenChatHasImages(messages))
	response, err := p.post(ctx, "/chat/completions", header, body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode >= 400 {
		raw, _ := readZenResponse(ctx, response)
		return nil, p.statusError(ctx, response, raw, call)
	}
	return response.Body, nil
}

// invokeChat serves Chat on the Chat endpoint. Anonymous access streams
// even a completion, as the gateway does, and assembles it.
func (p *Zen) invokeChat(ctx context.Context, call zenCall) (core.Response, error) {
	messages, options, err := zenChat(call.payload)
	if err != nil {
		return core.Response{}, err
	}
	if call.access.anonymous {
		// The gateway admits an anonymous completion twice: once as a
		// completion and again as the stream it reads it from. The second
		// admission only drops an empty tools list and a tool choice from
		// a first turn, and the request must stay the same.
		messages, options = zenAdmitChat(messages, options)
		messages, options = zenAdmitChat(messages, options)
	}
	body, losses := zenChatBody(call.model, messages, options, call.access.anonymous)
	stream, err := p.postChat(ctx, call, messages, body)
	if err != nil {
		return core.Response{}, err
	}
	defer stream.Close()
	var result []byte
	if call.access.anonymous {
		result, err = p.assembleChat(ctx, stream, call.model)
	} else {
		result, err = readZenChat(ctx, stream)
	}
	if err != nil {
		return core.Response{Losses: losses}, err
	}
	return core.Response{Body: result, ContentType: core.ContentTypeJSON, Losses: losses}, nil
}

// streamChat streams Chat from the Chat endpoint, Zen's records unchanged.
func (p *Zen) streamChat(ctx context.Context, call zenCall) (core.StreamIter, error) {
	messages, options, err := zenChat(call.payload)
	if err != nil {
		return nil, err
	}
	if call.access.anonymous {
		messages, options = zenAdmitChat(messages, options)
	}
	body, losses := zenChatBody(call.model, messages, options, true)
	stream, err := p.postChat(ctx, call, messages, body)
	if err != nil {
		return nil, err
	}
	return &zenStream{events: &zenChatStream{ctx: ctx, body: stream, reader: newZenSSEReader(stream)}, losses: losses}, nil
}

// readZenChat reads a Chat completion and returns it unchanged once it
// checks out as the gateway checks it.
func readZenChat(ctx context.Context, body io.Reader) ([]byte, error) {
	raw, err := readZenBody(ctx, body)
	if err != nil {
		return nil, err
	}
	var result map[string]any
	if json.Unmarshal(raw, &result) != nil || len(result) == 0 {
		return nil, zenUpstreamError("OpenCode Zen returned a completion that is not JSON", false, nil)
	}
	if zenSoftError(result) {
		return nil, zenUpstreamError("OpenCode Zen returned an error in a successful response", true, nil)
	}
	choices, _ := result["choices"].([]any)
	if len(choices) == 0 {
		return nil, zenUpstreamError("OpenCode Zen returned a completion without choices", false, nil)
	}
	if _, ok := choices[0].(map[string]any); !ok {
		return nil, zenUpstreamError("OpenCode Zen returned an invalid completion choice", false, nil)
	}
	return raw, nil
}

// zenSoftError reports an error a successful completion carries: a
// nonblank error string, an error object with a nonblank message or none.
func zenSoftError(response map[string]any) bool {
	raw, exists := response["error"]
	if !exists || raw == nil {
		return false
	}
	if message, ok := raw.(string); ok {
		return strings.TrimSpace(message) != ""
	}
	if details, ok := raw.(map[string]any); ok {
		if message, ok := details["message"].(string); ok {
			return strings.TrimSpace(message) != ""
		}
	}
	return true
}

// assembleChat reads a Chat stream into one completion, as the gateway
// answers an anonymous Chat request: the first id, creation time and model
// a chunk carries, the last usage, and the first choice's role, content
// and reasoning. Like the gateway it keeps no tool call and finishes with
// "stop". A stream without content fails, and may be retried.
func (p *Zen) assembleChat(ctx context.Context, body io.Reader, model string) ([]byte, error) {
	reader := newZenSSEReader(body)
	var id, responseModel string
	var created float64
	var content, reasoning strings.Builder
	var usage map[string]any
	role := "assistant"
	for {
		record, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, zenStreamFailure(ctx, err)
		}
		var chunk map[string]any
		if record.data == "[DONE]" || json.Unmarshal([]byte(record.data), &chunk) != nil {
			continue
		}
		if value, ok := chunk["id"].(string); ok && id == "" {
			id = value
		}
		if value, ok := chunk["created"].(float64); ok && created == 0 {
			created = value
		}
		if value, ok := chunk["model"].(string); ok && responseModel == "" {
			responseModel = value
		}
		if value, ok := chunk["usage"].(map[string]any); ok && value != nil {
			usage = value
		}
		choices, _ := chunk["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		choice, _ := choices[0].(map[string]any)
		delta, ok := choice["delta"].(map[string]any)
		if !ok {
			continue
		}
		if value, ok := delta["role"].(string); ok && value != "" {
			role = value
		}
		if value, ok := delta["content"].(string); ok {
			content.WriteString(value)
		}
		if value, ok := delta["reasoning"].(string); ok {
			reasoning.WriteString(value)
		} else if value, ok := delta["reasoning_content"].(string); ok {
			reasoning.WriteString(value)
		}
	}
	if id == "" {
		id = "chatcmpl-" + model
	}
	if created == 0 {
		created = float64(p.now().Unix())
	}
	if responseModel == "" {
		responseModel = model
	}
	text := content.String()
	if strings.TrimSpace(text) == "" && reasoning.Len() > 0 {
		text = reasoning.String()
	}
	if strings.TrimSpace(text) == "" {
		return nil, zenUpstreamError("the OpenCode Zen stream produced no content", true, nil)
	}
	message := map[string]any{"role": role, "content": text}
	if reasoning.Len() > 0 {
		message["reasoning_content"] = reasoning.String()
	}
	completion := map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": responseModel,
		"choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": "stop"}},
	}
	if usage != nil {
		completion["usage"] = usage
	}
	return json.Marshal(completion)
}
