package providers

import (
	"context"
	"maps"
	"net/http"
	"slices"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// openAIOutputLimit is where llm-translate carries a Responses request's
// max_output_tokens when it converts the request to Chat, and so where a
// translation.Adapter leaves it in the Chat body. The gateway sends it as
// max_completion_tokens unless the request limits its output already.
const openAIOutputLimit = "_max_output_tokens"

// openAIChatForwarded reports a Chat field the gateway's OpenAI transport
// sends besides the model, the messages and the stream flag: the fields its
// payload builder copies when they are set.
func openAIChatForwarded(field string) bool {
	switch field {
	case "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "tools", "tool_choice",
		"reasoning_effort", "stream_options", "metadata", "parallel_tool_calls", "thinking":
		return true
	}
	return false
}

// openAIChatFacadeField reports a Chat field the gateway's Chat facade
// hands its transport. Only these reach its Responses conversion.
func openAIChatFacadeField(field string) bool {
	switch field {
	case "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "tools", "tool_choice",
		"metadata", "reasoning_effort":
		return true
	}
	return false
}

// openAIChat is a Chat request shaped for upstream: the Chat body, or the
// Responses request that serves it.
type openAIChat struct {
	messages  []any
	body      map[string]any
	responses bool
	// marked is an explicit force_api_support: true, which the gateway
	// answers with its adaptation marker in the body.
	marked bool
	losses []core.Loss
}

// chat shapes a Chat request as the gateway does: the model, the messages,
// the stream flag and the forwarded fields that are set, or every field
// with ForwardAllFields. A dropped field is reported as an advisory loss
// at its path. The gateway's own fields, prefixed "_", never reach the
// upstream, and neither does its force_api_support control.
//
// With adaptation on, a model whose row prefers Responses is served over
// Responses, and a model whose row lists reasoning efforts takes
// max_tokens as max_completion_tokens.
func (p *OpenAICompatible) chat(call openAICompatibleCall, stream bool) (openAIChat, error) {
	messages, ok := call.payload["messages"].([]any)
	if call.payload["messages"] == nil {
		messages, ok = []any{}, true
	}
	for _, message := range messages {
		if _, object := message.(map[string]any); !object {
			ok = false
		}
	}
	if !ok {
		return openAIChat{}, openAIInvalid("Chat messages must be an array of objects", nil)
	}
	adapt, marked := p.adapt, false
	switch value := call.payload["force_api_support"].(type) {
	case nil:
	case bool:
		adapt, marked = value, value
	default:
		return openAIChat{}, openAIInvalid("force_api_support must be a boolean", nil)
	}
	options := make(map[string]any, len(call.payload))
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(call.payload)) {
		value := call.payload[field]
		switch {
		case value == nil, field == "model", field == "messages", field == "stream", field == "force_api_support", strings.HasPrefix(field, "_"):
		case p.forwardAll || openAIChatForwarded(field):
			options[field] = value
		default:
			losses = append(losses, openAIDropped(field, "the "+p.label+" transport does not forward this Chat field"))
		}
	}
	if limit := call.payload[openAIOutputLimit]; limit != nil && options["max_tokens"] == nil && options["max_completion_tokens"] == nil {
		options["max_completion_tokens"] = limit
	}
	// Only adaptation reads the row, so a Chat request that is not adapted
	// costs the product no catalog lookup.
	if row, known := p.adaptedRow(call.model, adapt); known {
		if translate.PreferredEndpoint(row.SupportedAPIs) == "responses" {
			return p.chatOverResponses(call.model, messages, options, marked, losses)
		}
		_, reasoning := row.LegacyCapabilities["reasoning_effort"]
		if maxTokens, present := options["max_tokens"]; present && reasoning {
			options["max_completion_tokens"] = maxTokens
			delete(options, "max_tokens")
			losses = append(losses, core.Loss{
				Path: "max_tokens", Class: translate.LossRenamed, Severity: translate.LossAdvisory,
				Detail: "sent as max_completion_tokens, which reasoning models take",
			})
		}
	}
	options["model"], options["messages"], options["stream"] = call.model, messages, stream
	return openAIChat{messages: messages, body: options, marked: marked, losses: translate.NewReport(losses...).Losses}, nil
}

// chatOverResponses converts a Chat request into the Responses request that
// serves it, as the gateway's adaptation does. Only the fields its Chat
// facade forwards are converted, and the others are dropped as advisory
// losses. The conversion's material losses refuse the request before
// anything is sent, Gemini thought signatures excepted.
func (p *OpenAICompatible) chatOverResponses(model string, messages []any, options map[string]any, marked bool, losses []core.Loss) (openAIChat, error) {
	history, err := codexMessages(messages)
	if err != nil {
		return openAIChat{}, openAIInvalid("Chat messages must be an array of objects", err)
	}
	converted := make(map[string]any, len(options))
	for _, field := range slices.Sorted(maps.Keys(options)) {
		if openAIChatFacadeField(field) {
			converted[field] = options[field]
		} else {
			losses = append(losses, openAIDropped(field, "Chat served over Responses does not carry this field"))
		}
	}
	conversion := translate.ChatToResponsesWithReport(model, history, converted, false)
	conversionLosses := exemptThoughtSignatures(conversion.Report.Losses)
	if err := translate.RejectMaterialLoss(translate.Report{Losses: conversionLosses}); err != nil {
		return openAIChat{}, openAIUnsupported("the Chat request cannot be served over the "+p.label+" Responses endpoint", err)
	}
	return openAIChat{
		body: conversion.Value, responses: true, marked: marked,
		losses: translate.NewReport(append(losses, conversionLosses...)...).Losses,
	}, nil
}

// chatPath is where the upstream takes Chat. Pollinations serves it at
// /v1/chat/completions under a base without the version.
func (p *OpenAICompatible) chatPath() string {
	if p.registryID == "pollinations" {
		return "/v1/chat/completions"
	}
	return "/chat/completions"
}

func (p *OpenAICompatible) invokeChat(ctx context.Context, call openAICompatibleCall) (core.Response, error) {
	chat, err := p.chat(call, false)
	if err != nil {
		return core.Response{}, err
	}
	if chat.responses {
		return p.invokeChatOverResponses(ctx, call, chat)
	}
	response, err := p.post(ctx, p.chatPath(), p.header(call.access, "", openAIChatImages(chat.messages)), chat.body)
	if err != nil {
		return core.Response{Losses: chat.losses}, err
	}
	raw, err := p.read(ctx, response)
	if err != nil {
		return core.Response{Losses: chat.losses}, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		return core.Response{Losses: chat.losses}, p.refused(response, raw)
	}
	if err := p.checkChat(raw); err != nil {
		return core.Response{Losses: chat.losses}, err
	}
	return core.Response{Body: raw, ContentType: core.ContentTypeJSON, Losses: chat.losses}, nil
}

// streamChat opens a Chat stream. The gateway asks for it with the same
// headers as for a completion.
func (p *OpenAICompatible) streamChat(ctx context.Context, call openAICompatibleCall) (core.StreamIter, error) {
	chat, err := p.chat(call, true)
	if err != nil {
		return nil, err
	}
	if chat.responses {
		return p.streamChatOverResponses(ctx, call, chat)
	}
	response, err := p.post(ctx, p.chatPath(), p.header(call.access, "", openAIChatImages(chat.messages)), chat.body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		raw, _ := p.read(ctx, response)
		return nil, p.refused(response, raw)
	}
	return &openAIStream{StreamIter: newSSEFrameStream(ctx, p.label, response.Body), losses: chat.losses}, nil
}

func openAIDropped(field, detail string) core.Loss {
	return core.Loss{Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: detail}
}

// openAIUnsupported reports a request, or an answer, that a conversion
// cannot carry without a material loss. Another target may serve it
// natively.
func openAIUnsupported(message string, err error) *core.ProviderError {
	return &core.ProviderError{
		Message: message, Class: core.ProviderErrorUnsupported,
		Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
	}
}
