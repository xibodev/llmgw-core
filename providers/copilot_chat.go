package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"mime"
	"slices"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// copilotChat is a Chat request Copilot has checked and shaped: the body for
// Chat Completions or, for a model served over Responses, the Responses
// request that serves it.
type copilotChat struct {
	payload   map[string]any
	responses bool
	vision    bool
	// marked is an explicit force_api_support: true, which the gateway
	// answers with its adaptation marker in the body.
	marked bool
	losses []core.Loss
}

// copilotOutputLimit is where llm-translate carries a Responses request's
// max_output_tokens when it converts the request to Chat, and so where a
// translation.Adapter leaves it in the Chat body. The gateway sends it as
// max_completion_tokens unless the request limits its output already.
const copilotOutputLimit = "_max_output_tokens"

// prepareChat shapes a Chat request exactly as the gateway's Copilot
// transport does. It sends the messages and the fields that transport
// forwards, the OpenAI-compatible transport's openAIChatForwarded, and drops
// every other field, reporting it at its path; see copilotDropped. The
// gateway drops those too, so the request is still served.
//
// With adaptation on, a model whose catalog row lists Responses but not
// Chat Completions is served over Responses, and a model that lists
// reasoning efforts takes max_tokens as max_completion_tokens. A boolean
// force_api_support turns adaptation on or off for the request.
func (p *Copilot) prepareChat(request core.Request, stream bool) (copilotChat, error) {
	body, err := copilotPayload(request)
	if err != nil {
		return copilotChat{}, err
	}
	messages, ok := body["messages"].([]any)
	if body["messages"] == nil {
		messages, ok = []any{}, true
	}
	for _, message := range messages {
		if _, object := message.(map[string]any); !object {
			ok = false
		}
	}
	if !ok {
		return copilotChat{}, copilotInvalid("Chat messages must be an array of objects", nil)
	}
	adapt, marked := p.adapt, false
	switch value := body["force_api_support"].(type) {
	case nil:
	case bool:
		adapt, marked = value, value
	default:
		return copilotChat{}, copilotInvalid("force_api_support must be a boolean", nil)
	}
	options := make(map[string]any, len(body))
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(body)) {
		value := body[field]
		switch {
		case value == nil, field == "model", field == "stream", field == "messages", field == "force_api_support", field == copilotOutputLimit:
		case openAIChatForwarded(field):
			options[field] = value
		default:
			losses = append(losses, copilotDropped(field, value, "the Copilot transport does not carry this Chat field"))
		}
	}
	if limit := body[copilotOutputLimit]; limit != nil && options["max_tokens"] == nil && options["max_completion_tokens"] == nil {
		options["max_completion_tokens"] = limit
	}
	route, known := p.route(request.Model)
	if adapt && known && route.preferred() == "responses" {
		return copilotChatOverResponses(request.Model, messages, options, marked, losses)
	}
	if maxTokens, present := options["max_tokens"]; present && adapt && known && route.reasoningEffort {
		options["max_completion_tokens"] = maxTokens
		delete(options, "max_tokens")
		losses = append(losses, core.Loss{
			Path: "max_tokens", Class: translate.LossRenamed, Severity: translate.LossAdvisory,
			Detail: "sent as max_completion_tokens, which Copilot's reasoning models take",
		})
	}
	options["model"], options["messages"], options["stream"] = request.Model, messages, stream
	return copilotChat{payload: options, vision: copilotChatImages(messages), marked: marked, losses: translate.NewReport(losses...).Losses}, nil
}

// copilotChatOverResponses converts a Chat request into the Responses
// request that serves it, as the gateway's adaptation does. As in
// OpenAICompatible, only the fields the gateway's Chat facade forwards are
// converted, and the others are dropped and reported. The conversion's
// material losses refuse it before anything is sent, Gemini thought
// signatures excepted, and its other findings are reported.
func copilotChatOverResponses(model string, messages []any, options map[string]any, marked bool, losses []core.Loss) (copilotChat, error) {
	history, err := codexMessages(messages)
	if err != nil {
		return copilotChat{}, copilotInvalid("Chat messages must be an array of objects", err)
	}
	fields := make(map[string]any, len(options))
	for _, field := range slices.Sorted(maps.Keys(options)) {
		if openAIChatFacadeField(field) {
			fields[field] = options[field]
		} else {
			losses = append(losses, copilotDropped(field, options[field], "Chat served over Responses does not carry this field"))
		}
	}
	converted := translate.ChatToResponsesWithReport(model, history, fields, false)
	conversion := exemptThoughtSignatures(converted.Report.Losses)
	if err := translate.RejectMaterialLoss(translate.Report{Losses: conversion}); err != nil {
		return copilotChat{}, copilotUnsupported("the Chat request cannot be served over Copilot's Responses endpoint", err)
	}
	return copilotChat{
		payload: converted.Value, responses: true, vision: copilotResponsesImages(converted.Value["input"]),
		marked: marked, losses: translate.NewReport(append(losses, conversion...)...).Losses,
	}, nil
}

// copilotDropped reports a Chat field Copilot is not sent: advisory, or
// material for a field that changes the answer's structure, as the Codex
// policy classifies them.
func copilotDropped(field string, value any, detail string) core.Loss {
	severity := translate.LossAdvisory
	if structuralChatField(field) && !chatFieldDefault(field, value) {
		severity = translate.LossMaterial
	}
	return core.Loss{Path: field, Class: translate.LossDropped, Severity: severity, Detail: detail}
}

// copilotPayload decodes the JSON object body. Numbers decode as the
// gateway's Chat and Responses facades decode them, so they reach Copilot as
// the gateway renders them.
func copilotPayload(request core.Request) (map[string]any, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return nil, copilotInvalid("a Copilot request body must be JSON", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(request.Body))
	var payload map[string]any
	err = decoder.Decode(&payload)
	if err == nil {
		if _, trailing := decoder.Token(); trailing != io.EOF {
			err = errors.New("the body continues after its JSON object")
		}
	}
	if err != nil || payload == nil {
		return nil, copilotInvalid("the Copilot request body is not a JSON object", err)
	}
	return payload, nil
}

// copilotEncode renders a request body as the gateway does: sorted keys,
// HTML left unescaped and no trailing newline.
func copilotEncode(payload map[string]any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(payload); err != nil {
		return nil, copilotInvalid("the Copilot request could not be encoded", err)
	}
	return bytes.TrimRight(buffer.Bytes(), "\n"), nil
}

// copilotChatImages reports a Chat message with an image part, which
// Copilot accepts only with its vision header.
func copilotChatImages(messages []any) bool {
	for _, raw := range messages {
		message, _ := raw.(map[string]any)
		parts, _ := message["content"].([]any)
		for _, rawPart := range parts {
			part, _ := rawPart.(map[string]any)
			switch part["type"] {
			case "image_url", "input_image", "image":
				return true
			}
		}
	}
	return false
}

// copilotResponsesImages reports an image anywhere in a Responses input.
func copilotResponsesImages(value any) bool {
	switch current := value.(type) {
	case []any:
		return slices.ContainsFunc(current, copilotResponsesImages)
	case map[string]any:
		switch current["type"] {
		case "input_image", "image_url":
			return true
		}
		for _, nested := range current {
			if copilotResponsesImages(nested) {
				return true
			}
		}
	}
	return false
}

// copilotStripRejected removes the sampling parameters a 400 answer names,
// which the gateway then sends once more without, and reports each as a
// dropped field.
func copilotStripRejected(payload map[string]any, answer []byte) []core.Loss {
	lower := bytes.ToLower(answer)
	var losses []core.Loss
	for _, field := range []string{"temperature", "top_p"} {
		if _, present := payload[field]; present && bytes.Contains(lower, []byte(field)) {
			delete(payload, field)
			losses = append(losses, core.Loss{
				Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory,
				Detail: "Copilot rejected this sampling parameter for the model",
			})
		}
	}
	return losses
}
