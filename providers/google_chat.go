package providers

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// chat performs one Chat Completions request over generateContent and
// returns Google's answer reshaped as the gateway reshapes it.
func (p *Google) chat(ctx context.Context, request core.Request) (core.Response, error) {
	payload, losses, err := googleChatPayload(request)
	if err != nil {
		return core.Response{}, err
	}
	access, err := p.access(request.Credential)
	if err != nil {
		return core.Response{}, err
	}
	endpoint, err := p.modelURL(request.Model, access.project, "generateContent")
	if err != nil {
		return core.Response{}, err
	}
	authorization, err := p.authorization(access)
	if err != nil {
		return core.Response{}, err
	}
	decoded, err := p.post(ctx, authorization, endpoint, payload)
	if err != nil {
		return core.Response{}, err
	}
	if err := googleEmptyReplyError(p.label(), decoded); err != nil {
		return core.Response{}, err
	}
	body, err := json.Marshal(googleToOpenAIChat(request.Model, decoded))
	if err != nil {
		return core.Response{}, unusableResponse("the "+p.label()+" answer could not be encoded", err)
	}
	return core.Response{Body: body, ContentType: core.ContentTypeJSON, Losses: losses}, nil
}

// GenerateImages asks an image-capable Gemini model for inline images over
// generateContent, as the gateway does, and returns at most Count of them
// with Google's usage, whose modality-tagged token counts carry the image
// cost. A model that answers without image data, such as a text-only one,
// fails and permits failover.
func (p *Google) GenerateImages(ctx context.Context, request core.GenerateImagesRequest, credential *core.Credential) (core.GenerateImagesResult, error) {
	if strings.TrimSpace(request.Prompt) == "" {
		return core.GenerateImagesResult{}, &core.ProviderError{Message: p.label() + ": a prompt is required", Class: core.ProviderErrorInvalidRequest}
	}
	access, err := p.access(credential)
	if err != nil {
		return core.GenerateImagesResult{}, err
	}
	endpoint, err := p.modelURL(request.Model, access.project, "generateContent")
	if err != nil {
		return core.GenerateImagesResult{}, err
	}
	payload := googleContentRequest([]map[string]any{{"role": "user", "content": request.Prompt}}, nil, nil, []string{"TEXT", "IMAGE"})
	authorization, err := p.authorization(access)
	if err != nil {
		return core.GenerateImagesResult{}, err
	}
	decoded, err := p.post(ctx, authorization, endpoint, payload)
	if err != nil {
		return core.GenerateImagesResult{}, err
	}
	images := make([]core.GeneratedImage, 0, 1)
	for _, part := range googleParts(decoded) {
		inline, ok := part["inlineData"].(map[string]any)
		if !ok {
			continue
		}
		encoded, _ := inline["data"].(string)
		mimeType, _ := inline["mimeType"].(string)
		raw, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || len(raw) == 0 {
			continue
		}
		images = append(images, core.GeneratedImage{Data: raw, MimeType: mimeType})
		if request.Count > 0 && len(images) >= request.Count {
			break
		}
	}
	if len(images) == 0 {
		message := p.label() + ": the model returned no image data — it may be a text-only model"
		return core.GenerateImagesResult{}, &core.ProviderError{
			Message: message, Class: core.ProviderErrorUnsupported, Classification: core.ProviderErrorClassification{FailoverEligible: true},
			Cause: &InvocationError{Msg: message, FailoverEligible: true},
		}
	}
	usage, _ := decoded["usageMetadata"].(map[string]any)
	return core.GenerateImagesResult{Images: images, Usage: usage}, nil
}

// googleChatPayload maps a Chat Completions body to the generateContent
// body the gateway sends for it. The gateway's Chat facade hands Google the
// messages, max_tokens and temperature, and Google maps only those, so every
// other field is dropped and reported at its path: tools, and a field that
// changes the structure of the answer unless it asks for what Google does
// anyway, as material losses, and the rest as advisory ones. Only a
// message's role and string content reach Google, so a message's other
// fields are reported too, and so is content that is not a string, which
// Google receives as empty text.
func googleChatPayload(request core.Request) (map[string]any, []core.Loss, error) {
	body, err := googleJSONBody(request)
	if err != nil {
		return nil, nil, err
	}
	messages, err := googleChatMessages(body["messages"])
	if err != nil {
		return nil, nil, err
	}
	losses := append(googleChatFieldLosses(body), googleChatMessageLosses(messages)...)
	return googleContentRequest(messages, body["max_tokens"], body["temperature"], nil), translate.NewReport(losses...).Losses, nil
}

// googleChatMessages reads the messages as the gateway's Chat facade
// decodes them: an array of objects, where null is an empty message.
func googleChatMessages(value any) ([]map[string]any, error) {
	if value == nil {
		return nil, nil
	}
	list, ok := value.([]any)
	if !ok {
		return nil, &core.ProviderError{Message: "the Chat messages must be an array", Class: core.ProviderErrorInvalidRequest}
	}
	messages := make([]map[string]any, 0, len(list))
	for _, raw := range list {
		message, ok := raw.(map[string]any)
		if !ok && raw != nil {
			return nil, &core.ProviderError{Message: "each Chat message must be an object", Class: core.ProviderErrorInvalidRequest}
		}
		messages = append(messages, message)
	}
	return messages, nil
}

// googleChatFieldLosses reports the Chat fields Google drops, in order, so
// the losses do not depend on map order.
func googleChatFieldLosses(body map[string]any) []core.Loss {
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(body)) {
		value := body[field]
		switch field {
		case "model", "stream", "messages", "max_tokens", "temperature":
			// The request's model and the operation decide the first two,
			// and Google maps the rest.
			continue
		}
		if value == nil {
			continue
		}
		loss := core.Loss{Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Google does not send this Chat field"}
		switch {
		case field == "tools":
			if tools, ok := value.([]any); ok && len(tools) == 0 {
				loss.Detail = "Google already behaves as this value asks"
			} else {
				loss.Severity, loss.Detail = translate.LossMaterial, "Google does not send tools, so the model cannot call them"
			}
		case structuralChatField(field):
			if chatFieldDefault(field, value) {
				loss.Detail = "Google already behaves as this value asks"
			} else {
				loss.Severity = translate.LossMaterial
			}
		}
		losses = append(losses, loss)
	}
	return losses
}

// googleChatMessageLosses reports what Google drops from each message.
func googleChatMessageLosses(messages []map[string]any) []core.Loss {
	var losses []core.Loss
	for index, message := range messages {
		for _, field := range slices.Sorted(maps.Keys(message)) {
			value := message[field]
			if field == "role" || value == nil {
				continue
			}
			path := "messages." + strconv.Itoa(index) + "." + field
			if field == "content" {
				if _, text := value.(string); !text {
					losses = append(losses, core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossMaterial,
						Detail: "Google receives only text content, so this content is sent as empty text"})
				}
				continue
			}
			severity := translate.LossAdvisory
			switch field {
			case "tool_calls", "function_call", "tool_call_id":
				severity = translate.LossMaterial
			}
			losses = append(losses, core.Loss{Path: path, Class: translate.LossDropped, Severity: severity, Detail: "Google does not send this message field"})
		}
	}
	return losses
}

// googleContentRequest maps messages to Gemini's contents and parts, as the
// gateway does. Gemini has no system role: system and developer text
// becomes systemInstruction. maxTokens and temperature pass through as the
// request holds them.
func googleContentRequest(messages []map[string]any, maxTokens, temperature any, modalities []string) map[string]any {
	contents := make([]map[string]any, 0, len(messages))
	var systemParts []map[string]any
	for _, message := range messages {
		role, _ := message["role"].(string)
		text, _ := message["content"].(string)
		if role == "system" || role == "developer" {
			systemParts = append(systemParts, map[string]any{"text": text})
			continue
		}
		if role == "assistant" {
			role = "model"
		}
		if role == "" {
			role = "user"
		}
		contents = append(contents, map[string]any{
			"role": role, "parts": []map[string]any{{"text": text}},
		})
	}
	request := map[string]any{"contents": contents}
	if len(systemParts) > 0 {
		request["systemInstruction"] = map[string]any{"parts": systemParts}
	}
	generation := map[string]any{}
	if maxTokens != nil {
		generation["maxOutputTokens"] = maxTokens
	}
	if temperature != nil {
		generation["temperature"] = temperature
	}
	if len(modalities) > 0 {
		generation["responseModalities"] = modalities
	}
	if len(generation) > 0 {
		request["generationConfig"] = generation
	}
	return request
}

// googleToOpenAIChat reshapes a generateContent answer into the Chat
// Completions envelope the gateway returns.
func googleToOpenAIChat(model string, decoded map[string]any) map[string]any {
	text := strings.Builder{}
	for _, part := range googleParts(decoded) {
		if value, ok := part["text"].(string); ok {
			text.WriteString(value)
		}
	}
	inputTokens, outputTokens, totalTokens := googleUsage(decoded)
	served := model
	if value, ok := decoded["modelVersion"].(string); ok && value != "" {
		served = value
	}
	return map[string]any{
		"id":     "chatcmpl-google",
		"object": "chat.completion",
		"model":  served,
		"choices": []any{map[string]any{
			"index":         0,
			"finish_reason": "stop",
			"message":       map[string]any{"role": "assistant", "content": text.String()},
		}},
		"usage": map[string]any{
			"prompt_tokens": inputTokens, "completion_tokens": outputTokens, "total_tokens": totalTokens,
		},
	}
}

// googleParts returns the parts of the first candidate.
func googleParts(decoded map[string]any) []map[string]any {
	candidates, _ := decoded["candidates"].([]any)
	if len(candidates) == 0 {
		return nil
	}
	first, _ := candidates[0].(map[string]any)
	content, _ := first["content"].(map[string]any)
	rawParts, _ := content["parts"].([]any)
	parts := make([]map[string]any, 0, len(rawParts))
	for _, raw := range rawParts {
		if part, ok := raw.(map[string]any); ok {
			parts = append(parts, part)
		}
	}
	return parts
}

// googleUsage reads usageMetadata. Thinking tokens are output the caller
// pays for, so they count as completion tokens.
func googleUsage(decoded map[string]any) (int, int, int) {
	usage, _ := decoded["usageMetadata"].(map[string]any)
	number := func(key string) int {
		if value, ok := usage[key].(float64); ok {
			return int(value)
		}
		return 0
	}
	input := number("promptTokenCount")
	output := number("candidatesTokenCount") + number("thoughtsTokenCount")
	total := number("totalTokenCount")
	if total == 0 {
		total = input + output
	}
	return input, output, total
}

// googleEmptyReplyError explains an answer that holds no text. Gemini's
// thinking models spend maxOutputTokens on reasoning before they answer, so
// a small max_tokens yields a 200 with an empty string, which a caller
// cannot tell from success. The diagnostic names the cause, as the
// gateway's does. Another target may still answer, so it permits failover;
// the upstream is healthy, so it counts against nothing.
func googleEmptyReplyError(label string, decoded map[string]any) error {
	for _, part := range googleParts(decoded) {
		if text, ok := part["text"].(string); ok && strings.TrimSpace(text) != "" {
			return nil
		}
		if _, ok := part["inlineData"]; ok {
			return nil
		}
	}
	usage, _ := decoded["usageMetadata"].(map[string]any)
	thoughts, _ := usage["thoughtsTokenCount"].(float64)
	finish := ""
	if candidates, ok := decoded["candidates"].([]any); ok && len(candidates) > 0 {
		if first, ok := candidates[0].(map[string]any); ok {
			finish, _ = first["finishReason"].(string)
		}
	}
	diagnostic := label + ": the model returned no text"
	message := diagnostic
	switch {
	case thoughts > 0:
		diagnostic = fmt.Sprintf("%s: the model spent its entire output budget on reasoning (%d thinking tokens) and returned no text — raise max_tokens or omit it",
			label, int(thoughts))
		message = diagnostic
	case finish != "" && finish != "STOP":
		diagnostic = fmt.Sprintf("%s: the model returned no text (finish reason %s)", label, finish)
		// A finish reason is one of Google's enum values; anything else is
		// upstream text, which stays out of the message.
		if codexErrorIdentifier.MatchString(finish) {
			message = diagnostic
		}
	}
	return &core.ProviderError{
		Message: message, Class: core.ProviderErrorUpstream,
		Classification: core.ProviderErrorClassification{FailoverEligible: true},
		Cause:          &InvocationError{Msg: sanitizeDiagnosticTextLimit(diagnostic, diagnosticErrorLimit), FailoverEligible: true},
	}
}
