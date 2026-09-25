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

// antigravityChatPayload applies the gateway's Chat field policy for
// Antigravity to a Chat Completions body. It returns the payload the
// gateway's Chat facade hands the legacy adapter, which is the messages
// and, when set, max_tokens, temperature and tools, so the request reaches
// Cloud Code Assist exactly as the gateway sends it.
//
// Every other field is dropped, as the gateway drops it, and reported at
// its path. A field that changes the structure of the answer, such as
// response_format or n, is a material loss unless it asks for what
// Antigravity does anyway; any other field, such as top_p, stop or
// max_completion_tokens, is an advisory one. Fields are read in order, so
// the losses do not depend on map order.
func antigravityChatPayload(request core.Request) (map[string]any, []core.Loss, error) {
	body, err := antigravityChatBody(request)
	if err != nil {
		return nil, nil, err
	}
	// The gateway sends an absent message list as an empty one.
	payload := map[string]any{"messages": []any{}}
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(body)) {
		value := body[field]
		switch field {
		case "model", "stream":
			// The request's model and the operation decide these.
			continue
		case "messages", "max_tokens", "temperature", "tools":
			if value != nil {
				payload[field] = value
			}
			continue
		}
		if value == nil {
			continue
		}
		loss := core.Loss{Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Antigravity does not send this Chat field"}
		if structuralChatField(field) {
			if chatFieldDefault(field, value) {
				loss.Detail = "Antigravity already behaves as this value asks"
			} else {
				loss.Severity = translate.LossMaterial
			}
		}
		losses = append(losses, loss)
	}
	return payload, translate.NewReport(losses...).Losses, nil
}

// antigravityChatBody decodes the JSON object body. Numbers decode as
// float64, as the gateway decodes a Chat body, so a number in a tool schema
// reaches Cloud Code Assist as the gateway sends it.
func antigravityChatBody(request core.Request) (map[string]any, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return nil, &core.ProviderError{Message: "an Antigravity request body must be JSON", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	decoder := json.NewDecoder(bytes.NewReader(request.Body))
	var body map[string]any
	err = decoder.Decode(&body)
	if err == nil {
		if _, trailing := decoder.Token(); trailing != io.EOF {
			err = errors.New("the body continues after its JSON object")
		}
	}
	if err != nil || body == nil {
		return nil, &core.ProviderError{Message: "the Antigravity request body is not a JSON object", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	return body, nil
}
