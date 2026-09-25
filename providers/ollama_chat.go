package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"mime"
	"slices"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// ollamaChatBody is what the gateway's Chat facade hands Ollama from a Chat
// Completions body, decoded as the facade decodes it: numbers as float64,
// so each reaches Ollama as the gateway sends it, and field names matched
// as encoding/json matches them.
type ollamaChatBody struct {
	Messages    []map[string]any `json:"messages"`
	Temperature any              `json:"temperature"`
	MaxTokens   any              `json:"max_tokens"`
	TopP        any              `json:"top_p"`
	Tools       any              `json:"tools"`
}

// ollamaCall is a request Ollama has checked and shaped.
type ollamaCall struct {
	body   []byte
	losses []core.Loss
}

// ollamaPrepare refuses what Ollama cannot serve before anything is sent,
// then shapes the /api/chat body as the gateway does and encodes it as the
// gateway does: keys sorted and HTML escaped.
func ollamaPrepare(request core.Request, stream bool) (ollamaCall, error) {
	if request.Surface != core.ModelSurfaceChatCompletions {
		return ollamaCall{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	if strings.TrimSpace(request.Model) == "" {
		return ollamaCall{}, &core.ProviderError{Message: "an Ollama request needs a model", Class: core.ProviderErrorInvalidRequest}
	}
	body, fields, err := ollamaDecodeChat(request)
	if err != nil {
		return ollamaCall{}, err
	}
	messages, losses := ollamaMessages(body.Messages)
	payload := map[string]any{"model": request.Model, "messages": messages, "stream": stream}
	options := map[string]any{}
	if body.Temperature != nil {
		options["temperature"] = body.Temperature
	}
	if body.MaxTokens != nil {
		options["num_predict"] = body.MaxTokens
	}
	if body.TopP != nil {
		options["top_p"] = body.TopP
	}
	if len(options) > 0 {
		payload["options"] = options
	}
	if body.Tools != nil {
		payload["tools"] = body.Tools
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return ollamaCall{}, &core.ProviderError{Message: "the Ollama request could not be encoded", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	losses = append(losses, ollamaDroppedFields(fields)...)
	return ollamaCall{body: encoded, losses: translate.NewReport(losses...).Losses}, nil
}

// ollamaDecodeChat decodes the JSON object body twice: as the gateway's
// Chat facade does, and whole, to report what Ollama drops.
func ollamaDecodeChat(request core.Request) (ollamaChatBody, map[string]any, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return ollamaChatBody{}, nil, &core.ProviderError{Message: "an Ollama request body must be JSON", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	decoder := json.NewDecoder(bytes.NewReader(request.Body))
	var fields map[string]any
	err = decoder.Decode(&fields)
	if err == nil {
		if _, trailing := decoder.Token(); trailing != io.EOF {
			err = errors.New("the body continues after its JSON object")
		}
	}
	var body ollamaChatBody
	if err == nil && fields != nil {
		err = json.Unmarshal(request.Body, &body)
	}
	if err != nil || fields == nil {
		return ollamaChatBody{}, nil, &core.ProviderError{Message: "the Ollama request body is not a Chat Completions object", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	return body, fields, nil
}

// ollamaDroppedFields reports each field of a Chat body that Ollama does
// not send, in order, so the losses do not depend on map order. A field
// that changes the structure of the answer, such as response_format, is a
// material loss unless it asks for what Ollama does anyway; any other,
// such as stop, seed or max_completion_tokens, is an advisory one.
func ollamaDroppedFields(fields map[string]any) []core.Loss {
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(fields)) {
		value := fields[field]
		if value == nil || ollamaSentField(field) {
			continue
		}
		loss := core.Loss{Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Ollama does not send this Chat field"}
		if structuralChatField(field) {
			if chatFieldDefault(field, value) {
				loss.Detail = "Ollama already behaves as this value asks"
			} else {
				loss.Severity = translate.LossMaterial
			}
		}
		losses = append(losses, loss)
	}
	return losses
}

// ollamaSentField reports a field ollamaChatBody decodes, as encoding/json
// matches it, or one the request decides: the model and whether to stream.
func ollamaSentField(field string) bool {
	for _, sent := range []string{"messages", "temperature", "max_tokens", "top_p", "tools", "model", "stream"} {
		if strings.EqualFold(field, sent) {
			return true
		}
	}
	return false
}
