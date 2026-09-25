package providers

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

func (p *Copilot) invokeChat(ctx context.Context, request core.Request) (core.Response, error) {
	chat, err := p.prepareChat(request, false)
	if err != nil {
		return core.Response{}, err
	}
	if chat.responses {
		return p.invokeChatOverResponses(ctx, request, chat)
	}
	body, err := copilotEncode(chat.payload)
	if err != nil {
		return core.Response{}, err
	}
	response, _, err := p.send(ctx, request.Credential, copilotCall{method: http.MethodPost, path: "/chat/completions", body: body, vision: chat.vision})
	if err != nil {
		return core.Response{}, err
	}
	raw, err := copilotAnswer(ctx, response)
	if err != nil {
		return core.Response{}, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		return core.Response{}, copilotStatusFailure(response, raw)
	}
	if err := copilotCheckChat(raw); err != nil {
		return core.Response{}, err
	}
	return core.Response{Body: raw, ContentType: core.ContentTypeJSON, Losses: chat.losses}, nil
}

// streamChat opens a Chat stream. The gateway asks Copilot for it with the
// same Accept header as any other request.
func (p *Copilot) streamChat(ctx context.Context, request core.Request) (core.StreamIter, error) {
	chat, err := p.prepareChat(request, true)
	if err != nil {
		return nil, err
	}
	if chat.responses {
		return p.streamChatOverResponses(ctx, request, chat)
	}
	body, err := copilotEncode(chat.payload)
	if err != nil {
		return nil, err
	}
	response, _, err := p.send(ctx, request.Credential, copilotCall{method: http.MethodPost, path: "/chat/completions", body: body, vision: chat.vision})
	if err != nil {
		return nil, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		raw, _ := copilotAnswer(ctx, response)
		return nil, copilotStatusFailure(response, raw)
	}
	return newCopilotStream(ctx, response.Body, chat.losses), nil
}

// copilotCheckChat refuses a Chat answer the gateway would not use: one that
// is not a JSON object with a first choice, or that carries an error beside
// a success status, which the gateway retries.
func copilotCheckChat(raw []byte) error {
	var answer map[string]any
	if json.Unmarshal(raw, &answer) != nil || len(answer) == 0 {
		return copilotMalformed("Copilot's Chat answer is not a JSON object", nil)
	}
	if copilotSoftError(answer) {
		return &core.ProviderError{
			Message: "Copilot answered with an error", Class: core.ProviderErrorUpstream,
			Classification: core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true},
		}
	}
	choices, _ := answer["choices"].([]any)
	if len(choices) == 0 {
		return copilotMalformed("Copilot's Chat answer has no choices", nil)
	}
	if _, ok := choices[0].(map[string]any); !ok {
		return copilotMalformed("Copilot's Chat answer has no choices", nil)
	}
	return nil
}

// copilotSoftError reports an error field the gateway reads as an error: a
// nonblank string, an object with a nonblank message, or an object without
// a string message.
func copilotSoftError(answer map[string]any) bool {
	switch value := answer["error"].(type) {
	case nil:
		return false
	case string:
		return strings.TrimSpace(value) != ""
	case map[string]any:
		if message, ok := value["message"].(string); ok {
			return strings.TrimSpace(message) != ""
		}
	}
	return true
}
