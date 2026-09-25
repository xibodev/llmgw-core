package providers

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

func (p *Copilot) invokeResponses(ctx context.Context, request core.Request) (core.Response, error) {
	call, err := copilotResponsesCall(request, false)
	if err != nil {
		return core.Response{}, err
	}
	response, _, err := p.send(ctx, request.Credential, call)
	if err != nil {
		return core.Response{}, err
	}
	raw, err := copilotAnswer(ctx, response)
	if err != nil {
		return core.Response{}, err
	}
	if failure := copilotResponsesFailure(request, response, raw); failure != nil {
		return core.Response{}, failure
	}
	if _, err := copilotResponsesAnswer(raw); err != nil {
		return core.Response{}, err
	}
	return core.Response{Body: raw, ContentType: core.ContentTypeJSON}, nil
}

func (p *Copilot) streamResponses(ctx context.Context, request core.Request) (core.StreamIter, error) {
	call, err := copilotResponsesCall(request, true)
	if err != nil {
		return nil, err
	}
	response, _, err := p.send(ctx, request.Credential, call)
	if err != nil {
		return nil, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		raw, _ := copilotAnswer(ctx, response)
		return nil, copilotResponsesFailure(request, response, raw)
	}
	return newCopilotStream(ctx, response.Body, nil), nil
}

// copilotResponsesCall shapes a native Responses request as the gateway
// does: the caller's body with the request's model and stream, without the
// gateway's force_api_support control.
func copilotResponsesCall(request core.Request, stream bool) (copilotCall, error) {
	payload, err := copilotPayload(request)
	if err != nil {
		return copilotCall{}, err
	}
	payload["model"], payload["stream"] = request.Model, stream
	delete(payload, "force_api_support")
	body, err := copilotEncode(payload)
	if err != nil {
		return copilotCall{}, err
	}
	accept := core.ContentTypeJSON
	if stream {
		accept = core.ContentTypeEventStream
	}
	return copilotCall{method: http.MethodPost, path: "/responses", body: body, accept: accept, vision: copilotResponsesImages(payload["input"])}, nil
}

// copilotResponsesFailure reports a native Responses error status. Copilot
// not finding the endpoint, 404 or 405, means the model does not serve
// Responses natively after all, which the gateway answers by translating;
// here it is a *core.SurfaceError, which permits failover.
func copilotResponsesFailure(request core.Request, response *http.Response, raw []byte) error {
	switch {
	case response.StatusCode == http.StatusNotFound, response.StatusCode == http.StatusMethodNotAllowed:
		return &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	case response.StatusCode >= http.StatusBadRequest:
		return copilotStatusFailure(response, raw)
	}
	return nil
}

// copilotResponsesAnswer decodes a Responses answer the gateway would use:
// a JSON object with an output array.
func copilotResponsesAnswer(raw []byte) (map[string]any, error) {
	var answer map[string]any
	if json.Unmarshal(raw, &answer) != nil || len(answer) == 0 {
		return nil, copilotMalformed("Copilot's Responses answer is not a JSON object", nil)
	}
	if _, ok := answer["output"].([]any); !ok {
		return nil, copilotMalformed("Copilot's Responses answer has no output", nil)
	}
	return answer, nil
}

// copilotAnswer reads and closes a response. A successful body must arrive
// whole; an error's body only explains the error, so a broken or oversized
// one is dropped.
func copilotAnswer(ctx context.Context, response *http.Response) ([]byte, error) {
	defer response.Body.Close()
	if response.StatusCode >= http.StatusBadRequest {
		return copilotErrorBody(response), nil
	}
	return copilotRead(ctx, response, copilotMaxResponseBytes)
}

// invokeChatOverResponses serves Chat over Responses and converts the answer
// back, as the gateway's adaptation does: an answer without text answers
// with its reasoning, and an explicit force_api_support: true is answered
// with the gateway's adaptation marker.
func (p *Copilot) invokeChatOverResponses(ctx context.Context, request core.Request, chat copilotChat) (core.Response, error) {
	answer, stripped, err := p.overResponses(ctx, request, chat)
	if err != nil {
		return core.Response{}, err
	}
	converted := translate.ResponsesToChatWithReport(request.Model, answer)
	if err := converted.RejectMaterialLoss(); err != nil {
		return core.Response{}, copilotUnsupported("Copilot's Responses answer cannot be converted to Chat", err)
	}
	result := converted.Value
	if choices, _ := result["choices"].([]any); len(choices) > 0 {
		choice, _ := choices[0].(map[string]any)
		if message, ok := choice["message"].(map[string]any); ok {
			content, _ := message["content"].(string)
			reasoning, _ := message["reasoning_content"].(string)
			if strings.TrimSpace(content) == "" && strings.TrimSpace(reasoning) != "" {
				message["content"] = reasoning
			}
		}
	}
	if chat.marked {
		result["forced_support"] = map[string]any{"req_api": "chat", "resp_api": "responses"}
	}
	body, err := json.Marshal(result)
	if err != nil {
		return core.Response{}, copilotMalformed("Copilot's Responses answer cannot be converted to Chat", err)
	}
	losses := slices.Concat(chat.losses, stripped, converted.Report.Losses)
	return core.Response{Body: body, ContentType: core.ContentTypeJSON, Losses: translate.NewReport(losses...).Losses}, nil
}

// streamChatOverResponses serves a Chat stream over Responses as the gateway
// does: from one complete answer, rendered as Chat chunks and [DONE].
func (p *Copilot) streamChatOverResponses(ctx context.Context, request core.Request, chat copilotChat) (core.StreamIter, error) {
	answer, stripped, err := p.overResponses(ctx, request, chat)
	if err != nil {
		return nil, err
	}
	converted := translate.ResponsesToChatChunksWithReport(request.Model, answer)
	if err := converted.RejectMaterialLoss(); err != nil {
		return nil, copilotUnsupported("Copilot's Responses answer cannot be converted to Chat", err)
	}
	frames := make([][]byte, 0, len(converted.Value)+1)
	for _, chunk := range converted.Value {
		frames = append(frames, []byte("data: "+chunk+"\n\n"))
	}
	frames = append(frames, []byte("data: [DONE]\n\n"))
	losses := slices.Concat(chat.losses, stripped, converted.Report.Losses)
	return &copilotFrames{frames: frames, losses: translate.NewReport(losses...).Losses}, nil
}

// overResponses sends a Chat request converted to Responses, streamed or
// not, as one complete answer. A 400 that names temperature or top_p is sent
// once more without them, with the same session, and they are reported as
// dropped, as the gateway retries such a request.
func (p *Copilot) overResponses(ctx context.Context, request core.Request, chat copilotChat) (map[string]any, []core.Loss, error) {
	body, err := copilotEncode(chat.payload)
	if err != nil {
		return nil, nil, err
	}
	call := copilotCall{method: http.MethodPost, path: "/responses", body: body, accept: core.ContentTypeJSON, vision: chat.vision}
	response, session, err := p.send(ctx, request.Credential, call)
	if err != nil {
		return nil, nil, err
	}
	raw, err := copilotAnswer(ctx, response)
	if err != nil {
		return nil, nil, err
	}
	var stripped []core.Loss
	if response.StatusCode == http.StatusBadRequest {
		if stripped = copilotStripRejected(chat.payload, raw); len(stripped) > 0 && ctx.Err() == nil {
			if call.body, err = copilotEncode(chat.payload); err != nil {
				return nil, nil, err
			}
			if retry, err := p.do(ctx, session, call); err == nil {
				if raw, err = copilotAnswer(ctx, retry); err != nil {
					return nil, nil, err
				}
				response = retry
			}
		}
	}
	if response.StatusCode >= http.StatusBadRequest {
		return nil, nil, copilotStatusFailure(response, raw)
	}
	answer, err := copilotResponsesAnswer(raw)
	return answer, stripped, err
}
