package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"slices"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// openAIResponsesBody is a native Responses request as the gateway sends
// it: the caller's body with the request's model and the operation's
// stream flag, without the gateway's force_api_support control.
func openAIResponsesBody(call openAICompatibleCall, stream bool) map[string]any {
	payload := maps.Clone(call.payload)
	payload["model"], payload["stream"] = call.model, stream
	delete(payload, "force_api_support")
	return payload
}

// invokeResponses returns the upstream's Responses answer as sent, once it
// checks out as the gateway checks it.
func (p *OpenAICompatible) invokeResponses(ctx context.Context, call openAICompatibleCall) (core.Response, error) {
	payload := openAIResponsesBody(call, false)
	response, err := p.post(ctx, "/responses", p.header(call.access, core.ContentTypeJSON, openAIResponsesImages(payload["input"])), payload)
	if err != nil {
		return core.Response{}, err
	}
	raw, err := p.read(ctx, response)
	if err != nil {
		return core.Response{}, err
	}
	if err := p.responsesRefused(call, response, raw); err != nil {
		return core.Response{}, err
	}
	if _, err := p.responsesAnswer(raw); err != nil {
		return core.Response{}, err
	}
	return core.Response{Body: raw, ContentType: core.ContentTypeJSON}, nil
}

func (p *OpenAICompatible) streamResponses(ctx context.Context, call openAICompatibleCall) (core.StreamIter, error) {
	payload := openAIResponsesBody(call, true)
	response, err := p.post(ctx, "/responses", p.header(call.access, core.ContentTypeEventStream, openAIResponsesImages(payload["input"])), payload)
	if err != nil {
		return nil, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		raw, _ := p.read(ctx, response)
		return nil, p.responsesRefused(call, response, raw)
	}
	return newSSEFrameStream(ctx, p.label, response.Body), nil
}

// responsesRefused reports a refused native Responses request. The
// upstream not finding the endpoint, 404 or 405, means the model does not
// serve Responses natively after all, which the gateway answers by
// translating; here it is a *core.SurfaceError, which permits failover.
func (p *OpenAICompatible) responsesRefused(call openAICompatibleCall, response *http.Response, raw []byte) error {
	switch status := response.StatusCode; {
	case status == http.StatusNotFound, status == http.StatusMethodNotAllowed:
		return &core.SurfaceError{Surface: core.ModelSurfaceResponses, Model: call.model}
	case status >= http.StatusBadRequest:
		return p.refused(response, raw)
	}
	return nil
}

// completeOverResponses sends Chat converted to Responses and reads one
// complete answer, as the gateway's adaptation does, streamed or not. A
// 400 that names temperature or top_p is sent once more without them, and
// they are reported as dropped.
func (p *OpenAICompatible) completeOverResponses(ctx context.Context, call openAICompatibleCall, chat openAIChat) (map[string]any, []core.Loss, error) {
	header := p.header(call.access, core.ContentTypeJSON, openAIResponsesImages(chat.body["input"]))
	response, err := p.post(ctx, "/responses", header, chat.body)
	if err != nil {
		return nil, nil, err
	}
	raw, err := p.read(ctx, response)
	if err != nil {
		return nil, nil, err
	}
	var stripped []core.Loss
	if response.StatusCode == http.StatusBadRequest {
		if stripped = openAIStripRejected(chat.body, raw); len(stripped) > 0 && ctx.Err() == nil {
			// A retry the upstream cannot be reached for leaves the first
			// refusal, as in the gateway.
			if retried, err := p.post(ctx, "/responses", header, chat.body); err == nil {
				if raw, err = p.read(ctx, retried); err != nil {
					return nil, nil, err
				}
				response = retried
			}
		}
	}
	if response.StatusCode >= http.StatusBadRequest {
		return nil, nil, p.refused(response, raw)
	}
	answer, err := p.responsesAnswer(raw)
	return answer, stripped, err
}

// openAIStripRejected removes the sampling parameters a 400 answer names,
// reporting each as dropped.
func openAIStripRejected(payload map[string]any, answer []byte) []core.Loss {
	lower := bytes.ToLower(answer)
	var losses []core.Loss
	for _, field := range []string{"temperature", "top_p"} {
		if _, present := payload[field]; present && bytes.Contains(lower, []byte(field)) {
			delete(payload, field)
			losses = append(losses, openAIDropped(field, "the upstream rejected this sampling parameter for the model"))
		}
	}
	return losses
}

// invokeChatOverResponses converts the answer back to Chat as the gateway's
// adaptation does: an answer without text answers with its reasoning, and
// an explicit force_api_support: true is answered with the gateway's
// adaptation marker.
func (p *OpenAICompatible) invokeChatOverResponses(ctx context.Context, call openAICompatibleCall, chat openAIChat) (core.Response, error) {
	answer, stripped, err := p.completeOverResponses(ctx, call, chat)
	if err != nil {
		return core.Response{Losses: chat.losses}, err
	}
	converted := translate.ResponsesToChatWithReport(call.model, answer)
	losses := translate.NewReport(slices.Concat(chat.losses, stripped, converted.Report.Losses)...).Losses
	if err := converted.RejectMaterialLoss(); err != nil {
		return core.Response{Losses: losses}, openAIUnsupported("the "+p.label+" Responses answer cannot be converted to Chat", err)
	}
	result := converted.Value
	openAIReasoningAsContent(result)
	if chat.marked {
		result["forced_support"] = map[string]any{"req_api": "chat", "resp_api": "responses"}
	}
	body, err := json.Marshal(result)
	if err != nil {
		return core.Response{Losses: losses}, unusableResponse("the "+p.label+" Responses answer cannot be converted to Chat", err)
	}
	return core.Response{Body: body, ContentType: core.ContentTypeJSON, Losses: losses}, nil
}

// streamChatOverResponses streams Chat served over Responses as the
// gateway does: one complete answer, rendered as Chat chunks and [DONE].
func (p *OpenAICompatible) streamChatOverResponses(ctx context.Context, call openAICompatibleCall, chat openAIChat) (core.StreamIter, error) {
	answer, stripped, err := p.completeOverResponses(ctx, call, chat)
	if err != nil {
		return nil, err
	}
	converted := translate.ResponsesToChatChunksWithReport(call.model, answer)
	if err := converted.RejectMaterialLoss(); err != nil {
		return nil, openAIUnsupported("the "+p.label+" Responses answer cannot be converted to Chat", err)
	}
	frames := make([][]byte, 0, len(converted.Value)+1)
	for _, chunk := range converted.Value {
		frames = append(frames, []byte("data: "+chunk+"\n\n"))
	}
	frames = append(frames, []byte("data: [DONE]\n\n"))
	losses := translate.NewReport(slices.Concat(chat.losses, stripped, converted.Report.Losses)...).Losses
	return &openAIStream{StreamIter: &openAIFrames{frames: frames}, losses: losses}, nil
}

// openAIReasoningAsContent answers with the reasoning when a completion has
// no content, as the gateway does.
func openAIReasoningAsContent(chat map[string]any) {
	choices, _ := chat["choices"].([]any)
	if len(choices) == 0 {
		return
	}
	choice, _ := choices[0].(map[string]any)
	message, _ := choice["message"].(map[string]any)
	content, _ := message["content"].(string)
	reasoning, _ := message["reasoning_content"].(string)
	if message != nil && strings.TrimSpace(content) == "" && strings.TrimSpace(reasoning) != "" {
		message["content"] = reasoning
	}
}
