package providers

import (
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers/zen"
)

// chatOverResponses converts a Chat request for a Responses-only model as
// the gateway does: the Chat admission first, then the conversion, whose
// material losses refuse the request. The Responses admission follows when
// the payload is sent.
func (p *Zen) chatOverResponses(call zenCall, stream bool) (map[string]any, []core.Loss, error) {
	messages, options, err := zenChat(call.payload)
	if err != nil {
		return nil, nil, err
	}
	if call.access.anonymous {
		messages, options = zenAdmitChat(messages, options)
		// The gateway gives an anonymous Muse completion, but not a
		// stream, minimal reasoning unless the caller chose an effort.
		if !stream && zenMuse(call.model) && options["reasoning_effort"] == nil {
			options = maps.Clone(options)
			options["reasoning_effort"] = "minimal"
		}
	}
	reportable := make(map[string]any, len(options))
	for key, value := range options {
		if !strings.HasPrefix(key, "_") {
			reportable[key] = value
		}
	}
	converted := translate.ChatToResponsesWithReport(call.model, messages, reportable, false)
	losses := exemptThoughtSignatures(converted.Report.Losses)
	if err := translate.RejectMaterialLoss(translate.Report{Losses: losses}); err != nil {
		return nil, losses, &core.ProviderError{
			Message: "the Chat request cannot be sent to OpenCode Zen over Responses", Class: core.ProviderErrorUnsupported,
			Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
		}
	}
	return converted.Value, losses, nil
}

func zenMuse(model string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "muse-spark-")
}

// completeResponse sends a Responses payload and reads the whole response.
// Anonymous access always streams, so the response is assembled from its
// events. With retry, a 400 naming a sampling parameter the payload sets
// is retried once without it, as the gateway retries Chat over Responses.
func (p *Zen) completeResponse(ctx context.Context, call zenCall, payload map[string]any, retry bool) (map[string]any, []byte, error) {
	if call.access.anonymous {
		payload["stream"] = true
		payload = zen.AdmitResponses(payload)
	}
	header := zenHeaders(call.access, call.identity)
	header.Set("Accept", core.ContentTypeJSON)
	zenVisionHeader(header, zenResponsesHasImages(payload["input"]))
	response, err := p.post(ctx, "/responses", header, payload)
	if err != nil {
		return nil, nil, err
	}
	raw, err := readZenResponse(ctx, response)
	if err != nil {
		return nil, nil, err
	}
	if retry && response.StatusCode == http.StatusBadRequest && zenStripRejected(payload, raw) && ctx.Err() == nil {
		// A retry Zen cannot be reached for leaves the first refusal.
		if retried, err := p.post(ctx, "/responses", header, payload); err == nil {
			if raw, err = readZenResponse(ctx, retried); err != nil {
				return nil, nil, err
			}
			response = retried
		}
	}
	if response.StatusCode >= 400 {
		return nil, nil, p.statusError(ctx, response, raw, call)
	}
	var result map[string]any
	if call.access.anonymous {
		result = zenFinalResponse(raw)
	}
	if len(result) == 0 && (json.Unmarshal(raw, &result) != nil || len(result) == 0) {
		return nil, nil, zenUpstreamError("OpenCode Zen returned a response that is not JSON", false, nil)
	}
	if _, ok := result["output"].([]any); !ok {
		return nil, nil, zenUpstreamError("OpenCode Zen returned a response without output", false, nil)
	}
	return result, raw, nil
}

// zenStripRejected drops the sampling parameters a 400 names, reporting
// whether it dropped any.
func zenStripRejected(payload map[string]any, body []byte) bool {
	text := strings.ToLower(string(body))
	stripped := false
	for _, key := range []string{"temperature", "top_p"} {
		if _, ok := payload[key]; ok && strings.Contains(text, key) {
			delete(payload, key)
			stripped = true
		}
	}
	return stripped
}

func (p *Zen) invokeChatOverResponses(ctx context.Context, call zenCall) (core.Response, error) {
	payload, losses, err := p.chatOverResponses(call, false)
	if err != nil {
		return core.Response{Losses: losses}, err
	}
	result, _, err := p.completeResponse(ctx, call, payload, true)
	if err != nil {
		return core.Response{Losses: losses}, err
	}
	converted := translate.ResponsesToChatWithReport(call.model, result)
	losses = append(losses, converted.Report.Losses...)
	if err := converted.RejectMaterialLoss(); err != nil {
		return core.Response{Losses: losses}, zenUnrepresentable(err)
	}
	chat := converted.Value
	zenReasoningAsContent(chat)
	body, err := json.Marshal(chat)
	if err != nil {
		return core.Response{Losses: losses}, zenUpstreamError("the OpenCode Zen completion could not be encoded", false, err)
	}
	return core.Response{Body: body, ContentType: core.ContentTypeJSON, Losses: losses}, nil
}

// streamChatOverResponses renders the complete response as Chat chunks,
// as the gateway streams Chat for a Responses-only model.
func (p *Zen) streamChatOverResponses(ctx context.Context, call zenCall) (core.StreamIter, error) {
	payload, losses, err := p.chatOverResponses(call, true)
	if err != nil {
		return nil, err
	}
	result, _, err := p.completeResponse(ctx, call, payload, true)
	if err != nil {
		return nil, err
	}
	converted := translate.ResponsesToChatChunksWithReport(call.model, result)
	losses = append(losses, converted.Report.Losses...)
	if err := converted.RejectMaterialLoss(); err != nil {
		return nil, zenUnrepresentable(err)
	}
	frames := make([][]byte, 0, len(converted.Value)+1)
	for _, chunk := range converted.Value {
		frames = append(frames, []byte("data: "+chunk+"\n\n"))
	}
	frames = append(frames, []byte("data: [DONE]\n\n"))
	return &zenStream{events: &zenFrames{frames: frames}, losses: losses}, nil
}

func zenUnrepresentable(err error) error {
	return &core.ProviderError{
		Message: "the OpenCode Zen response cannot be represented as Chat Completions", Class: core.ProviderErrorUnsupported,
		Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
	}
}

// zenReasoningAsContent answers with the reasoning when a completion has
// no content, as the gateway does.
func zenReasoningAsContent(chat map[string]any) {
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

// responsesPayload is a native Responses request as the gateway sends it:
// the caller's fields with the request's model and the operation's stream
// flag, and without the gateway's force_api_support hint.
func responsesPayload(call zenCall, stream bool) map[string]any {
	payload := maps.Clone(call.payload)
	payload["model"] = call.model
	payload["stream"] = stream
	delete(payload, "force_api_support")
	return payload
}

// invokeResponses returns a keyed response as Zen sent it, and an
// anonymous one as assembled from its events.
func (p *Zen) invokeResponses(ctx context.Context, call zenCall) (core.Response, error) {
	result, raw, err := p.completeResponse(ctx, call, responsesPayload(call, false), false)
	if err != nil {
		return core.Response{}, err
	}
	if call.access.anonymous {
		if raw, err = json.Marshal(result); err != nil {
			return core.Response{}, zenUpstreamError("the OpenCode Zen response could not be encoded", false, err)
		}
	}
	return core.Response{Body: raw, ContentType: core.ContentTypeJSON}, nil
}

func (p *Zen) streamResponses(ctx context.Context, call zenCall) (core.StreamIter, error) {
	payload := responsesPayload(call, true)
	if call.access.anonymous {
		payload = zen.AdmitResponses(payload)
	}
	header := zenHeaders(call.access, call.identity)
	header.Set("Accept", core.ContentTypeEventStream)
	zenVisionHeader(header, zenResponsesHasImages(payload["input"]))
	response, err := p.post(ctx, "/responses", header, payload)
	if err != nil {
		return nil, err
	}
	if response.StatusCode >= 400 {
		raw, _ := readZenResponse(ctx, response)
		return nil, p.statusError(ctx, response, raw, call)
	}
	return &zenStream{events: &zenResponsesStream{ctx: ctx, body: response.Body, reader: newZenSSEReader(response.Body)}}, nil
}

// zenFinalResponse assembles an anonymous response from its events, as the
// gateway does: the response object of the last event carrying one, with
// the streamed text added as a message when its output has none. It reads
// data lines one by one, not whole records.
func zenFinalResponse(raw []byte) map[string]any {
	var last map[string]any
	var text strings.Builder
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		var event map[string]any
		if json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), &event) != nil {
			continue
		}
		if response, ok := event["response"].(map[string]any); ok && len(response) > 0 {
			last = response
		}
		if event["type"] == "response.output_text.delta" {
			delta, _ := event["delta"].(string)
			text.WriteString(delta)
		}
	}
	if last == nil {
		return nil
	}
	output, ok := last["output"].([]any)
	if !ok {
		output = []any{}
	}
	if strings.TrimSpace(text.String()) != "" && !zenOutputHasText(output) {
		output = append(output, map[string]any{
			"type": "message", "role": "assistant", "status": "completed",
			"content": []any{map[string]any{"type": "output_text", "text": text.String()}},
		})
	}
	last["output"] = output
	return last
}

func zenOutputHasText(output []any) bool {
	for _, raw := range output {
		item, _ := raw.(map[string]any)
		if item["type"] != "message" {
			continue
		}
		for _, rawPart := range anySlice(item["content"]) {
			part, _ := rawPart.(map[string]any)
			if text, _ := part["text"].(string); part["type"] == "output_text" && strings.TrimSpace(text) != "" {
				return true
			}
		}
	}
	return false
}
