// Package translation serves surfaces a provider lacks by translating through
// llm-translate. Every conversion reports its losses. An Adapter checks them
// against a core.LossPolicy and returns all of them, including the losses the
// policy allows.
package translation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"slices"

	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
)

// Adapter wraps a provider and serves, besides its native surfaces, every
// surface llm-translate can convert to one of them:
//
//   - Messages over a Chat Completions provider, streaming included;
//   - Chat Completions over a Responses provider;
//   - Responses over a Chat Completions provider.
//
// Losses the provider itself reports are merged into the same report. Request
// losses are checked before the provider is called, so a rejected translation
// never reaches it. Response losses of a stream are reported through
// core.LossReporter, not enforced, because their frames are already
// delivered.
type Adapter struct {
	Provider core.Provider
	Policy   core.LossPolicy
}

// route converts one client surface to one native surface and back.
type route struct {
	from, target core.ModelSurface
	request      func(model string, payload map[string]any, stream bool) (map[string]any, []core.Loss, error)
	response     func(model string, response map[string]any) (map[string]any, []core.Loss)
	streamable   bool
}

func routes() []route {
	return []route{
		{from: core.ModelSurfaceMessages, target: core.ModelSurfaceChatCompletions, request: messagesToChat, response: chatToMessagesResponse, streamable: true},
		{from: core.ModelSurfaceChatCompletions, target: core.ModelSurfaceResponses, request: chatToResponses, response: responsesToChatResponse},
		{from: core.ModelSurfaceResponses, target: core.ModelSurfaceChatCompletions, request: responsesToChat, response: chatToResponsesResponse},
	}
}

// NativeSurfaces reports the wrapped provider's native surfaces: translated
// surfaces are, by definition, not native.
func (a Adapter) NativeSurfaces(model string) []core.ModelSurface {
	return a.Provider.NativeSurfaces(model)
}

// Surfaces lists every surface the adapter serves for model, native first.
func (a Adapter) Surfaces(model string) []core.ModelSurface {
	surfaces := append([]core.ModelSurface(nil), a.Provider.NativeSurfaces(model)...)
	for _, candidate := range routes() {
		if !slices.Contains(surfaces, candidate.from) && core.ServesNatively(a.Provider, model, candidate.target) {
			surfaces = append(surfaces, candidate.from)
		}
	}
	return surfaces
}

// ListModels delegates to the wrapped provider.
func (a Adapter) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	return a.Provider.ListModels(ctx, credential)
}

func (a Adapter) route(model string, surface core.ModelSurface) (route, bool) {
	for _, candidate := range routes() {
		if candidate.from == surface && core.ServesNatively(a.Provider, model, candidate.target) {
			return candidate, true
		}
	}
	return route{}, false
}

// translateRequest converts a request to the route's native surface and
// enforces the policy on the conversion's losses.
func (a Adapter) translateRequest(r route, request core.Request, stream bool) (core.Request, []core.Loss, error) {
	payload, err := jsonPayload(request)
	if err != nil {
		return core.Request{}, nil, err
	}
	translated, losses, err := r.request(request.Model, payload, stream)
	if err != nil {
		return core.Request{}, losses, &core.ProviderError{Message: "the request cannot be translated: " + err.Error(), Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	if err := a.Policy.Check(losses); err != nil {
		return core.Request{}, losses, err
	}
	body, err := json.Marshal(translated)
	if err != nil {
		return core.Request{}, losses, err
	}
	upstream := request
	upstream.Surface, upstream.Body, upstream.ContentType = r.target, body, core.ContentTypeJSON
	return upstream, losses, nil
}

// Invoke serves request natively when the provider can, and otherwise by
// translation.
func (a Adapter) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	if core.ServesNatively(a.Provider, request.Model, request.Surface) {
		return a.Provider.Invoke(ctx, request)
	}
	r, ok := a.route(request.Model, request.Surface)
	if !ok {
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	upstream, losses, err := a.translateRequest(r, request, false)
	if err != nil {
		return core.Response{Losses: losses}, err
	}
	response, err := a.Provider.Invoke(ctx, upstream)
	losses = append(losses, response.Losses...)
	if err != nil {
		return core.Response{Losses: losses}, err
	}
	var result map[string]any
	if err := json.Unmarshal(response.Body, &result); err != nil {
		return core.Response{Losses: losses}, &core.ProviderError{
			Message: "the provider returned a response that is not JSON", Class: core.ProviderErrorUpstream,
			Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
		}
	}
	converted, responseLosses := r.response(request.Model, result)
	losses = append(losses, responseLosses...)
	if err := a.Policy.Check(responseLosses); err != nil {
		return core.Response{Losses: losses}, err
	}
	body, err := json.Marshal(converted)
	if err != nil {
		return core.Response{Losses: losses}, err
	}
	return core.Response{Body: body, ContentType: core.ContentTypeJSON, Losses: losses}, nil
}

// Stream serves request natively when the provider can, and otherwise by
// translation on routes that stream. The returned stream implements
// core.LossReporter.
func (a Adapter) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	if core.ServesNatively(a.Provider, request.Model, request.Surface) {
		return a.Provider.Stream(ctx, request)
	}
	r, ok := a.route(request.Model, request.Surface)
	if !ok || !r.streamable {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	upstream, losses, err := a.translateRequest(r, request, true)
	if err != nil {
		return nil, err
	}
	stream, err := a.Provider.Stream(ctx, upstream)
	if err != nil {
		return nil, err
	}
	losses = append(losses, core.StreamLosses(stream)...)
	return newMessagesStream(stream, request.Model, losses), nil
}

func jsonPayload(request core.Request) (map[string]any, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return nil, &core.ProviderError{Message: fmt.Sprintf("a translated %s request must be JSON", request.Surface), Class: core.ProviderErrorInvalidRequest}
	}
	var payload map[string]any
	if err := json.Unmarshal(request.Body, &payload); err != nil || payload == nil {
		return nil, &core.ProviderError{Message: fmt.Sprintf("the %s request body is not a JSON object", request.Surface), Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	return payload, nil
}

// chatBody assembles a Chat Completions request from converted parts.
func chatBody(model string, converted translate.OpenAIChatRequest, stream bool) map[string]any {
	body := make(map[string]any, len(converted.Keywords)+3)
	for key, value := range converted.Keywords {
		body[key] = value
	}
	body["model"] = model
	body["messages"] = converted.Messages
	delete(body, "stream")
	if stream {
		body["stream"] = true
	}
	return body
}

func messagesToChat(model string, payload map[string]any, stream bool) (map[string]any, []core.Loss, error) {
	converted := translate.AnthropicRequestToOpenAIWithReport(payload)
	return chatBody(model, converted.Value, stream), converted.Report.Losses, nil
}

func chatToMessagesResponse(model string, response map[string]any) (map[string]any, []core.Loss) {
	converted := translate.OpenAIResponseToAnthropicWithReport(response, model)
	return converted.Value, converted.Report.Losses
}

func chatToResponses(model string, payload map[string]any, stream bool) (map[string]any, []core.Loss, error) {
	rawMessages, _ := payload["messages"].([]any)
	messages := make([]map[string]any, 0, len(rawMessages))
	for _, raw := range rawMessages {
		message, ok := raw.(map[string]any)
		if !ok {
			return nil, nil, errors.New("every Chat message must be an object")
		}
		messages = append(messages, message)
	}
	options := make(map[string]any, len(payload))
	for key, value := range payload {
		switch key {
		case "messages", "model", "stream":
		default:
			options[key] = value
		}
	}
	converted := translate.ChatToResponsesWithReport(model, messages, options, stream)
	return converted.Value, converted.Report.Losses, nil
}

func responsesToChatResponse(model string, response map[string]any) (map[string]any, []core.Loss) {
	converted := translate.ResponsesToChatWithReport(model, response)
	return converted.Value, converted.Report.Losses
}

func responsesToChat(model string, payload map[string]any, stream bool) (map[string]any, []core.Loss, error) {
	converted, err := translate.ResponsesRequestToChatWithReport(payload)
	if err != nil {
		return nil, converted.Report.Losses, err
	}
	return chatBody(model, converted.Value, stream), converted.Report.Losses, nil
}

func chatToResponsesResponse(model string, response map[string]any) (map[string]any, []core.Loss) {
	converted := translate.ChatResponseToResponsesWithReport(model, response)
	return converted.Value, converted.Report.Losses
}
