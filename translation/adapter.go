// Package translation serves surfaces a provider lacks by translating through
// llm-translate. Every conversion reports its losses. An Adapter checks them
// against a core.LossPolicy and returns all of them, including the losses the
// policy allows.
package translation

import (
	"context"
	"encoding/json"
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
//   - Responses over a Chat Completions provider;
//   - Chat Completions over a Messages provider, streaming included.
//
// When a provider serves more than one surface a route could target, the
// first route in that order is used. Serves reports what an Adapter serves
// for a set of native surfaces.
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

// route converts one client surface to one native surface and back. stream,
// when set, renders the native surface's stream as the client surface's.
type route struct {
	from, target core.ModelSurface
	request      func(model string, payload map[string]any, stream bool) (map[string]any, []core.Loss, error)
	response     func(model string, response map[string]any) (map[string]any, []core.Loss)
	stream       func(upstream core.StreamIter, model string, payload map[string]any, losses []core.Loss) core.StreamIter
}

func routes() []route {
	return []route{
		{from: core.ModelSurfaceMessages, target: core.ModelSurfaceChatCompletions, request: messagesToChat, response: chatToMessagesResponse, stream: func(upstream core.StreamIter, model string, _ map[string]any, losses []core.Loss) core.StreamIter {
			return newMessagesStream(upstream, model, losses)
		}},
		{from: core.ModelSurfaceChatCompletions, target: core.ModelSurfaceResponses, request: chatToResponses, response: responsesToChatResponse},
		{from: core.ModelSurfaceResponses, target: core.ModelSurfaceChatCompletions, request: responsesToChat, response: chatToResponsesResponse},
		{from: core.ModelSurfaceChatCompletions, target: core.ModelSurfaceMessages, request: chatToMessages, response: messagesToChatResponse, stream: func(upstream core.StreamIter, model string, payload map[string]any, losses []core.Loss) core.StreamIter {
			return newChatStream(upstream, model, includesUsage(payload), losses)
		}},
	}
}

// Serves reports whether an Adapter in front of a provider with the native
// surfaces serves target: natively, or through a route to one of them. A
// route serves Invoke; Stream also needs a route that streams, which
// ServesStream reports. It agrees with Adapter.Surfaces by construction.
func Serves(target core.ModelSurface, native ...core.ModelSurface) bool {
	_, ok := routeFor(target, native, false)
	return ok || slices.Contains(native, target)
}

// ServesStream reports whether an Adapter in front of a provider with the
// native surfaces streams target: natively, or through a route that streams.
func ServesStream(target core.ModelSurface, native ...core.ModelSurface) bool {
	_, ok := routeFor(target, native, true)
	return ok || slices.Contains(native, target)
}

// ServesChat reports whether an Adapter in front of a provider with the
// native surfaces serves Chat Completions.
func ServesChat(native ...core.ModelSurface) bool {
	return Serves(core.ModelSurfaceChatCompletions, native...)
}

// routeFor returns the first route from target to one of the native
// surfaces, one that streams when stream is set.
func routeFor(target core.ModelSurface, native []core.ModelSurface, stream bool) (route, bool) {
	for _, candidate := range routes() {
		if candidate.from == target && slices.Contains(native, candidate.target) && (!stream || candidate.stream != nil) {
			return candidate, true
		}
	}
	return route{}, false
}

// NativeSurfaces reports the wrapped provider's native surfaces: translated
// surfaces are, by definition, not native.
func (a Adapter) NativeSurfaces(model string) []core.ModelSurface {
	return a.Provider.NativeSurfaces(model)
}

// Surfaces lists every surface the adapter serves for model, native first.
func (a Adapter) Surfaces(model string) []core.ModelSurface {
	native := a.Provider.NativeSurfaces(model)
	surfaces := append([]core.ModelSurface(nil), native...)
	for _, candidate := range routes() {
		if !slices.Contains(surfaces, candidate.from) && Serves(candidate.from, native...) {
			surfaces = append(surfaces, candidate.from)
		}
	}
	return surfaces
}

// ListModels delegates to the wrapped provider.
func (a Adapter) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	return a.Provider.ListModels(ctx, credential)
}

// Unwrap returns the wrapped provider. The adapter hands it the requests of
// its native surfaces unchanged, so core.PreservesWire and core.CountTokens
// read the wrapped provider's declarations through the adapter.
func (a Adapter) Unwrap() core.Provider { return a.Provider }

func (a Adapter) route(model string, surface core.ModelSurface, stream bool) (route, bool) {
	return routeFor(surface, a.Provider.NativeSurfaces(model), stream)
}

// translateRequest converts a request to the route's native surface and
// enforces the policy on the conversion's losses. It also returns the
// client's payload, which a translated stream reads.
func (a Adapter) translateRequest(r route, request core.Request, stream bool) (core.Request, map[string]any, []core.Loss, error) {
	payload, err := jsonPayload(request)
	if err != nil {
		return core.Request{}, nil, nil, err
	}
	translated, losses, err := r.request(request.Model, payload, stream)
	if err != nil {
		return core.Request{}, payload, losses, &core.ProviderError{Message: "the request cannot be translated: " + err.Error(), Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	if err := a.Policy.Check(losses); err != nil {
		return core.Request{}, payload, losses, err
	}
	body, err := json.Marshal(translated)
	if err != nil {
		return core.Request{}, payload, losses, err
	}
	upstream := request
	upstream.Surface, upstream.Body, upstream.ContentType = r.target, body, core.ContentTypeJSON
	return upstream, payload, losses, nil
}

// Invoke serves request natively when the provider can, and otherwise by
// translation.
func (a Adapter) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	if core.ServesNatively(a.Provider, request.Model, request.Surface) {
		return a.Provider.Invoke(ctx, request)
	}
	r, ok := a.route(request.Model, request.Surface, false)
	if !ok {
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	upstream, _, losses, err := a.translateRequest(r, request, false)
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
	r, ok := a.route(request.Model, request.Surface, true)
	if !ok {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	upstream, payload, losses, err := a.translateRequest(r, request, true)
	if err != nil {
		return nil, err
	}
	stream, err := a.Provider.Stream(ctx, upstream)
	if err != nil {
		return nil, err
	}
	losses = append(losses, core.StreamLosses(stream)...)
	return r.stream(stream, request.Model, payload, losses), nil
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
	messages, err := chatMessages(payload)
	if err != nil {
		return nil, nil, err
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
