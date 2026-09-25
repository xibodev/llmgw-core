package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"slices"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

const (
	openAIResponsesAnswer = `{"id":"resp_fixture","object":"response","status":"completed","provider_extra":{"kept":true},` +
		`"output":[{"type":"message","id":"msg_fixture","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}]}`
	openAIResponsesEvents = ": keepalive\n\n" +
		"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_fixture\",\"output\":[]}}\n\n" +
		"event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\r\n\r\n" +
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fixture\",\"output\":[]}}\n\n"
)

func answerOpenAIResponses(w http.ResponseWriter, r *http.Request, _ int) {
	if r.Header.Get("Accept") == core.ContentTypeEventStream {
		w.Header().Set("Content-Type", core.ContentTypeEventStream)
		_, _ = io.WriteString(w, openAIResponsesEvents)
		return
	}
	_, _ = io.WriteString(w, openAIResponsesAnswer)
}

// A native Responses request is the caller's body with the request's
// model and the operation's stream flag, without force_api_support, and
// its answer comes back as sent.
func TestOpenAICompatibleForwardsResponsesAsSent(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, answerOpenAIResponses)
	provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) {
		config.Models = openAIRows(map[string]core.ModelInfo{"responses-fixture": {SupportedAPIs: []string{"/responses"}}})
	})
	body := `{"model":"ignored","stream":true,"input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AA=="}]}],` +
		`"force_api_support":true,"store":false,"previous_response_id":"resp_prior","text":{"format":{"type":"text"}}}`
	request := openAIRequest(core.ModelSurfaceResponses, "responses-fixture", body, &core.Credential{APIKey: "fixture-key"})
	response, err := provider.Invoke(context.Background(), request)
	if err != nil || string(response.Body) != openAIResponsesAnswer || len(response.Losses) != 0 {
		t.Fatalf("response = %s, err = %v", response.Body, err)
	}
	stream, err := provider.Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var frames strings.Builder
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		frames.Write(frame)
	}
	if want := strings.TrimPrefix(openAIResponsesEvents, ": keepalive\n\n"); frames.String() != want {
		t.Fatalf("frames = %q", frames.String())
	}
	sent := `{"input":[{"content":[{"image_url":"data:image/png;base64,AA==","type":"input_image"}],"role":"user"}],"model":"responses-fixture",` +
		`"previous_response_id":"resp_prior","store":false,"stream":%s,"text":{"format":{"type":"text"}}}`
	calls := backend.take()
	if len(calls) != 2 {
		t.Fatalf("upstream = %+v", calls)
	}
	for index, want := range []struct{ stream, accept string }{{"false", core.ContentTypeJSON}, {"true", core.ContentTypeEventStream}} {
		call := calls[index]
		if call.path != "/v1/responses" || call.body != strings.Replace(sent, "%s", want.stream, 1) || call.accept != want.accept ||
			call.vision != "true" || call.authorization != "Bearer fixture-key" {
			t.Fatalf("upstream %d = %+v", index, call)
		}
	}
}

// Not finding the Responses endpoint means the model does not serve it
// natively after all, which another target may.
func TestOpenAICompatibleReportsAMissingResponsesEndpointAsASurface(t *testing.T) {
	t.Parallel()
	for _, status := range []int{http.StatusNotFound, http.StatusMethodNotAllowed} {
		_, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) { w.WriteHeader(status) })
		provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) { config.RegistryID = "openai" })
		request := openAIRequest(core.ModelSurfaceResponses, "gpt-fixture", `{"input":"hi"}`, nil)
		_, invokeErr := provider.Invoke(context.Background(), request)
		_, streamErr := provider.Stream(context.Background(), request)
		for _, err := range []error{invokeErr, streamErr} {
			var surface *core.SurfaceError
			if !errors.As(err, &surface) || surface.Surface != core.ModelSurfaceResponses || surface.Model != "gpt-fixture" {
				t.Fatalf("%d: err = %v", status, err)
			}
		}
	}
}

// Ported from the gateway's TestV043ChatPayloadPreservesFallbackControls.
func TestOpenAICompatibleChatKeepsTheFallbackControls(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, answerOpenAIChat)
	provider := newTestOpenAICompatible(t, server, nil)
	body := `{"messages":[{"role":"user","content":"hi"}],"metadata":{"client":"fixture"},"parallel_tool_calls":false,` +
		`"reasoning_effort":"high","thinking":{"type":"disabled"}}`
	if _, err := provider.Stream(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "m", body, nil)); err != nil {
		t.Fatal(err)
	}
	want := `{"messages":[{"content":"hi","role":"user"}],"metadata":{"client":"fixture"},"model":"m","parallel_tool_calls":false,` +
		`"reasoning_effort":"high","stream":true,"thinking":{"type":"disabled"}}`
	if calls := backend.take(); len(calls) != 1 || calls[0].body != want || calls[0].accept != "" {
		t.Fatalf("upstream = %+v", calls)
	}
}

func openAIResponsesOnly(model string) (core.ModelInfo, bool) {
	return core.ModelInfo{ID: model, SupportedAPIs: []string{"/responses"}}, strings.HasPrefix(model, "responses-")
}

func lossPathsOf(losses []core.Loss) []string {
	paths := lossPaths(losses)
	slices.Sort(paths)
	return paths
}

// Ported from the gateway's TestChatToResponsesTranslationLossPolicy: with
// adaptation on, Chat for a model that lists only Responses is served over
// Responses, and a material conversion loss refuses it before anything is
// sent. Fields the gateway's Chat facade never hands its conversion are
// dropped instead.
func TestOpenAICompatibleAdaptsChatToResponsesUnderTheLossPolicy(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, answerOpenAIResponses)
	provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) {
		config.Models, config.ForceAPISupport = openAIResponsesOnly, true
	})
	stop := openAIRequest(core.ModelSurfaceChatCompletions, "responses-fixture", `{"messages":[{"role":"user","content":"hi"}],"stop":["END"]}`, nil)
	var failure *core.ProviderError
	if _, err := provider.Invoke(context.Background(), stop); !errors.As(err, &failure) || failure.Class != core.ProviderErrorUnsupported ||
		!failure.Classification.FailoverEligible || !strings.Contains(err.Error(), "Responses") {
		t.Fatalf("material loss: err = %v", err)
	}
	if calls := backend.take(); len(calls) != 0 {
		t.Fatalf("a refused conversion reached the upstream: %+v", calls)
	}
	body := `{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream_options":{"include_usage":true},"parallel_tool_calls":true}`
	response, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "responses-fixture", body, nil))
	if err != nil || !strings.Contains(string(response.Body), `"object":"chat.completion"`) || strings.Contains(string(response.Body), "forced_support") {
		t.Fatalf("response = %s, err = %v", response.Body, err)
	}
	if got, want := lossPathsOf(response.Losses), []string{
		"advisory dropped parallel_tool_calls", "advisory dropped stream_options", "advisory renamed max_tokens",
	}; !slices.Equal(got, want) {
		t.Fatalf("losses = %v, want %v", got, want)
	}
	want := `{"input":[{"content":"hi","role":"user"}],"max_output_tokens":8,"model":"responses-fixture"}`
	if calls := backend.take(); len(calls) != 1 || calls[0].path != "/v1/responses" || calls[0].body != want || calls[0].accept != core.ContentTypeJSON {
		t.Fatalf("upstream = %+v", calls)
	}
}

// A 400 naming a sampling parameter is sent once more without it, with the
// same headers, as the gateway retries Chat it serves over Responses.
func TestOpenAICompatibleRetriesAResponsesRefusalWithoutTheNamedParameter(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, func(w http.ResponseWriter, r *http.Request, attempt int) {
		if attempt%2 == 1 {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"message":"Unsupported parameter: 'temperature' is not supported with this model."}}`)
			return
		}
		answerOpenAIResponses(w, r, attempt)
	})
	provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) { config.Models = openAIResponsesOnly })
	body := `{"force_api_support":true,"temperature":0.2,"top_p":0.9,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}`
	request := openAIRequest(core.ModelSurfaceChatCompletions, "responses-fixture", body, nil)
	response, err := provider.Invoke(context.Background(), request)
	if err != nil || !strings.Contains(string(response.Body), `"forced_support":{"req_api":"chat","resp_api":"responses"}`) {
		t.Fatalf("response = %s, err = %v", response.Body, err)
	}
	if !slices.Contains(lossPaths(response.Losses), "advisory dropped temperature") || slices.Contains(lossPaths(response.Losses), "advisory dropped top_p") {
		t.Fatalf("losses = %v", lossPaths(response.Losses))
	}
	stream, err := provider.Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	var frames []string
	for frame, err := stream.Next(); err != io.EOF; frame, err = stream.Next() {
		if err != nil {
			t.Fatal(err)
		}
		frames = append(frames, string(frame))
	}
	if len(frames) < 2 || !strings.Contains(frames[0], `"object":"chat.completion.chunk"`) || frames[len(frames)-1] != "data: [DONE]\n\n" ||
		!slices.Contains(lossPaths(core.StreamLosses(stream)), "advisory dropped temperature") {
		t.Fatalf("frames = %q, losses = %v", frames, core.StreamLosses(stream))
	}
	calls := backend.take()
	if len(calls) != 4 {
		t.Fatalf("upstream = %+v", calls)
	}
	for index, call := range calls {
		if call.vision != "true" || call.accept != core.ContentTypeJSON || strings.Contains(call.body, `"stream":true`) ||
			strings.Contains(call.body, "temperature") == (index%2 == 1) || !strings.Contains(call.body, `"top_p":0.9`) {
			t.Fatalf("upstream %d = %+v", index, call)
		}
	}
}

// A retry the upstream cannot be reached for leaves the first refusal.
func TestOpenAICompatibleKeepsTheRefusalWhenItsRetryFails(t *testing.T) {
	t.Parallel()
	_, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, attempt int) {
		if attempt > 1 {
			panic(http.ErrAbortHandler)
		}
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"top_p is not supported"}}`)
	})
	provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) {
		config.Models, config.ForceAPISupport = openAIResponsesOnly, true
	})
	body := `{"top_p":0.5,"messages":[{"role":"user","content":"hi"}]}`
	_, err := provider.Invoke(context.Background(), openAIRequest(core.ModelSurfaceChatCompletions, "responses-fixture", body, nil))
	if classification := core.ClassifyError(err); classification.StatusCode != http.StatusBadRequest || classification.Retryable {
		t.Fatalf("err = %v, classification = %+v", err, classification)
	}
}
