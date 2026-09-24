package translation_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/translation"
)

// fakeProvider serves one native surface and records what it receives.
type fakeProvider struct {
	native   core.ModelSurface
	response string
	losses   []core.Loss
	frames   []string
	failAt   int // fail the stream after this many frames; zero never fails
	calls    atomic.Int32

	mu       sync.Mutex
	received []core.Request
	closed   atomic.Bool
}

func (p *fakeProvider) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{p.native}
}

func (p *fakeProvider) record(request core.Request) {
	p.calls.Add(1)
	p.mu.Lock()
	defer p.mu.Unlock()
	p.received = append(p.received, request)
}

func (p *fakeProvider) last(t *testing.T) (core.Request, map[string]any) {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.received) == 0 {
		t.Fatal("the provider was not called")
	}
	request := p.received[len(p.received)-1]
	var body map[string]any
	if err := json.Unmarshal(request.Body, &body); err != nil {
		t.Fatalf("the provider received a body that is not JSON: %v", err)
	}
	return request, body
}

func (p *fakeProvider) Invoke(_ context.Context, request core.Request) (core.Response, error) {
	p.record(request)
	if request.Surface != p.native {
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	return core.Response{Body: []byte(p.response), ContentType: core.ContentTypeJSON, Losses: p.losses}, nil
}

func (p *fakeProvider) Stream(_ context.Context, request core.Request) (core.StreamIter, error) {
	p.record(request)
	return &fakeStream{provider: p, frames: p.frames, failAt: p.failAt}, nil
}

func (p *fakeProvider) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	return []core.ModelInfo{{ID: "m"}}, nil
}

type fakeStream struct {
	provider *fakeProvider
	frames   []string
	failAt   int
	sent     int
}

var errUpstream = errors.New("upstream connection reset")

func (s *fakeStream) Next() ([]byte, error) {
	if s.provider.closed.Load() {
		return nil, io.ErrClosedPipe
	}
	if s.failAt > 0 && s.sent == s.failAt {
		return nil, errUpstream
	}
	if s.sent >= len(s.frames) {
		return nil, io.EOF
	}
	frame := s.frames[s.sent]
	s.sent++
	return []byte(frame), nil
}

func (s *fakeStream) Close() error {
	s.provider.closed.Store(true)
	return nil
}

const chatCompletion = `{"id":"chatcmpl-1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`

const responsesObject = `{"id":"resp_1","object":"response","model":"m","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":3,"output_tokens":1,"total_tokens":4}}`

func chatStreamFrames() []string {
	return []string{
		`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}` + "\n\n",
		": keepalive\n\n",
		`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hel"}}]}` + "\n\n",
		`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo"}}]}` + "\n\n",
		`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
}

func jsonRequest(surface core.ModelSurface, body string) core.Request {
	return core.Request{Surface: surface, Model: "m", Body: []byte(body), ContentType: core.ContentTypeJSON}
}

func decode(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal(body, &value); err != nil {
		t.Fatalf("body is not JSON: %v", err)
	}
	return value
}

func TestNativeSurfacesPassThroughUntouched(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, response: chatCompletion}
	adapter := translation.Adapter{Provider: provider}
	request := jsonRequest(core.ModelSurfaceChatCompletions, `{"model":"m","messages":[{"role":"user","content":"hi"}],"seed":7}`)
	response, err := adapter.Invoke(context.Background(), request)
	if err != nil || string(response.Body) != chatCompletion || len(response.Losses) != 0 {
		t.Fatalf("response=%s losses=%v err=%v", response.Body, response.Losses, err)
	}
	if received, _ := provider.last(t); string(received.Body) != string(request.Body) {
		t.Fatal("a native request was rewritten")
	}
}

func TestMessagesOverChat(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, response: chatCompletion}
	adapter := translation.Adapter{Provider: provider}
	response, err := adapter.Invoke(context.Background(), jsonRequest(core.ModelSurfaceMessages,
		`{"model":"m","max_tokens":64,"system":"be brief","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	received, body := provider.last(t)
	messages, _ := body["messages"].([]any)
	if received.Surface != core.ModelSurfaceChatCompletions || body["model"] != "m" || len(messages) != 2 || body["stream"] != nil {
		t.Fatalf("upstream surface=%s body=%v", received.Surface, body)
	}
	message := decode(t, response.Body)
	content, _ := message["content"].([]any)
	first, _ := content[0].(map[string]any)
	if message["type"] != "message" || message["role"] != "assistant" || first["text"] != "hello" {
		t.Fatalf("Messages response=%v", message)
	}
}

func TestChatOverResponses(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceResponses, response: responsesObject}
	adapter := translation.Adapter{Provider: provider}
	response, err := adapter.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0.2,"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	received, body := provider.last(t)
	if received.Surface != core.ModelSurfaceResponses || body["input"] == nil || body["stream"] == true {
		t.Fatalf("upstream surface=%s body=%v", received.Surface, body)
	}
	completion := decode(t, response.Body)
	choices, _ := completion["choices"].([]any)
	choice, _ := choices[0].(map[string]any)
	message, _ := choice["message"].(map[string]any)
	if message["content"] != "hello" {
		t.Fatalf("Chat response=%v", completion)
	}
}

func TestResponsesOverChat(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, response: chatCompletion}
	adapter := translation.Adapter{Provider: provider}
	response, err := adapter.Invoke(context.Background(), jsonRequest(core.ModelSurfaceResponses, `{"model":"m","input":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	received, body := provider.last(t)
	if received.Surface != core.ModelSurfaceChatCompletions || body["messages"] == nil {
		t.Fatalf("upstream surface=%s body=%v", received.Surface, body)
	}
	if object := decode(t, response.Body); !strings.Contains(string(response.Body), `"hello"`) || object["output"] == nil {
		t.Fatalf("Responses response=%s", response.Body)
	}
}

func TestLossyTranslationsAreRejectedByDefaultAndReportedWhenAllowed(t *testing.T) {
	t.Parallel()
	request := jsonRequest(core.ModelSurfaceChatCompletions, `{"model":"m","messages":[{"role":"user","content":"hi"}],"seed":7}`)

	provider := &fakeProvider{native: core.ModelSurfaceResponses, response: responsesObject}
	_, err := translation.Adapter{Provider: provider}.Invoke(context.Background(), request)
	var policyErr *core.LossPolicyError
	if !errors.As(err, &policyErr) || policyErr.Losses[0].Path != "seed" || core.ClassifyError(err).Disposition() != core.DispositionFailover {
		t.Fatalf("err=%v, want the seed loss rejected with failover", err)
	}
	if provider.calls.Load() != 0 {
		t.Fatal("a rejected translation reached the provider")
	}

	allowing := translation.Adapter{Provider: provider, Policy: core.LossPolicy{Rules: []core.LossRule{
		{Path: "seed", Class: translate.LossDropped, Action: core.LossAllow},
	}}}
	response, err := allowing.Invoke(context.Background(), request)
	if err != nil || !slices.ContainsFunc(response.Losses, func(loss core.Loss) bool { return loss.Path == "seed" }) {
		t.Fatalf("losses=%v err=%v, want the allowed seed loss reported", response.Losses, err)
	}
}

func TestProviderLossesJoinTheReport(t *testing.T) {
	t.Parallel()
	adaptation := core.Loss{Path: "temperature", Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "the provider ignores temperature"}
	provider := &fakeProvider{native: core.ModelSurfaceResponses, response: responsesObject, losses: []core.Loss{adaptation}}
	response, err := translation.Adapter{Provider: provider}.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions,
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil || !slices.Contains(response.Losses, adaptation) {
		t.Fatalf("losses=%v err=%v, want the provider's adaptation loss", response.Losses, err)
	}
}

func TestUnsupportedTranslationsAreTypedErrors(t *testing.T) {
	t.Parallel()
	chatOnly := translation.Adapter{Provider: &fakeProvider{native: core.ModelSurfaceChatCompletions, response: chatCompletion}}
	responsesOnly := translation.Adapter{Provider: &fakeProvider{native: core.ModelSurfaceResponses, response: responsesObject}}
	messagesOnly := translation.Adapter{Provider: &fakeProvider{native: core.ModelSurfaceMessages}}
	var surfaceErr *core.SurfaceError
	if _, err := chatOnly.Invoke(context.Background(), jsonRequest(core.ModelSurfaceEmbeddings, `{"input":"x"}`)); !errors.As(err, &surfaceErr) {
		t.Fatalf("embeddings over chat: err=%v", err)
	}
	if _, err := messagesOnly.Invoke(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, `{"messages":[]}`)); !errors.As(err, &surfaceErr) {
		t.Fatalf("chat over messages: err=%v", err)
	}
	if _, err := responsesOnly.Stream(context.Background(), jsonRequest(core.ModelSurfaceChatCompletions, `{"messages":[]}`)); !errors.As(err, &surfaceErr) {
		t.Fatalf("a streamed chat over responses: err=%v", err)
	}
	_, err := chatOnly.Invoke(context.Background(), core.Request{Surface: core.ModelSurfaceMessages, Model: "m", Body: []byte("x"), ContentType: "text/plain"})
	var providerErr *core.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Class != core.ProviderErrorInvalidRequest {
		t.Fatalf("non-JSON body: err=%v", err)
	}
}

func TestSurfacesListNativeThenTranslated(t *testing.T) {
	t.Parallel()
	chat := translation.Adapter{Provider: &fakeProvider{native: core.ModelSurfaceChatCompletions}}
	if got := chat.Surfaces("m"); !slices.Equal(got, []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceMessages, core.ModelSurfaceResponses}) {
		t.Fatalf("chat surfaces=%v", got)
	}
	if got := chat.NativeSurfaces("m"); !slices.Equal(got, []core.ModelSurface{core.ModelSurfaceChatCompletions}) {
		t.Fatalf("native surfaces=%v", got)
	}
	responses := translation.Adapter{Provider: &fakeProvider{native: core.ModelSurfaceResponses}}
	if got := responses.Surfaces("m"); !slices.Equal(got, []core.ModelSurface{core.ModelSurfaceResponses, core.ModelSurfaceChatCompletions}) {
		t.Fatalf("responses surfaces=%v", got)
	}
}

func collect(t *testing.T, stream core.StreamIter) ([]string, error) {
	t.Helper()
	var frames []string
	for {
		frame, err := stream.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return frames, nil
			}
			return frames, err
		}
		frames = append(frames, string(frame))
	}
}

func TestMessagesStreamOverChat(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, frames: chatStreamFrames()}
	stream, err := translation.Adapter{Provider: provider}.Stream(context.Background(), jsonRequest(core.ModelSurfaceMessages,
		`{"model":"m","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if _, body := provider.last(t); body["stream"] != true {
		t.Fatalf("the upstream request did not stream: %v", body)
	}
	frames, err := collect(t, stream)
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(frames, "")
	for _, want := range []string{"event: message_start", `"text":"hel"`, `"text":"lo"`, "event: message_stop"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("stream lacks %q:\n%s", want, joined)
		}
	}
	losses := core.StreamLosses(stream)
	if !slices.ContainsFunc(losses, func(loss core.Loss) bool { return loss.Path == "usage.output_tokens" }) {
		t.Fatalf("losses=%v, want the estimated output usage reported", losses)
	}
}

func TestMessagesStreamReportsUpstreamFailureWithoutCompleting(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, frames: chatStreamFrames(), failAt: 3}
	stream, err := translation.Adapter{Provider: provider}.Stream(context.Background(), jsonRequest(core.ModelSurfaceMessages,
		`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	frames, err := collect(t, stream)
	if !errors.Is(err, errUpstream) {
		t.Fatalf("err=%v, want the upstream failure", err)
	}
	if joined := strings.Join(frames, ""); strings.Contains(joined, "message_stop") || !strings.Contains(joined, `"text":"hel"`) {
		t.Fatalf("a failed stream looked complete or lost its delivered text:\n%s", joined)
	}
}

func TestMessagesStreamCloseStopsTheConversion(t *testing.T) {
	t.Parallel()
	frames := chatStreamFrames()
	long := append([]string(nil), frames[:2]...)
	for range 1000 {
		long = append(long, frames[2])
	}
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, frames: long}
	stream, err := translation.Adapter{Provider: provider}.Stream(context.Background(), jsonRequest(core.ModelSurfaceMessages,
		`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Next(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Close(); err != nil || !provider.closed.Load() {
		t.Fatalf("close err=%v upstream closed=%v", err, provider.closed.Load())
	}
	if _, err := stream.Next(); err == nil {
		t.Fatal("a closed stream kept delivering")
	}
}
