package core_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

type mockProvider struct {
	id           string
	complete     func(context.Context, string, map[string]any, *core.Credential) (map[string]any, error)
	stream       func(context.Context, string, map[string]any, *core.Credential) (core.StreamIter, error)
	completeCall int
	streamCall   int
}

func (m *mockProvider) Complete(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (map[string]any, error) {
	m.completeCall++
	if m.complete != nil {
		return m.complete(ctx, model, payload, cred)
	}
	return map[string]any{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "Hello from mock provider " + m.id,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     10,
			"completion_tokens": 8,
			"total_tokens":      18,
		},
	}, nil
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func (m *mockProvider) Stream(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (core.StreamIter, error) {
	m.streamCall++
	if m.stream != nil {
		return m.stream(ctx, model, payload, cred)
	}
	return &testStreamIter{}, nil
}

type testStreamIter struct {
	chunks   [][]byte
	err      error
	closeErr error
	closed   bool
}

func (s *testStreamIter) Next() ([]byte, error) {
	if len(s.chunks) == 0 {
		if s.err != nil {
			err := s.err
			s.err = nil
			return nil, err
		}
		return nil, io.EOF
	}
	chunk := s.chunks[0]
	s.chunks = s.chunks[1:]
	return chunk, nil
}

func (s *testStreamIter) Close() error {
	s.closed = true
	return s.closeErr
}

func (m *mockProvider) ListModels(ctx context.Context, cred *core.Credential) ([]core.ModelInfo, error) {
	return []core.ModelInfo{
		{ID: "mock-model", Object: "model", OwnedBy: m.id},
	}, nil
}

func TestEngineChatCompletionsAndAnthropicMessages(t *testing.T) {
	var capturedRecord *core.UsageRecord
	cfg := core.Config{
		Routes: map[string]core.RouteConfig{
			"smart": {
				Targets: []core.Target{
					{Provider: "mock1", Model: "model-alpha"},
				},
			},
		},
		UsageHook: core.UsageHookFunc(func(ctx context.Context, record core.UsageRecord) {
			capturedRecord = &record
		}),
	}

	engine := core.NewEngine(cfg)
	engine.RegisterProvider("mock1", &mockProvider{id: "mock1"})

	// 1. Test /v1/models
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("GET /v1/models status=%d", w.Code)
	}

	// 2. Test /v1/chat/completions (OpenAI format)
	chatReqBody, _ := json.Marshal(map[string]any{
		"model": "smart",
		"messages": []map[string]string{
			{"role": "user", "content": "hi"},
		},
	})
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(chatReqBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("POST /v1/chat/completions status=%d body=%s", w.Code, w.Body.String())
	}
	if capturedRecord == nil || capturedRecord.Provider != "mock1" {
		t.Fatalf("expected usage record captured, got: %+v", capturedRecord)
	}

	// 3. Test /v1/messages (Anthropic Messages API format)
	anthropicReqBody, _ := json.Marshal(map[string]any{
		"model":      "mock1/model-alpha",
		"max_tokens": 1024,
		"messages": []map[string]any{
			{"role": "user", "content": "hello claude"},
		},
	})
	req = httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(anthropicReqBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("POST /v1/messages status=%d body=%s", w.Code, w.Body.String())
	}

	var anthropicResp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &anthropicResp); err != nil {
		t.Fatalf("failed unmarshaling anthropic response: %v", err)
	}
	if anthropicResp["role"] != "assistant" {
		t.Fatalf("expected assistant role in anthropic response, got: %+v", anthropicResp)
	}
}

func TestEngineStreamsProviderSSEFrames(t *testing.T) {
	iter := &testStreamIter{chunks: [][]byte{
		[]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"),
		[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"),
		[]byte("data: [DONE]\n\n"),
	}}
	provider := &mockProvider{id: "stream", stream: func(context.Context, string, map[string]any, *core.Credential) (core.StreamIter, error) {
		return iter, nil
	}}
	engine := core.NewEngine(core.Config{Routes: map[string]core.RouteConfig{
		"streaming": {Targets: []core.Target{{Provider: "stream", Model: "exact"}}},
	}})
	engine.RegisterProvider("stream", provider)

	body := []byte(`{"model":"streaming","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)

	want := "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n" +
		"data: [DONE]\n\n"
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "text/event-stream" || w.Body.String() != want {
		t.Fatalf("stream status=%d headers=%v body=%q", w.Code, w.Header(), w.Body.String())
	}
	if !iter.closed {
		t.Fatal("stream iterator was not closed")
	}
}

func TestEngineChatStreamAcceptsValidDoneSSEField(t *testing.T) {
	iter := &testStreamIter{chunks: [][]byte{[]byte("event: complete\r\ndata:[DONE]\r\n\r\n")}}
	provider := &mockProvider{id: "stream", stream: func(context.Context, string, map[string]any, *core.Credential) (core.StreamIter, error) {
		return iter, nil
	}}
	engine := core.NewEngine(core.Config{Routes: map[string]core.RouteConfig{
		"route": {Targets: []core.Target{{Provider: "stream", Model: "exact"}}},
	}})
	engine.RegisterProvider("stream", provider)

	body := []byte(`{"model":"route","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))

	if w.Code != http.StatusOK || w.Body.String() != "event: complete\r\ndata:[DONE]\r\n\r\n" {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
}

func TestEngineChatStreamRejectsInvalidTermination(t *testing.T) {
	tests := []struct {
		name string
		iter *testStreamIter
		want string
	}{
		{name: "EOF before terminal", iter: &testStreamIter{chunks: [][]byte{[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")}}, want: "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"},
		{name: "duplicate terminal", iter: &testStreamIter{chunks: [][]byte{[]byte("data: [DONE]\n\n"), []byte("data: [DONE]\n\n")}}, want: "data: [DONE]\n\n"},
		{name: "event after terminal", iter: &testStreamIter{chunks: [][]byte{[]byte("data: [DONE]\n\n"), []byte("data: {\"late\":true}\n\n")}}, want: "data: [DONE]\n\n"},
		{name: "error after terminal", iter: &testStreamIter{chunks: [][]byte{[]byte("data: [DONE]\n\n")}, err: errors.New("late failure")}, want: "data: [DONE]\n\n"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var record core.UsageRecord
			provider := &mockProvider{id: "stream", stream: func(context.Context, string, map[string]any, *core.Credential) (core.StreamIter, error) {
				return test.iter, nil
			}}
			engine := core.NewEngine(core.Config{
				Routes:    map[string]core.RouteConfig{"route": {Targets: []core.Target{{Provider: "stream", Model: "exact"}}}},
				UsageHook: core.UsageHookFunc(func(_ context.Context, usage core.UsageRecord) { record = usage }),
			})
			engine.RegisterProvider("stream", provider)

			body := []byte(`{"model":"route","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
			w := httptest.NewRecorder()
			engine.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))

			if w.Body.String() != test.want {
				t.Fatalf("body=%q, want %q", w.Body.String(), test.want)
			}
			if record.StatusCode != http.StatusBadGateway || record.Error == "" {
				t.Fatalf("usage=%+v, want recorded stream failure", record)
			}
			if !test.iter.closed {
				t.Fatal("stream iterator was not closed")
			}
		})
	}
}

func TestEngineChatStreamReturnsAndRecordsPreOutputError(t *testing.T) {
	streamErr := &providers.InvocationError{Msg: "stream failed", Status: http.StatusUnauthorized}
	iter := &testStreamIter{err: streamErr}
	var record core.UsageRecord
	provider := &mockProvider{id: "stream", stream: func(context.Context, string, map[string]any, *core.Credential) (core.StreamIter, error) {
		return iter, nil
	}}
	engine := core.NewEngine(core.Config{
		Routes:    map[string]core.RouteConfig{"route": {Targets: []core.Target{{Provider: "stream", Model: "exact"}}}},
		UsageHook: core.UsageHookFunc(func(_ context.Context, usage core.UsageRecord) { record = usage }),
	})
	engine.RegisterProvider("stream", provider)

	body := []byte(`{"model":"route","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))

	if w.Code != http.StatusBadGateway || !strings.Contains(w.Body.String(), streamErr.Error()) {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
	if record.Provider != "stream" || record.Model != "exact" || record.Error != streamErr.Error() {
		t.Fatalf("usage=%+v", record)
	}
	if !iter.closed {
		t.Fatal("stream iterator was not closed")
	}
}

func TestEngineDoesNotFailOverCallerCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	first := &mockProvider{id: "first", stream: func(context.Context, string, map[string]any, *core.Credential) (core.StreamIter, error) {
		cancel()
		return nil, context.Canceled
	}}
	second := &mockProvider{id: "second", stream: func(context.Context, string, map[string]any, *core.Credential) (core.StreamIter, error) {
		return nil, errors.New("second target must not run")
	}}
	engine := core.NewEngine(core.Config{Routes: map[string]core.RouteConfig{
		"route": {Targets: []core.Target{{Provider: "first", Model: "one"}, {Provider: "second", Model: "two"}}},
	}})
	engine.RegisterProvider("first", first)
	engine.RegisterProvider("second", second)

	body := []byte(`{"model":"route","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)).WithContext(ctx)
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)

	if w.Code != http.StatusBadGateway || first.streamCall != 1 || second.streamCall != 0 {
		t.Fatalf("status=%d first calls=%d second calls=%d", w.Code, first.streamCall, second.streamCall)
	}
}

func TestEngineFailsOverLegacyProviderFailuresInOrder(t *testing.T) {
	tests := []struct {
		name     string
		provider any
	}{
		{
			name: "openai rate limit",
			provider: providers.NewOpenAIProvider("openai", "http://upstream.invalid", "none", &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusTooManyRequests, Body: io.NopCloser(strings.NewReader("limited")), Header: make(http.Header)}, nil
			})}),
		},
		{
			name: "anthropic server error",
			provider: providers.NewAnthropicProvider("anthropic", "http://upstream.invalid", "key", &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusServiceUnavailable, Body: io.NopCloser(strings.NewReader("unavailable")), Header: make(http.Header)}, nil
			})}),
		},
		{
			name: "openai transport timeout",
			provider: providers.NewOpenAIProvider("openai", "http://upstream.invalid", "none", &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
				return nil, context.DeadlineExceeded
			})}),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fallback := &mockProvider{id: "fallback"}
			engine := core.NewEngine(core.Config{Routes: map[string]core.RouteConfig{
				"route": {Targets: []core.Target{{Provider: "first", Model: "one"}, {Provider: "fallback", Model: "two"}}},
			}})
			engine.RegisterProvider("first", test.provider)
			engine.RegisterProvider("fallback", fallback)

			body := []byte(`{"model":"route","messages":[{"role":"user","content":"hi"}]}`)
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
			w := httptest.NewRecorder()
			engine.ServeHTTP(w, req)

			if w.Code != http.StatusOK || fallback.completeCall != 1 {
				t.Fatalf("status=%d fallback calls=%d body=%s", w.Code, fallback.completeCall, w.Body.String())
			}
		})
	}
}

func TestEngineDoesNotFailOverLegacyProviderAuthFailure(t *testing.T) {
	first := providers.NewOpenAIProvider("openai", "http://upstream.invalid", "key", &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusUnauthorized, Body: io.NopCloser(strings.NewReader("unauthorized")), Header: make(http.Header)}, nil
	})})
	fallback := &mockProvider{id: "fallback"}
	engine := core.NewEngine(core.Config{Routes: map[string]core.RouteConfig{
		"route": {Targets: []core.Target{{Provider: "first", Model: "one"}, {Provider: "fallback", Model: "two"}}},
	}})
	engine.RegisterProvider("first", first)
	engine.RegisterProvider("fallback", fallback)

	body := []byte(`{"model":"route","messages":[{"role":"user","content":"hi"}]}`)
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))

	if w.Code != http.StatusBadGateway || fallback.completeCall != 0 {
		t.Fatalf("status=%d fallback calls=%d body=%s", w.Code, fallback.completeCall, w.Body.String())
	}
}

func TestEngineAnthropicTranslationReceivesFramedProviderSSE(t *testing.T) {
	stream := "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	provider := providers.NewOpenAIProvider("openai", "http://upstream.invalid", "none", &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: &chunkedReadCloser{chunks: [][]byte{
			[]byte(stream[:17]), []byte(stream[17:43]), []byte(stream[43:]),
		}}, Header: http.Header{"Content-Type": []string{"text/event-stream"}}}, nil
	})})
	engine := core.NewEngine(core.Config{})
	engine.RegisterProvider("openai", provider)

	body := []byte(`{"model":"openai/exact","stream":true,"max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`)
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body)))

	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "hello") || !strings.Contains(w.Body.String(), " world") {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
}

func TestEngineAnthropicTranslationDoesNotStopAfterSourceFailure(t *testing.T) {
	tests := []struct {
		name       string
		ctx        context.Context
		iter       *testStreamIter
		wantStatus int
		wantError  bool
	}{
		{name: "iterator error before output", ctx: context.Background(), iter: &testStreamIter{err: errors.New("read failed")}, wantStatus: http.StatusBadGateway},
		{name: "iterator error after output", ctx: context.Background(), iter: &testStreamIter{chunks: [][]byte{[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")}, err: errors.New("read failed")}, wantStatus: http.StatusOK, wantError: true},
		{name: "truncated EOF", ctx: context.Background(), iter: &testStreamIter{chunks: [][]byte{[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")}}, wantStatus: http.StatusOK, wantError: true},
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	tests = append(tests, struct {
		name       string
		ctx        context.Context
		iter       *testStreamIter
		wantStatus int
		wantError  bool
	}{name: "caller cancellation", ctx: canceled, iter: &testStreamIter{err: context.Canceled}, wantStatus: http.StatusBadGateway})
	tests = append(tests, struct {
		name       string
		ctx        context.Context
		iter       *testStreamIter
		wantStatus int
		wantError  bool
	}{name: "caller cancellation after output", ctx: canceled, iter: &testStreamIter{chunks: [][]byte{[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")}, err: context.Canceled}, wantStatus: http.StatusOK})

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			provider := &mockProvider{id: "stream", stream: func(context.Context, string, map[string]any, *core.Credential) (core.StreamIter, error) {
				return test.iter, nil
			}}
			engine := core.NewEngine(core.Config{})
			engine.RegisterProvider("stream", provider)

			body := []byte(`{"model":"stream/exact","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
			w := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body)).WithContext(test.ctx)
			engine.ServeHTTP(w, req)

			if w.Code != test.wantStatus {
				t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
			}
			if strings.Contains(w.Body.String(), "message_stop") {
				t.Fatalf("source failure was translated to success: %q", w.Body.String())
			}
			if got := strings.Contains(w.Body.String(), "event: error"); got != test.wantError {
				t.Fatalf("error event present=%v, want %v; body=%q", got, test.wantError, w.Body.String())
			}
			if !test.iter.closed {
				t.Fatal("stream iterator was not closed")
			}
		})
	}
}

type chunkedReadCloser struct {
	chunks [][]byte
}

func (r *chunkedReadCloser) Read(p []byte) (int, error) {
	if len(r.chunks) == 0 {
		return 0, io.EOF
	}
	chunk := r.chunks[0]
	r.chunks = r.chunks[1:]
	return copy(p, chunk), nil
}

func (*chunkedReadCloser) Close() error { return nil }
