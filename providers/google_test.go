package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// googleCall is one request a fake Google received.
type googleCall struct {
	method, path, query                string
	authorization, apiKey, requestType string
	body                               string
}

// googleFake is a synthetic Google API. It records every request and
// answers with reply. The response fixtures are the shapes the gateway's
// tests captured from the live services.
type googleFake struct {
	mu    sync.Mutex
	calls []googleCall
	reply func(*http.Request) (int, string)
}

func newGoogleFake(t *testing.T, reply func(*http.Request) (int, string)) (*googleFake, string) {
	t.Helper()
	fake := &googleFake{reply: reply}
	server := httptest.NewServer(fake)
	t.Cleanup(server.Close)
	return fake, server.URL
}

// googleAnswer answers every request with one status and body.
func googleAnswer(status int, body string) func(*http.Request) (int, string) {
	return func(*http.Request) (int, string) { return status, body }
}

func (f *googleFake) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	f.mu.Lock()
	f.calls = append(f.calls, googleCall{
		method: r.Method, path: r.URL.Path, query: r.URL.RawQuery,
		authorization: r.Header.Get("Authorization"), apiKey: r.Header.Get("x-goog-api-key"),
		requestType: r.Header.Get("X-Vertex-AI-LLM-Request-Type"), body: string(body),
	})
	f.mu.Unlock()
	status, answer := f.reply(r)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = io.WriteString(w, answer)
}

// take returns the requests since the last take.
func (f *googleFake) take() []googleCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	calls := f.calls
	f.calls = nil
	return calls
}

func newGoogleTest(t *testing.T, config GoogleConfig) *Google {
	t.Helper()
	provider, err := NewGoogle(config)
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func googleKey(key string) *core.Credential {
	return &core.Credential{APIKey: key, TokenType: core.TokenTypeAPIKey}
}

func googleBearer(token string) *core.Credential {
	return &core.Credential{Token: token, TokenType: "Bearer"}
}

func googleChatRequest(model, body string, credential *core.Credential) core.Request {
	return core.Request{Surface: core.ModelSurfaceChatCompletions, Model: model, Body: []byte(body), ContentType: core.ContentTypeJSON, Credential: credential}
}

const googleHelloAnswer = `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`

func TestGoogleModelURLDiffersPerDeployment(t *testing.T) {
	t.Parallel()
	studio := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio})
	if got, err := studio.modelURL("models/gemini-3.5-flash", "", "generateContent"); err != nil ||
		got != "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent" {
		t.Fatalf("ai_studio url = %q, %v", got, err)
	}
	// The global endpoint has no location prefix in the host.
	global := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI})
	want := "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/global/publishers/google/models/gemini-3.5-flash:generateContent"
	if got, err := global.modelURL("gemini-3.5-flash", "proj-1", "generateContent"); err != nil || got != want {
		t.Fatalf("vertex global url = %q, %v, want %q", got, err, want)
	}
	// A regional location prefixes the host and appears in the path.
	regional := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, Location: "us-central1"})
	want = "https://us-central1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-central1/publishers/google/models/veo-3.0-generate-001:predictLongRunning"
	if got, err := regional.modelURL("veo-3.0-generate-001", "proj-1", "predictLongRunning"); err != nil || got != want {
		t.Fatalf("vertex regional url = %q, %v, want %q", got, err, want)
	}
	var failure *core.ProviderError
	if _, err := global.modelURL("m", "", "generateContent"); !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration {
		t.Fatalf("vertex without a project: %v, want a configuration error rather than a broken url", err)
	}
	// The gateway interpolates the model unescaped; one that would reshape
	// the URL is refused rather than sent.
	for _, model := range []string{"models/", "gemini/../x", "gemini?alt=sse", "gemini#x", "gemini%2Fx", "gemini x", "tunedModels/x"} {
		if _, err := studio.modelURL(model, "", "generateContent"); !errors.As(err, &failure) || failure.Class != core.ProviderErrorInvalidRequest {
			t.Errorf("model %q: %v, want an invalid request", model, err)
		}
	}
}

// Ported from the gateway's TestCompleteTranslatesGeminiShape, with the
// bytes the gateway sends and returns.
func TestGoogleChatTranslatesGeminiShape(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, `{
          "candidates":[{"content":{"role":"model","parts":[{"text":"VERTEX OK"}]}}],
          "usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":2,"totalTokenCount":95},
          "modelVersion":"gemini-3.5-flash"}`))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	response, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"model":"gemini-3.5-flash","messages":[`+
		`{"role":"system","content":"be terse"},{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}],"max_tokens":16}`, googleKey("secret")))
	if err != nil {
		t.Fatal(err)
	}
	// System messages become systemInstruction; assistant becomes "model".
	want := googleCall{
		method: http.MethodPost, path: "/models/gemini-3.5-flash:generateContent", apiKey: "secret",
		body: `{"contents":[{"parts":[{"text":"hi"}],"role":"user"},{"parts":[{"text":"hello"}],"role":"model"}],` +
			`"generationConfig":{"maxOutputTokens":16},"systemInstruction":{"parts":[{"text":"be terse"}]}}`,
	}
	// The key travels in x-goog-api-key, never in the URL.
	if calls := fake.take(); !reflect.DeepEqual(calls, []googleCall{want}) {
		t.Fatalf("upstream = %+v", calls)
	}
	body := `{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"VERTEX OK","role":"assistant"}}],` +
		`"id":"chatcmpl-google","model":"gemini-3.5-flash","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":7,"total_tokens":95}}`
	if string(response.Body) != body || response.ContentType != core.ContentTypeJSON || len(response.Losses) != 0 {
		t.Fatalf("response = %s %q losses=%v", response.Body, response.ContentType, response.Losses)
	}
}

// The gateway maps a message's role and string content and nothing else,
// so Google reports everything else it drops. The body keeps the gateway's
// bytes, HTML escaping and float64 numbers included.
func TestGoogleChatReportsWhatItDrops(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	response, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"stream":true,"messages":[null,{"role":5,"content":["x"]},`+
		`{"role":"tool","content":"r","tool_call_id":"c1"},{"role":"developer","content":"<dev> & co"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1"}]}],`+
		`"max_tokens":1e3,"temperature":0.70,"max_completion_tokens":5,"n":1,"response_format":{"type":"json_object"},"tools":[{"type":"function"}],"top_p":null}`, googleKey("k")))
	if err != nil {
		t.Fatal(err)
	}
	calls := fake.take()
	want := `{"contents":[{"parts":[{"text":""}],"role":"user"},{"parts":[{"text":""}],"role":"user"},{"parts":[{"text":"r"}],"role":"tool"},` +
		`{"parts":[{"text":""}],"role":"model"}],"generationConfig":{"maxOutputTokens":1000,"temperature":0.7},` +
		`"systemInstruction":{"parts":[{"text":"\u003cdev\u003e \u0026 co"}]}}`
	if len(calls) != 1 || calls[0].body != want {
		t.Fatalf("upstream = %+v", calls)
	}
	dropped := func(path string, severity translate.LossSeverity, detail string) core.Loss {
		return core.Loss{Path: path, Class: translate.LossDropped, Severity: severity, Detail: detail}
	}
	losses := []core.Loss{
		dropped("max_completion_tokens", translate.LossAdvisory, "Google does not send this Chat field"),
		dropped("messages.1.content", translate.LossMaterial, "Google receives only text content, so this content is sent as empty text"),
		dropped("messages.2.tool_call_id", translate.LossMaterial, "Google does not send this message field"),
		dropped("messages.4.tool_calls", translate.LossMaterial, "Google does not send this message field"),
		dropped("n", translate.LossAdvisory, "Google already behaves as this value asks"),
		dropped("response_format", translate.LossMaterial, "Google does not send this Chat field"),
		dropped("tools", translate.LossMaterial, "Google does not send tools, so the model cannot call them"),
	}
	if !reflect.DeepEqual(response.Losses, losses) {
		t.Fatalf("losses = %+v", response.Losses)
	}
}

func TestGoogleRefusesMalformedChatBodies(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	for _, body := range []string{`[]`, `{"messages":"hi"}`, `{"messages":[1]}`, `{"messages":[]} {}`, `null`} {
		_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", body, googleKey("k")))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorInvalidRequest || core.ClassifyError(err).Disposition() != core.DispositionTerminal {
			t.Errorf("body %s: %v, want an invalid request", body, err)
		}
	}
	request := googleChatRequest("gemini-3.5-flash", `{}`, googleKey("k"))
	request.ContentType = "text/plain"
	if _, err := provider.Invoke(context.Background(), request); err == nil {
		t.Fatal("a body that is not JSON was sent")
	}
	if calls := fake.take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
}

// Ported from the gateway's TestGoogleMapsDeveloperMessageToSystemInstruction.
func TestGoogleMapsDeveloperMessageToSystemInstruction(t *testing.T) {
	t.Parallel()
	payload := googleContentRequest([]map[string]any{
		{"role": "developer", "content": "developer policy"},
		{"role": "user", "content": "hello"},
	}, nil, nil, nil)
	system, _ := payload["systemInstruction"].(map[string]any)
	parts, _ := system["parts"].([]map[string]any)
	if len(parts) != 1 || parts[0]["text"] != "developer policy" {
		t.Fatalf("systemInstruction=%+v", system)
	}
	contents, _ := payload["contents"].([]map[string]any)
	if len(contents) != 1 || contents[0]["role"] != "user" {
		t.Fatalf("contents=%+v", contents)
	}
	if _, ok := payload["generationConfig"]; ok {
		t.Fatalf("an empty generationConfig was sent: %+v", payload)
	}
}

// Ported from the gateway's TestGoogleUsageIncludesThinkingTokens.
func TestGoogleUsageIncludesThinkingTokens(t *testing.T) {
	t.Parallel()
	input, output, total := googleUsage(map[string]any{
		"usageMetadata": map[string]any{
			"promptTokenCount": float64(7), "candidatesTokenCount": float64(3),
			"thoughtsTokenCount": float64(29), "totalTokenCount": float64(39),
		},
	})
	if input != 7 || output != 32 || total != 39 {
		t.Fatalf("usage=(%d,%d,%d)", input, output, total)
	}
	if input, output, total := googleUsage(map[string]any{"usageMetadata": map[string]any{"promptTokenCount": float64(2), "candidatesTokenCount": float64(3)}}); input != 2 || output != 3 || total != 5 {
		t.Fatalf("usage without a total=(%d,%d,%d)", input, output, total)
	}
}

// A 200 carrying an empty string is the worst failure: the caller sees
// success and no content. Gemini's thinking models reach it whenever
// max_tokens is small enough for reasoning to spend it all. Ported from the
// gateway's TestEmptyReplyExplainsItself and
// TestEmptyReplyWithoutThinkingReportsFinishReason.
func TestGoogleEmptyReplyExplainsItself(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name, answer, message string
	}{
		{
			name: "thinking", answer: `{"candidates":[{"content":{"role":"model"},"finishReason":"MAX_TOKENS"}],` +
				`"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":0,"totalTokenCount":36,"thoughtsTokenCount":29}}`,
			message: "vertex_ai: the model spent its entire output budget on reasoning (29 thinking tokens) and returned no text — raise max_tokens or omit it",
		},
		{
			name: "finish reason", answer: `{"candidates":[{"content":{"role":"model"},"finishReason":"SAFETY"}]}`,
			message: "vertex_ai: the model returned no text (finish reason SAFETY)",
		},
		{name: "blank text", answer: `{"candidates":[{"content":{"parts":[{"text":"  "}]},"finishReason":"STOP"}]}`, message: "vertex_ai: the model returned no text"},
		{name: "no candidates", answer: `{"candidates":[]}`, message: "vertex_ai: the model returned no text"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			_, base := newGoogleFake(t, googleAnswer(http.StatusOK, testCase.answer))
			provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base, Project: "p"})
			_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash",
				`{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}`, googleKey("k")))
			var cause *InvocationError
			if err == nil || err.Error() != testCase.message || !errors.As(err, &cause) || cause.Msg != testCase.message {
				t.Fatalf("error = %v, want %q", err, testCase.message)
			}
			// Another target may answer; the upstream itself is healthy.
			if classification := core.ClassifyError(err); classification != (core.ProviderErrorClassification{FailoverEligible: true}) {
				t.Fatalf("classification = %+v", classification)
			}
		})
	}
	// Image data is an answer too.
	if err := googleEmptyReplyError("ai_studio", map[string]any{"candidates": []any{map[string]any{"content": map[string]any{
		"parts": []any{map[string]any{"inlineData": map[string]any{}}},
	}}}}); err != nil {
		t.Fatalf("inline data was taken for an empty reply: %v", err)
	}
	// A finish reason is an enum value; anything else stays out of the message.
	err := googleEmptyReplyError("ai_studio", map[string]any{"candidates": []any{map[string]any{"finishReason": "see https://example.test/x"}}})
	if err == nil || err.Error() != "ai_studio: the model returned no text" {
		t.Fatalf("error = %v", err)
	}
}

func TestGoogleStreamRefusesAndSendsNothing(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	stream, err := provider.Stream(context.Background(), googleChatRequest("gemini-3.5-flash", `{"stream":true,"messages":[]}`, googleKey("k")))
	var failure *core.ProviderError
	if stream != nil || !errors.As(err, &failure) || failure.Class != core.ProviderErrorUnsupported ||
		core.ClassifyError(err).Disposition() != core.DispositionFailover || !strings.Contains(err.Error(), "streaming is not implemented") {
		t.Fatalf("stream = %v, err = %v, want a refusal that permits failover", stream, err)
	}
	messages := core.Request{Surface: core.ModelSurfaceMessages, Model: "gemini-3.5-flash", Body: []byte(`{}`), ContentType: core.ContentTypeJSON}
	var surface *core.SurfaceError
	if _, err := provider.Stream(context.Background(), messages); !errors.As(err, &surface) {
		t.Fatalf("stream err = %v, want a surface error", err)
	}
	if _, err := provider.Invoke(context.Background(), messages); !errors.As(err, &surface) {
		t.Fatalf("invoke err = %v, want a surface error", err)
	}
	if calls := fake.take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
}

// googleRecorder records the URL of every request and answers with a body
// that serves both a model action and either catalog.
type googleRecorder struct {
	mu   sync.Mutex
	urls []string
}

func (r *googleRecorder) RoundTrip(request *http.Request) (*http.Response, error) {
	r.mu.Lock()
	r.urls = append(r.urls, request.URL.String())
	r.mu.Unlock()
	return &http.Response{
		StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {core.ContentTypeJSON}}, Request: request,
		Body: io.NopCloser(strings.NewReader(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"models":[],"publisherModels":[]}`)),
	}, nil
}

// Without a base URL, inference and discovery reach Google's own hosts: a
// regional location prefixes the Vertex AI host of both, and discovery
// speaks v1beta1 where inference speaks v1.
func TestGoogleDefaultEndpoints(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		config GoogleConfig
		urls   []string
	}{
		{GoogleConfig{Deployment: GoogleAIStudio}, []string{
			"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
			"https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
		}},
		{GoogleConfig{Deployment: GoogleVertexAI, Project: "p"}, []string{
			"https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/gemini-3.5-flash:generateContent",
			"https://aiplatform.googleapis.com/v1beta1/publishers/google/models?pageSize=200",
		}},
		{GoogleConfig{Deployment: GoogleVertexAI, Project: "p", Location: "us-central1"}, []string{
			"https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-3.5-flash:generateContent",
			"https://us-central1-aiplatform.googleapis.com/v1beta1/publishers/google/models?pageSize=200",
		}},
	} {
		recorder := &googleRecorder{}
		testCase.config.Client = &http.Client{Transport: recorder}
		provider := newGoogleTest(t, testCase.config)
		if _, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleBearer("t"))); err != nil {
			t.Fatal(err)
		}
		if _, err := provider.ListModels(context.Background(), googleBearer("t")); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(recorder.urls, testCase.urls) {
			t.Fatalf("%+v reached %v, want %v", testCase.config, recorder.urls, testCase.urls)
		}
	}
}
