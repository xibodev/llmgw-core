package translation_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	"github.com/xibodev/llmgw-core/translation"
)

func chatTranscriptionRequest(t *testing.T, fileName string, file []byte) core.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", fileName)
	if err == nil {
		_, err = part.Write(file)
	}
	if err == nil {
		err = writer.WriteField("language", "en")
	}
	if err == nil {
		err = writer.WriteField("prompt", "A product demonstration")
	}
	if closeErr := writer.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	return core.Request{
		Surface: core.ModelSurfaceAudioTranscriptions, Model: "m",
		Body: body.Bytes(), ContentType: writer.FormDataContentType(),
	}
}

func TestChatTranscriptionRequiresExplicitOptIn(t *testing.T) {
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, response: chatCompletion}
	request := chatTranscriptionRequest(t, "sample.wav", []byte("RIFFfixture"))
	_, err := (translation.ChatTranscriptionAdapter{Provider: provider}).Invoke(context.Background(), request)
	var surfaceErr *core.SurfaceError
	if !errors.As(err, &surfaceErr) || provider.calls.Load() != 0 {
		t.Fatalf("err=%v calls=%d", err, provider.calls.Load())
	}

	disabled := translation.ChatTranscriptionAdapter{Provider: provider, Enabled: func(model string) bool { return model == "another-model" }}
	_, err = disabled.Invoke(context.Background(), request)
	if !errors.As(err, &surfaceErr) || provider.calls.Load() != 0 {
		t.Fatalf("model-level opt-out err=%v calls=%d", err, provider.calls.Load())
	}
}

func TestChatTranscriptionConvertsAudioAndNormalizesResponse(t *testing.T) {
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, response: chatCompletion}
	adapter := translation.ChatTranscriptionAdapter{
		Provider: provider,
		Enabled:  func(model string) bool { return model == "m" },
	}
	file := []byte("RIFFfixture")
	response, err := adapter.Invoke(context.Background(), chatTranscriptionRequest(t, "sample.wav", file))
	if err != nil || response.ContentType != core.ContentTypeJSON || string(response.Body) != `{"text":"hello"}` {
		t.Fatalf("content-type=%q body=%s err=%v", response.ContentType, response.Body, err)
	}
	received, body := provider.last(t)
	if received.Surface != core.ModelSurfaceChatCompletions || body["stream"] != false {
		t.Fatalf("surface=%s body=%v", received.Surface, body)
	}
	messages, _ := body["messages"].([]any)
	message, _ := messages[0].(map[string]any)
	content, _ := message["content"].([]any)
	textPart, _ := content[0].(map[string]any)
	audioPart, _ := content[1].(map[string]any)
	inputAudio, _ := audioPart["input_audio"].(map[string]any)
	decoded, decodeErr := base64.StdEncoding.DecodeString(inputAudio["data"].(string))
	if decodeErr != nil || !bytes.Equal(decoded, file) || inputAudio["format"] != "wav" || textPart["type"] != "text" {
		t.Fatalf("content=%v decoded=%q err=%v", content, decoded, decodeErr)
	}
	if text := textPart["text"].(string); !bytes.Contains([]byte(text), []byte("language is en")) || !bytes.Contains([]byte(text), []byte("product demonstration")) {
		t.Fatalf("instruction=%q", text)
	}
}

func TestChatTranscriptionCatalogMarksOnlyEnabledRows(t *testing.T) {
	provider := &catalogProvider{fakeProvider: fakeProvider{native: core.ModelSurfaceChatCompletions, response: chatCompletion}}
	adapter := translation.ChatTranscriptionAdapter{Provider: provider, Enabled: func(model string) bool { return model == "audio-chat" }}
	models, err := adapter.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(models[0].SupportedAPIs, []string{"/v1/chat/completions", "/v1/audio/transcriptions"}) || models[0].LegacyCapabilities["transcription"] != true {
		t.Fatalf("enabled row=%+v", models[0])
	}
	if !slices.Equal(models[1].SupportedAPIs, []string{"/v1/chat/completions"}) || models[1].LegacyCapabilities != nil {
		t.Fatalf("disabled row=%+v", models[1])
	}
}

type catalogProvider struct{ fakeProvider }

func (p *catalogProvider) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

func (p *catalogProvider) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	return []core.ModelInfo{
		{ID: "audio-chat", SupportedAPIs: []string{"/v1/chat/completions"}},
		{ID: "ordinary-chat", SupportedAPIs: []string{"/v1/chat/completions"}},
	}, nil
}

func TestChatTranscriptionMalformedProviderAnswerFailsOver(t *testing.T) {
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, response: `{"choices":[{"message":{"content":""}}]}`}
	adapter := translation.ChatTranscriptionAdapter{Provider: provider, Enabled: func(string) bool { return true }}
	_, err := adapter.Invoke(context.Background(), chatTranscriptionRequest(t, "sample.mp3", []byte("ID3fixture")))
	if err == nil || core.ClassifyError(err).Disposition() != core.DispositionFailover {
		t.Fatalf("err=%v classification=%+v", err, core.ClassifyError(err))
	}
}

func TestChatTranscriptionResponseShapeIsOpenAICompatible(t *testing.T) {
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, response: chatCompletion}
	adapter := translation.ChatTranscriptionAdapter{Provider: provider, Enabled: func(string) bool { return true }}
	response, err := adapter.Invoke(context.Background(), chatTranscriptionRequest(t, "sample.mp3", []byte("ID3fixture")))
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if json.Unmarshal(response.Body, &body) != nil || body["text"] != "hello" || len(body) != 1 {
		t.Fatalf("response=%s", response.Body)
	}
}

// providers.OpenAICompatible lists audio_transcriptions natively for every
// model. The opt-in still wins for the models it names: they transcribe
// through Chat, and their transcription reads as translated. Every other
// model's upload reaches the audio endpoint unchanged.
func TestChatTranscriptionOptInWinsOverANativeTranscription(t *testing.T) {
	t.Parallel()
	var mu sync.Mutex
	var calls []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		calls = append(calls, r.URL.Path+" "+string(body))
		mu.Unlock()
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		if r.URL.Path == "/v1/chat/completions" {
			_, _ = io.WriteString(w, chatCompletion)
			return
		}
		_, _ = io.WriteString(w, `{"text":"native"}`)
	}))
	defer server.Close()
	provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{BaseURL: server.URL + "/v1", Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	adapter := translation.ChatTranscriptionAdapter{Provider: provider, Enabled: func(model string) bool { return model == "gpt-audio" }}
	transcriptions := core.ModelSurfaceAudioTranscriptions
	if slices.Contains(adapter.NativeSurfaces("gpt-audio"), transcriptions) || !slices.Contains(adapter.Surfaces("gpt-audio"), transcriptions) ||
		!slices.Contains(adapter.NativeSurfaces("gpt-audio"), core.ModelSurfaceAudioSpeech) ||
		!slices.Contains(adapter.NativeSurfaces("whisper-1"), transcriptions) || !slices.Contains(provider.NativeSurfaces("gpt-audio"), transcriptions) {
		t.Fatalf("gpt-audio native=%v surfaces=%v, whisper-1 native=%v", adapter.NativeSurfaces("gpt-audio"), adapter.Surfaces("gpt-audio"), adapter.NativeSurfaces("whisper-1"))
	}
	if core.PreservesWire(adapter, "gpt-audio", transcriptions) || !core.PreservesWire(adapter, "whisper-1", transcriptions) {
		t.Fatal("the transcription through Chat reads as preserved, or the native one does not")
	}

	bridged := chatTranscriptionRequest(t, "sample.wav", []byte("RIFFfixture"))
	bridged.Model = "gpt-audio"
	response, err := adapter.Invoke(context.Background(), bridged)
	if err != nil || string(response.Body) != `{"text":"hello"}` {
		t.Fatalf("through Chat: body=%s err=%v", response.Body, err)
	}
	native := chatTranscriptionRequest(t, "sample.wav", []byte("RIFFfixture"))
	native.Model = "whisper-1"
	response, err = adapter.Invoke(context.Background(), native)
	if err != nil || string(response.Body) != `{"text":"native"}` {
		t.Fatalf("native: body=%s err=%v", response.Body, err)
	}
	mu.Lock()
	sent := slices.Clone(calls)
	mu.Unlock()
	if len(sent) != 2 || !strings.HasPrefix(sent[0], "/v1/chat/completions ") || !strings.Contains(sent[0], `"input_audio"`) ||
		!strings.Contains(sent[0], `"model":"gpt-audio"`) || sent[1] != "/v1/audio/transcriptions "+string(native.Body) {
		t.Fatalf("upstream = %q", sent)
	}

	// Transcription through Chat does not stream; a native one is the
	// provider's to refuse.
	var surface *core.SurfaceError
	if _, err := adapter.Stream(context.Background(), bridged); !errors.As(err, &surface) {
		t.Fatalf("stream through Chat: err = %v", err)
	}
	var failure *core.ProviderError
	if _, err := adapter.Stream(context.Background(), native); !errors.As(err, &failure) || failure.Class != core.ProviderErrorUnsupported {
		t.Fatalf("native stream: err = %v", err)
	}
}
