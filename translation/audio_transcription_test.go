package translation_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"mime/multipart"
	"slices"
	"testing"

	core "github.com/xibodev/llmgw-core"
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
