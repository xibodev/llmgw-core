package runtime_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
	"github.com/xibodev/llmgw-core/translation"
)

// A Runtime over a translation.Adapter hands the OpenAI audio surfaces to
// OpenAICompatible natively, with the instance's credential: the upload
// reaches the upstream unchanged, speech with the request's model, and
// both answers come back as sent and labelled native. The Adapter's routes
// stay the Chat surfaces'. A stream is refused before anything is sent,
// and the refusal leaves the instance's health alone.
func TestRuntimeServesTheOpenAIAudioSurfacesNatively(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	var mu sync.Mutex
	var calls []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		calls = append(calls, r.URL.Path+" "+r.Header.Get("Authorization")+" "+r.Header.Get("Content-Type")+" "+string(body))
		mu.Unlock()
		switch r.URL.Path {
		case "/v1/audio/transcriptions":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"text":"Hello"}`)
		case "/v1/audio/speech":
			w.Header().Set("Content-Type", "audio/mpeg")
			_, _ = io.WriteString(w, "ID3fixture")
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(ctx, "fixture-upstream", core.APIKeyRecord("fixture-upstream-token")); err != nil {
		t.Fatal(err)
	}
	owner := core.Caller{ID: "owner", Kind: core.CallerHuman}
	store.Bind(owner, "voice", "fixture-upstream")
	var adapter translation.Adapter
	runtime, err := coreruntime.New(coreruntime.Options[struct{}]{
		Settings: coreruntime.NewMemorySettings(struct{}{}),
		Providers: func(struct{}, string) (core.Provider, error) {
			provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{BaseURL: server.URL + "/v1", Client: server.Client()})
			adapter = translation.Adapter{Provider: provider}
			return adapter, err
		},
		Credentials: store,
	})
	if err != nil {
		t.Fatal(err)
	}

	var upload bytes.Buffer
	writer := multipart.NewWriter(&upload)
	part, _ := writer.CreateFormFile("file", "sample.wav")
	_, _ = part.Write([]byte("RIFF\x00\xff fixture"))
	_ = writer.WriteField("model", "whisper-1")
	_ = writer.Close()
	transcription := core.Request{Surface: core.ModelSurfaceAudioTranscriptions, Model: "whisper-1", Body: upload.Bytes(), ContentType: writer.FormDataContentType()}
	response, err := runtime.Invoke(ctx, owner, "voice", transcription)
	if err != nil || string(response.Body) != `{"text":"Hello"}` || response.ContentType != "application/json" {
		t.Fatalf("transcription = %q %s, err = %v", response.ContentType, response.Body, err)
	}
	speech := gatewayRequest(core.ModelSurfaceAudioSpeech, "tts-1", map[string]any{"voice": "alloy", "input": "Hello"})
	response, err = runtime.Invoke(ctx, owner, "voice", speech)
	if err != nil || string(response.Body) != "ID3fixture" || response.ContentType != "audio/mpeg" {
		t.Fatalf("speech = %q %q, err = %v", response.ContentType, response.Body, err)
	}
	mu.Lock()
	sent := slices.Clone(calls)
	mu.Unlock()
	if want := []string{
		"/v1/audio/transcriptions Bearer fixture-upstream-token " + writer.FormDataContentType() + " " + upload.String(),
		`/v1/audio/speech Bearer fixture-upstream-token application/json {"input":"Hello","model":"tts-1","voice":"alloy"}`,
	}; !slices.Equal(sent, want) {
		t.Fatalf("upstream = %q, want %q", sent, want)
	}
	for _, surface := range []core.ModelSurface{core.ModelSurfaceAudioTranscriptions, core.ModelSurfaceAudioSpeech} {
		if mode := runtime.TransportMode(ctx, owner, "voice", "whisper-1", surface); mode != core.TransportModeNative {
			t.Fatalf("%s transport mode = %s", surface, mode)
		}
	}
	if surfaces := adapter.Surfaces("whisper-1"); !slices.Equal(surfaces, []core.ModelSurface{
		core.ModelSurfaceChatCompletions, core.ModelSurfaceAudioTranscriptions, core.ModelSurfaceAudioSpeech,
		core.ModelSurfaceMessages, core.ModelSurfaceResponses,
	}) {
		t.Fatalf("surfaces = %v", surfaces)
	}

	var failure *core.ProviderError
	if stream, err := runtime.Stream(ctx, owner, "voice", speech); stream != nil || !errors.As(err, &failure) || failure.Class != core.ProviderErrorUnsupported {
		t.Fatalf("stream = %v, err = %v", stream, err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(calls) != 2 || runtime.Health("voice").Status != core.ProviderHealthHealthy {
		t.Fatalf("a refused stream reached the upstream, or changed health: %q %+v", calls, runtime.Health("voice"))
	}
}
