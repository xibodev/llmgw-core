package providers

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

type audioWireFixture struct {
	Request struct {
		Method       string `json:"method"`
		Path         string `json:"path"`
		AuthHeader   string `json:"auth_header"`
		ModelID      string `json:"model_id"`
		Model        string `json:"model"`
		LanguageCode string `json:"language_code"`
		FileName     string `json:"file_name"`
		FileBase64   string `json:"file_base64"`
		Voice        string `json:"voice"`
		Input        string `json:"input"`
		InputRole    string `json:"input_role"`
		Format       string `json:"format"`
		OutputFormat string `json:"output_format"`
	} `json:"request"`
	Response struct {
		Status      int             `json:"status"`
		ContentType string          `json:"content_type"`
		RetryAfter  string          `json:"retry_after"`
		Body        json.RawMessage `json:"body"`
		BodyBase64  string          `json:"body_base64"`
	} `json:"response"`
}

func loadAudioFixture(t *testing.T, provider, name string) audioWireFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", provider, name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture audioWireFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func writeAudioFixture(t *testing.T, w http.ResponseWriter, fixture audioWireFixture) {
	t.Helper()
	if fixture.Response.ContentType != "" {
		w.Header().Set("Content-Type", fixture.Response.ContentType)
	}
	if fixture.Response.RetryAfter != "" {
		w.Header().Set("Retry-After", fixture.Response.RetryAfter)
	}
	w.WriteHeader(fixture.Response.Status)
	if fixture.Response.BodyBase64 != "" {
		body, err := base64.StdEncoding.DecodeString(fixture.Response.BodyBase64)
		if err != nil {
			t.Error(err)
		}
		_, _ = w.Write(body)
		return
	}
	_, _ = w.Write(fixture.Response.Body)
}

func transcriptionRequest(t *testing.T, model, fileName string, file []byte, fields map[string]string) core.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", fileName)
	if err == nil {
		_, err = part.Write(file)
	}
	for name, value := range fields {
		if err == nil {
			err = writer.WriteField(name, value)
		}
	}
	if closeErr := writer.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	return core.Request{
		Surface: core.ModelSurfaceAudioTranscriptions, Model: model,
		Body: body.Bytes(), ContentType: writer.FormDataContentType(),
		Credential: &core.Credential{APIKey: "fixture-key"},
	}
}

func speechRequest(model, body string) core.Request {
	return core.Request{
		Surface: core.ModelSurfaceAudioSpeech, Model: model, Body: []byte(body),
		ContentType: core.ContentTypeJSON, Credential: &core.Credential{APIKey: "fixture-key"},
	}
}

func TestElevenLabsTranscriptionWireFixture(t *testing.T) {
	fixture := loadAudioFixture(t, "elevenlabs", "transcription-success")
	file, _ := base64.StdEncoding.DecodeString(fixture.Request.FileBase64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != fixture.Request.Method || r.URL.Path != fixture.Request.Path || r.Header.Get(fixture.Request.AuthHeader) != "fixture-key" {
			t.Errorf("request method=%s path=%s auth=%q", r.Method, r.URL.Path, r.Header.Get(fixture.Request.AuthHeader))
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("parse multipart: %v", err)
		}
		if r.FormValue("model_id") != fixture.Request.ModelID || r.FormValue("language_code") != fixture.Request.LanguageCode {
			t.Errorf("form=%v", r.MultipartForm.Value)
		}
		part, header, err := r.FormFile("file")
		if err != nil {
			t.Errorf("file: %v", err)
		} else {
			defer part.Close()
			var received bytes.Buffer
			_, _ = received.ReadFrom(part)
			if header.Filename != fixture.Request.FileName || !bytes.Equal(received.Bytes(), file) {
				t.Errorf("file name=%q body=%q", header.Filename, received.Bytes())
			}
		}
		writeAudioFixture(t, w, fixture)
	}))
	defer server.Close()

	provider, err := NewElevenLabs(ElevenLabsConfig{BaseURL: server.URL, Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	response, err := provider.Invoke(context.Background(), transcriptionRequest(t, fixture.Request.ModelID, fixture.Request.FileName, file, map[string]string{"language": "en"}))
	if err != nil || response.ContentType != core.ContentTypeJSON || string(response.Body) != `{"text":"Hello world!"}` {
		t.Fatalf("content-type=%q body=%s err=%v", response.ContentType, response.Body, err)
	}
}

func TestElevenLabsSpeechWireFixture(t *testing.T) {
	fixture := loadAudioFixture(t, "elevenlabs", "speech-success")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != fixture.Request.Method || r.URL.Path != fixture.Request.Path || r.Header.Get(fixture.Request.AuthHeader) != "fixture-key" {
			t.Errorf("request method=%s path=%s auth=%q", r.Method, r.URL.Path, r.Header.Get(fixture.Request.AuthHeader))
		}
		if r.URL.Query().Get("output_format") != fixture.Request.OutputFormat {
			t.Errorf("output_format=%q", r.URL.Query().Get("output_format"))
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body["model_id"] != fixture.Request.ModelID || body["text"] != fixture.Request.Input {
			t.Errorf("body=%v err=%v", body, err)
		}
		writeAudioFixture(t, w, fixture)
	}))
	defer server.Close()

	provider, _ := NewElevenLabs(ElevenLabsConfig{BaseURL: server.URL, Client: server.Client()})
	requestBody, _ := json.Marshal(map[string]any{
		"input": fixture.Request.Input, "voice": fixture.Request.Voice, "response_format": "mp3",
	})
	response, err := provider.Invoke(context.Background(), speechRequest(fixture.Request.ModelID, string(requestBody)))
	want, _ := base64.StdEncoding.DecodeString(fixture.Response.BodyBase64)
	if err != nil || response.ContentType != fixture.Response.ContentType || !bytes.Equal(response.Body, want) {
		t.Fatalf("content-type=%q body=%q err=%v", response.ContentType, response.Body, err)
	}
}

func TestElevenLabsErrorFixtureClassifies429(t *testing.T) {
	fixture := loadAudioFixture(t, "elevenlabs", "error-429")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { writeAudioFixture(t, w, fixture) }))
	defer server.Close()
	provider, _ := NewElevenLabs(ElevenLabsConfig{BaseURL: server.URL, Client: server.Client()})
	_, err := provider.Invoke(context.Background(), speechRequest("eleven_flash_v2_5", `{"input":"hello","voice":"voice-fixture"}`))
	if err == nil || core.ClassifyError(err).Disposition() != core.DispositionRetryable || strings.Contains(err.Error(), "fixture quota") {
		t.Fatalf("err=%v classification=%+v", err, core.ClassifyError(err))
	}
}

func TestMiMoSpeechWireFixture(t *testing.T) {
	fixture := loadAudioFixture(t, "mimo", "speech-success")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != fixture.Request.Method || r.URL.Path != fixture.Request.Path || r.Header.Get(fixture.Request.AuthHeader) != "fixture-key" || r.Header.Get("Authorization") != "" {
			t.Errorf("request method=%s path=%s api-key=%q authorization=%q", r.Method, r.URL.Path, r.Header.Get(fixture.Request.AuthHeader), r.Header.Get("Authorization"))
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		audio, _ := body["audio"].(map[string]any)
		messages, _ := body["messages"].([]any)
		message, _ := messages[len(messages)-1].(map[string]any)
		if body["model"] != fixture.Request.Model || audio["voice"] != fixture.Request.Voice || audio["format"] != fixture.Request.Format || message["role"] != fixture.Request.InputRole || message["content"] != fixture.Request.Input {
			t.Errorf("body=%v", body)
		}
		writeAudioFixture(t, w, fixture)
	}))
	defer server.Close()

	provider, _ := NewMiMo(MiMoConfig{BaseURL: server.URL, Client: server.Client()})
	requestBody, _ := json.Marshal(map[string]any{
		"input": fixture.Request.Input, "voice": fixture.Request.Voice, "response_format": fixture.Request.Format,
	})
	response, err := provider.Invoke(context.Background(), speechRequest(fixture.Request.Model, string(requestBody)))
	want, _ := base64.StdEncoding.DecodeString("UklGRmZpeHR1cmUtd2F2")
	if err != nil || response.ContentType != "audio/wav" || !bytes.Equal(response.Body, want) {
		t.Fatalf("content-type=%q body=%q err=%v", response.ContentType, response.Body, err)
	}
}

func TestMiMoErrorFixtureClassifies401AsTerminal(t *testing.T) {
	fixture := loadAudioFixture(t, "mimo", "error-401")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { writeAudioFixture(t, w, fixture) }))
	defer server.Close()
	provider, _ := NewMiMo(MiMoConfig{BaseURL: server.URL, Client: server.Client()})
	_, err := provider.Invoke(context.Background(), speechRequest("mimo-v2.5-tts", `{"input":"hello","voice":"Mia"}`))
	if err == nil || core.ClassifyError(err).Disposition() != core.DispositionTerminal || strings.Contains(err.Error(), "fixture credential") {
		t.Fatalf("err=%v classification=%+v", err, core.ClassifyError(err))
	}
}

func TestAudioProviderCatalogsDeclareOnlyTheirExactSurfaces(t *testing.T) {
	eleven, _ := NewElevenLabs(ElevenLabsConfig{})
	elevenModels, _ := eleven.ListModels(context.Background(), nil)
	for _, model := range elevenModels {
		want := "/v1/audio/speech"
		if model.ID == "scribe_v2" {
			want = "/v1/audio/transcriptions"
		}
		if !slices.Equal(model.SupportedAPIs, []string{want}) {
			t.Fatalf("ElevenLabs model %q surfaces=%v", model.ID, model.SupportedAPIs)
		}
	}
	mimo, _ := NewMiMo(MiMoConfig{})
	mimoModels, _ := mimo.ListModels(context.Background(), nil)
	for _, model := range mimoModels {
		if !slices.Equal(model.SupportedAPIs, []string{"/v1/audio/speech"}) {
			t.Fatalf("MiMo model %q surfaces=%v", model.ID, model.SupportedAPIs)
		}
	}
}
