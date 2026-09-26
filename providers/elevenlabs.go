package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"path"
	"slices"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

const elevenLabsDefaultBaseURL = "https://api.elevenlabs.io/v1"

func elevenLabsDefaultTranscriptionModels() []string { return []string{"scribe_v2"} }

func elevenLabsDefaultSpeechModels() []string {
	return []string{"eleven_v3", "eleven_multilingual_v2", "eleven_flash_v2_5"}
}

// ElevenLabsConfig configures the ElevenLabs speech provider. Nil model lists
// select the reviewed defaults; an explicit empty list disables that surface.
type ElevenLabsConfig struct {
	BaseURL             string
	Client              *http.Client
	TranscriptionModels []string
	SpeechModels        []string
}

// ElevenLabs serves the OpenAI audio surfaces over ElevenLabs' native speech
// APIs. It has no chat surface.
type ElevenLabs struct {
	baseURL             string
	client              *http.Client
	transcriptionModels []string
	speechModels        []string
}

var _ core.Provider = (*ElevenLabs)(nil)

// NewElevenLabs returns an ElevenLabs provider with explicit audio capability
// lists. Capability is never inferred from an arbitrary model name.
func NewElevenLabs(config ElevenLabsConfig) (*ElevenLabs, error) {
	base := strings.TrimSpace(config.BaseURL)
	if base == "" {
		base = elevenLabsDefaultBaseURL
	}
	parsed, err := url.Parse(base)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, core.NewConfigurationError("the ElevenLabs base URL must be an absolute URL", err)
	}
	client := config.Client
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Minute}
	}
	transcriptionModels := config.TranscriptionModels
	if transcriptionModels == nil {
		transcriptionModels = elevenLabsDefaultTranscriptionModels()
	}
	speechModels := config.SpeechModels
	if speechModels == nil {
		speechModels = elevenLabsDefaultSpeechModels()
	}
	return &ElevenLabs{
		baseURL:             strings.TrimRight(base, "/"),
		client:              client,
		transcriptionModels: normalizedModelIDs(transcriptionModels),
		speechModels:        normalizedModelIDs(speechModels),
	}, nil
}

func normalizedModelIDs(values []string) []string {
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" && !slices.Contains(result, value) {
			result = append(result, value)
		}
	}
	return result
}

// NativeSurfaces reports only the explicitly configured model capabilities.
func (p *ElevenLabs) NativeSurfaces(model string) []core.ModelSurface {
	var surfaces []core.ModelSurface
	if slices.Contains(p.transcriptionModels, model) {
		surfaces = append(surfaces, core.ModelSurfaceAudioTranscriptions)
	}
	if slices.Contains(p.speechModels, model) {
		surfaces = append(surfaces, core.ModelSurfaceAudioSpeech)
	}
	return surfaces
}

// Invoke serves speech-to-text and text-to-speech and normalizes successful
// transcription answers to the OpenAI {"text": ...} response shape.
func (p *ElevenLabs) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	if !core.ServesNatively(p, request.Model, request.Surface) {
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	switch request.Surface {
	case core.ModelSurfaceAudioTranscriptions:
		return p.transcribe(ctx, request)
	case core.ModelSurfaceAudioSpeech:
		return p.speak(ctx, request)
	default:
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
}

func (p *ElevenLabs) Stream(_ context.Context, request core.Request) (core.StreamIter, error) {
	return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
}

// ListModels returns the reviewed model roster with explicit audio surfaces.
func (p *ElevenLabs) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	models := make([]core.ModelInfo, 0, len(p.transcriptionModels)+len(p.speechModels))
	for _, model := range p.transcriptionModels {
		models = append(models, audioModelInfo(model, "ElevenLabs", core.ModelSurfaceAudioTranscriptions))
	}
	for _, model := range p.speechModels {
		models = append(models, audioModelInfo(model, "ElevenLabs", core.ModelSurfaceAudioSpeech))
	}
	return models, nil
}

func audioModelInfo(id, owner string, surface core.ModelSurface) core.ModelInfo {
	path := "/v1/audio/speech"
	legacy := map[string]any{"tts": true, "audio_out": true}
	if surface == core.ModelSurfaceAudioTranscriptions {
		path = "/v1/audio/transcriptions"
		legacy = map[string]any{"transcription": true, "audio_in": true}
	}
	return core.ModelInfo{
		ID: id, Object: "model", OwnedBy: owner, Vendor: owner,
		SupportedAPIs: []string{path}, LegacyCapabilities: legacy,
	}
}

func audioCredentialKey(credential *core.Credential) (string, error) {
	if credential == nil {
		return "", core.NewConfigurationError("the audio provider requires an API key", nil)
	}
	if credential.TokenType != "" && credential.TokenType != core.TokenTypeAPIKey {
		return "", core.NewConfigurationError("the audio provider requires an API-key credential", nil)
	}
	key := strings.TrimSpace(credential.APIKey)
	if key == "" {
		key = strings.TrimSpace(credential.Token)
	}
	if key == "" {
		return "", core.NewConfigurationError("the audio provider requires an API key", nil)
	}
	return key, nil
}

func (p *ElevenLabs) transcribe(ctx context.Context, request core.Request) (core.Response, error) {
	key, err := audioCredentialKey(request.Credential)
	if err != nil {
		return core.Response{}, err
	}
	fileName, file, fields, err := openAITranscriptionParts(request)
	if err != nil {
		return core.Response{}, err
	}
	if format := strings.TrimSpace(fields["response_format"]); format != "" && format != "json" && format != "verbose_json" {
		return core.Response{}, openAIInvalid("ElevenLabs transcription supports the JSON response format", nil)
	}

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", fileName)
	if err == nil {
		_, err = part.Write(file)
	}
	if err == nil {
		err = writer.WriteField("model_id", request.Model)
	}
	if err == nil && strings.TrimSpace(fields["language"]) != "" {
		err = writer.WriteField("language_code", strings.TrimSpace(fields["language"]))
	}
	if closeErr := writer.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return core.Response{}, openAIInvalid("the ElevenLabs transcription request could not be encoded", err)
	}

	upstream, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/speech-to-text", &body)
	if err != nil {
		return core.Response{}, core.NewConfigurationError("the ElevenLabs transcription request could not be created", err)
	}
	upstream.Header.Set("Content-Type", writer.FormDataContentType())
	upstream.Header.Set("xi-api-key", key)
	response, err := p.client.Do(upstream)
	if err != nil {
		return core.Response{}, transportFailure(ctx, "ElevenLabs could not be reached", err)
	}
	defer response.Body.Close()
	raw, err := readInvocationResponseBody(ctx, response, "ElevenLabs")
	if err != nil {
		return core.Response{}, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		return core.Response{}, httpStatusFailure("ElevenLabs", response, raw, time.Now())
	}
	var answer struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &answer) != nil || strings.TrimSpace(answer.Text) == "" {
		return core.Response{}, unusableResponse("the ElevenLabs transcription has no text", nil)
	}
	normalized, _ := json.Marshal(map[string]any{"text": answer.Text})
	return core.Response{Body: normalized, ContentType: core.ContentTypeJSON}, nil
}

func (p *ElevenLabs) speak(ctx context.Context, request core.Request) (core.Response, error) {
	key, err := audioCredentialKey(request.Credential)
	if err != nil {
		return core.Response{}, err
	}
	payload, err := openAIPayload(request)
	if err != nil {
		return core.Response{}, err
	}
	input, _ := payload["input"].(string)
	voice, _ := payload["voice"].(string)
	if strings.TrimSpace(input) == "" || strings.TrimSpace(voice) == "" {
		return core.Response{}, openAIInvalid("ElevenLabs speech requires input and voice", nil)
	}
	format, _ := payload["response_format"].(string)
	if format != "" && !strings.EqualFold(format, "mp3") {
		return core.Response{}, openAIInvalid("ElevenLabs speech currently supports response_format mp3", nil)
	}
	body := map[string]any{"text": input, "model_id": request.Model}
	if speed, ok := payload["speed"].(float64); ok {
		body["voice_settings"] = map[string]any{"speed": speed}
	}
	encoded, err := openAIEncode(body)
	if err != nil {
		return core.Response{}, openAIInvalid("the ElevenLabs speech request could not be encoded", err)
	}
	endpoint := p.baseURL + path.Join("/text-to-speech/", url.PathEscape(strings.TrimSpace(voice))) + "?output_format=mp3_44100_128"
	upstream, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(encoded))
	if err != nil {
		return core.Response{}, core.NewConfigurationError("the ElevenLabs speech request could not be created", err)
	}
	upstream.Header.Set("Content-Type", core.ContentTypeJSON)
	upstream.Header.Set("Accept", "audio/mpeg")
	upstream.Header.Set("xi-api-key", key)
	response, err := p.client.Do(upstream)
	if err != nil {
		return core.Response{}, transportFailure(ctx, "ElevenLabs could not be reached", err)
	}
	defer response.Body.Close()
	raw, err := readInvocationResponseBody(ctx, response, "ElevenLabs")
	if err != nil {
		return core.Response{}, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		return core.Response{}, httpStatusFailure("ElevenLabs", response, raw, time.Now())
	}
	contentType := response.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "audio/mpeg"
	}
	return core.Response{Body: raw, ContentType: contentType}, nil
}

func openAITranscriptionParts(request core.Request) (string, []byte, map[string]string, error) {
	mediaType, params, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != "multipart/form-data" || strings.TrimSpace(params["boundary"]) == "" {
		return "", nil, nil, openAIInvalid("an audio transcription request must be multipart form data", err)
	}
	reader := multipart.NewReader(bytes.NewReader(request.Body), params["boundary"])
	fields := map[string]string{}
	var file []byte
	fileName := "audio"
	for {
		part, nextErr := reader.NextPart()
		if nextErr == io.EOF {
			break
		}
		if nextErr != nil {
			return "", nil, nil, openAIInvalid("the audio transcription multipart body is invalid", nextErr)
		}
		data, readErr := io.ReadAll(part)
		_ = part.Close()
		if readErr != nil {
			return "", nil, nil, openAIInvalid("the audio transcription multipart body could not be read", readErr)
		}
		if part.FormName() == "file" {
			file = data
			if strings.TrimSpace(part.FileName()) != "" {
				fileName = part.FileName()
			}
			continue
		}
		if len(data) <= 64<<10 {
			fields[part.FormName()] = string(data)
		}
	}
	if len(file) == 0 {
		return "", nil, nil, openAIInvalid("an audio transcription request requires a nonempty file", nil)
	}
	return fileName, file, fields, nil
}
