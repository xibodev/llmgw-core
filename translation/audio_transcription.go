package translation

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"mime"
	"mime/multipart"
	"path"
	"slices"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

const defaultTranscriptionInstruction = "Transcribe the supplied audio. Return only the transcription text."

// ChatTranscriptionAdapter serves the OpenAI audio_transcriptions surface by
// sending input_audio to a provider's native Chat Completions surface.
//
// Enabled is mandatory and is evaluated for every model. A provider-level
// opt-in returns true for every model; a model-level opt-in names exact rows.
// No model-name inference is performed.
type ChatTranscriptionAdapter struct {
	Provider    core.Provider
	Enabled     func(model string) bool
	Instruction string
}

var _ core.Provider = ChatTranscriptionAdapter{}

func (a ChatTranscriptionAdapter) enabled(model string) bool {
	return a.Provider != nil && a.Enabled != nil && a.Enabled(model) &&
		core.ServesNatively(a.Provider, model, core.ModelSurfaceChatCompletions)
}

func (a ChatTranscriptionAdapter) NativeSurfaces(model string) []core.ModelSurface {
	if a.Provider == nil {
		return nil
	}
	return a.Provider.NativeSurfaces(model)
}

// Surfaces reports translated transcription only for explicitly enabled
// models, after the wrapped provider's native surfaces.
func (a ChatTranscriptionAdapter) Surfaces(model string) []core.ModelSurface {
	surfaces := append([]core.ModelSurface(nil), a.NativeSurfaces(model)...)
	if a.enabled(model) && !slices.Contains(surfaces, core.ModelSurfaceAudioTranscriptions) {
		surfaces = append(surfaces, core.ModelSurfaceAudioTranscriptions)
	}
	return surfaces
}

func (a ChatTranscriptionAdapter) Unwrap() core.Provider { return a.Provider }

func (a ChatTranscriptionAdapter) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	if a.Provider == nil {
		return core.Response{}, core.NewConfigurationError("the chat transcription adapter has no provider", nil)
	}
	if core.ServesNatively(a.Provider, request.Model, request.Surface) {
		return a.Provider.Invoke(ctx, request)
	}
	if request.Surface != core.ModelSurfaceAudioTranscriptions || !a.enabled(request.Model) {
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	fileName, file, fields, err := transcriptionParts(request)
	if err != nil {
		return core.Response{}, err
	}
	format, ok := chatAudioFormat(fileName)
	if !ok {
		return core.Response{}, invalidTranscription("chat transcription accepts wav or mp3 audio", nil)
	}
	instruction := strings.TrimSpace(a.Instruction)
	if instruction == "" {
		instruction = defaultTranscriptionInstruction
	}
	if language := strings.TrimSpace(fields["language"]); language != "" {
		instruction += " The spoken language is " + language + "."
	}
	if prompt := strings.TrimSpace(fields["prompt"]); prompt != "" {
		instruction += " Context supplied by the caller: " + prompt
	}
	payload := map[string]any{
		"model": request.Model,
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": instruction},
				map[string]any{"type": "input_audio", "input_audio": map[string]any{
					"data": base64.StdEncoding.EncodeToString(file), "format": format,
				}},
			},
		}},
		"stream": false,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return core.Response{}, invalidTranscription("the chat transcription request could not be encoded", err)
	}
	upstream := request
	upstream.Surface = core.ModelSurfaceChatCompletions
	upstream.Body = body
	upstream.ContentType = core.ContentTypeJSON
	response, err := a.Provider.Invoke(ctx, upstream)
	if err != nil {
		return core.Response{}, err
	}
	text, err := chatCompletionText(response.Body)
	if err != nil {
		return core.Response{}, &core.ProviderError{
			Message: "the chat transcription response has no text", Class: core.ProviderErrorUpstream,
			Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
		}
	}
	normalized, _ := json.Marshal(map[string]any{"text": text})
	return core.Response{Body: normalized, ContentType: core.ContentTypeJSON, Losses: response.Losses}, nil
}

func (a ChatTranscriptionAdapter) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	if a.Provider != nil && core.ServesNatively(a.Provider, request.Model, request.Surface) {
		return a.Provider.Stream(ctx, request)
	}
	return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
}

// ListModels adds the translated transcription surface only to exact rows the
// caller opted in. Other catalog rows remain unchanged.
func (a ChatTranscriptionAdapter) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	if a.Provider == nil {
		return nil, core.NewConfigurationError("the chat transcription adapter has no provider", nil)
	}
	models, err := a.Provider.ListModels(ctx, credential)
	if err != nil {
		return nil, err
	}
	for index := range models {
		if !a.enabled(models[index].ID) {
			continue
		}
		if !slices.Contains(models[index].SupportedAPIs, "/v1/audio/transcriptions") {
			models[index].SupportedAPIs = append(models[index].SupportedAPIs, "/v1/audio/transcriptions")
		}
		models[index].LegacyCapabilities = maps.Clone(models[index].LegacyCapabilities)
		if models[index].LegacyCapabilities == nil {
			models[index].LegacyCapabilities = map[string]any{}
		}
		models[index].LegacyCapabilities["transcription"] = true
		models[index].LegacyCapabilities["audio_in"] = true
	}
	return models, nil
}

func transcriptionParts(request core.Request) (string, []byte, map[string]string, error) {
	mediaType, params, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != "multipart/form-data" || strings.TrimSpace(params["boundary"]) == "" {
		return "", nil, nil, invalidTranscription("an audio transcription request must be multipart form data", err)
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
			return "", nil, nil, invalidTranscription("the audio transcription multipart body is invalid", nextErr)
		}
		data, readErr := io.ReadAll(part)
		_ = part.Close()
		if readErr != nil {
			return "", nil, nil, invalidTranscription("the audio transcription multipart body could not be read", readErr)
		}
		if part.FormName() == "file" {
			file = data
			if strings.TrimSpace(part.FileName()) != "" {
				fileName = part.FileName()
			}
		} else if len(data) <= 64<<10 {
			fields[part.FormName()] = string(data)
		}
	}
	if len(file) == 0 {
		return "", nil, nil, invalidTranscription("an audio transcription request requires a nonempty file", nil)
	}
	return fileName, file, fields, nil
}

func chatAudioFormat(fileName string) (string, bool) {
	extension := strings.TrimPrefix(strings.ToLower(path.Ext(fileName)), ".")
	switch extension {
	case "wav", "mp3":
		return extension, true
	case "mpeg", "mpga":
		return "mp3", true
	}
	return "", false
}

func chatCompletionText(body []byte) (string, error) {
	var answer struct {
		Choices []struct {
			Message struct {
				Content any `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &answer); err != nil {
		return "", fmt.Errorf("decode chat completion: %w", err)
	}
	if len(answer.Choices) == 0 {
		return "", fmt.Errorf("chat completion has no choices")
	}
	switch content := answer.Choices[0].Message.Content.(type) {
	case string:
		if strings.TrimSpace(content) != "" {
			return content, nil
		}
	case []any:
		var text strings.Builder
		for _, raw := range content {
			part, _ := raw.(map[string]any)
			if value, _ := part["text"].(string); value != "" {
				text.WriteString(value)
			}
		}
		if strings.TrimSpace(text.String()) != "" {
			return text.String(), nil
		}
	}
	return "", fmt.Errorf("chat completion has no text content")
}

func invalidTranscription(message string, cause error) *core.ProviderError {
	return &core.ProviderError{Message: message, Class: core.ProviderErrorInvalidRequest, Cause: cause}
}
