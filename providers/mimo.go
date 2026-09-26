package providers

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

const mimoDefaultBaseURL = "https://api.xiaomimimo.com/v1"

func mimoDefaultSpeechModels() []string {
	return []string{"mimo-v2.5-tts", "mimo-v2.5-tts-voicedesign", "mimo-v2.5-tts-voiceclone"}
}

// MiMoConfig configures Xiaomi MiMo's chat-completions speech bridge. A nil
// SpeechModels list selects the reviewed roster; an empty list disables it.
type MiMoConfig struct {
	BaseURL      string
	Client       *http.Client
	SpeechModels []string
}

// MiMo serves OpenAI audio_speech by translating it to MiMo's documented
// Chat Completions audio request and decoding the returned base64 audio.
type MiMo struct {
	baseURL      string
	client       *http.Client
	speechModels []string
}

var _ core.Provider = (*MiMo)(nil)

func NewMiMo(config MiMoConfig) (*MiMo, error) {
	base := strings.TrimSpace(config.BaseURL)
	if base == "" {
		base = mimoDefaultBaseURL
	}
	parsed, err := url.Parse(base)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, core.NewConfigurationError("the MiMo base URL must be an absolute URL", err)
	}
	client := config.Client
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Minute}
	}
	models := config.SpeechModels
	if models == nil {
		models = mimoDefaultSpeechModels()
	}
	return &MiMo{
		baseURL: strings.TrimRight(base, "/"), client: client,
		speechModels: normalizedModelIDs(models),
	}, nil
}

func (p *MiMo) NativeSurfaces(model string) []core.ModelSurface {
	if slices.Contains(p.speechModels, model) {
		return []core.ModelSurface{core.ModelSurfaceAudioSpeech}
	}
	return nil
}

func (p *MiMo) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	if request.Surface != core.ModelSurfaceAudioSpeech || !core.ServesNatively(p, request.Model, request.Surface) {
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
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
	if strings.TrimSpace(input) == "" {
		return core.Response{}, openAIInvalid("MiMo speech requires input", nil)
	}
	format, _ := payload["response_format"].(string)
	if format == "" {
		format = "wav"
	}
	format = strings.ToLower(strings.TrimSpace(format))
	if format != "wav" && format != "pcm16" {
		return core.Response{}, openAIInvalid("MiMo speech supports response_format wav or pcm16", nil)
	}

	messages := make([]any, 0, 2)
	if instruction, _ := payload["instructions"].(string); strings.TrimSpace(instruction) != "" {
		messages = append(messages, map[string]any{"role": "user", "content": instruction})
	}
	messages = append(messages, map[string]any{"role": "assistant", "content": input})
	audio := map[string]any{"format": format}
	if strings.TrimSpace(voice) != "" {
		audio["voice"] = voice
	}
	upstreamPayload := map[string]any{
		"model": request.Model, "messages": messages, "audio": audio, "stream": false,
	}
	encoded, err := openAIEncode(upstreamPayload)
	if err != nil {
		return core.Response{}, openAIInvalid("the MiMo speech request could not be encoded", err)
	}
	upstream, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(encoded))
	if err != nil {
		return core.Response{}, core.NewConfigurationError("the MiMo speech request could not be created", err)
	}
	upstream.Header.Set("Content-Type", core.ContentTypeJSON)
	upstream.Header.Set("Api-Key", key)
	response, err := p.client.Do(upstream)
	if err != nil {
		return core.Response{}, transportFailure(ctx, "MiMo could not be reached", err)
	}
	defer response.Body.Close()
	raw, err := readInvocationResponseBody(ctx, response, "MiMo")
	if err != nil {
		return core.Response{}, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		return core.Response{}, httpStatusFailure("MiMo", response, raw, time.Now())
	}
	var answer struct {
		Choices []struct {
			Message struct {
				Audio *struct {
					Data   string `json:"data"`
					Format string `json:"format"`
				} `json:"audio"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(raw, &answer) != nil || len(answer.Choices) == 0 || answer.Choices[0].Message.Audio == nil {
		return core.Response{}, unusableResponse("the MiMo speech response has no audio", nil)
	}
	audioAnswer := answer.Choices[0].Message.Audio
	decoded, err := base64.StdEncoding.DecodeString(audioAnswer.Data)
	if err != nil || len(decoded) == 0 {
		return core.Response{}, unusableResponse("the MiMo speech response has invalid audio", err)
	}
	answerFormat := strings.ToLower(strings.TrimSpace(audioAnswer.Format))
	if answerFormat == "" {
		answerFormat = format
	}
	contentType := "audio/wav"
	if answerFormat == "pcm16" || answerFormat == "pcm" {
		contentType = "audio/L16"
	}
	return core.Response{Body: decoded, ContentType: contentType}, nil
}

func (p *MiMo) Stream(_ context.Context, request core.Request) (core.StreamIter, error) {
	return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
}

func (p *MiMo) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	models := make([]core.ModelInfo, 0, len(p.speechModels))
	for _, model := range p.speechModels {
		models = append(models, audioModelInfo(model, "Xiaomi MiMo", core.ModelSurfaceAudioSpeech))
	}
	return models, nil
}
