package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"mime"
	"slices"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// edgeTTSSpeech is what Edge TTS speaks of an OpenAI speech request.
type edgeTTSSpeech struct{ input, voice, rate string }

// speechRequest reads an OpenAI speech body as the gateway reads one for a
// native synthesizer: numbers as float64, input as a string, and a speed
// that is not a number as normal. The voice is the model's, "default"
// naming the default voice. A body voice that names another voice, and
// every field Edge TTS does not read, are reported as losses: a
// stream_format other than audio materially, because the answer is not a
// stream, and any other advisorily.
func (p *EdgeTTS) speechRequest(request core.Request) (edgeTTSSpeech, []core.Loss, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return edgeTTSSpeech{}, nil, &core.ProviderError{Message: "an Edge TTS request body must be JSON", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	decoder := json.NewDecoder(bytes.NewReader(request.Body))
	var body map[string]any
	err = decoder.Decode(&body)
	if err == nil {
		if _, trailing := decoder.Token(); trailing != io.EOF {
			err = errors.New("the body continues after its JSON object")
		}
	}
	if err != nil || body == nil {
		return edgeTTSSpeech{}, nil, &core.ProviderError{Message: "the Edge TTS request body is not a JSON object", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	speech := edgeTTSSpeech{voice: p.speaks(request.Model)}
	speech.input, _ = body["input"].(string)
	if strings.TrimSpace(speech.input) == "" {
		return edgeTTSSpeech{}, nil, &core.ProviderError{Message: "an Edge TTS speech request needs its input", Class: core.ProviderErrorInvalidRequest}
	}
	if format, _ := body["response_format"].(string); format != "" && !strings.EqualFold(format, "mp3") {
		return edgeTTSSpeech{}, nil, &core.ProviderError{
			Message: "Edge TTS produces mp3 audio; ask for response_format mp3 or omit it", Class: core.ProviderErrorUnsupported,
			Classification: core.ProviderErrorClassification{FailoverEligible: true},
		}
	}
	speed := 1.0
	if value, ok := body["speed"].(float64); ok {
		speed = value
	}
	var ok bool
	if speech.rate, ok = edgeTTSRate(speed); !ok {
		return edgeTTSSpeech{}, nil, &core.ProviderError{Message: "the Edge TTS speed is out of range", Class: core.ProviderErrorInvalidRequest}
	}
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(body)) {
		value := body[field]
		loss := core.Loss{Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Edge TTS does not read this speech field"}
		switch field {
		case "model", "input", "response_format", "speed":
			continue
		case "voice":
			asked, isString := value.(string)
			if isString && (strings.TrimSpace(asked) == "" || strings.EqualFold(p.speaks(asked), speech.voice)) {
				continue
			}
			loss.Detail = "Edge TTS speaks the voice the model names"
		case "stream_format":
			if value == "audio" {
				loss.Detail = "Edge TTS answers with the whole audio anyway"
			} else {
				loss.Severity = translate.LossMaterial
			}
		}
		if value != nil {
			losses = append(losses, loss)
		}
	}
	return speech, losses, nil
}

// speaks is the voice a model or a voice field names.
func (p *EdgeTTS) speaks(name string) string {
	name = strings.TrimSpace(name)
	if name == "" || strings.EqualFold(name, "default") {
		return p.voice
	}
	return name
}
