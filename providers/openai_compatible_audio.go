package providers

import (
	"context"
	"mime"
	"net/http"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// The OpenAI audio endpoints, under the base URL.
const (
	openAITranscriptionsPath = "/audio/transcriptions"
	openAISpeechPath         = "/audio/speech"
)

// openAIUnknownContentType labels an answer the upstream sent without a
// content type, as RFC 9110 lets a recipient assume.
const openAIUnknownContentType = "application/octet-stream"

// servesAudio reports an OpenAI audio surface this instance serves.
func (p *OpenAICompatible) servesAudio(surface core.ModelSurface) bool {
	return p.audio && (surface == core.ModelSurfaceAudioTranscriptions || surface == core.ModelSurfaceAudioSpeech)
}

// invokeAudio performs one audio_transcriptions or audio_speech request.
//
// A transcription's multipart upload is sent byte for byte with its own
// Content-Type, boundary included, so the upstream reads the model the
// upload's model field names; the caller builds the upload for the
// request's model. Speech is the request's JSON object with the request's
// model, as Responses is sent. Authorization and headers are the Chat
// surfaces', and so is the classification of a refusal. The answer is
// returned as sent, within the gateway's bound on an inference answer, with
// the upstream's content type: JSON, text, SRT or WebVTT for a
// transcription, as its response_format asks, and audio for speech.
func (p *OpenAICompatible) invokeAudio(ctx context.Context, request core.Request) (core.Response, error) {
	if strings.TrimSpace(request.Model) == "" {
		return core.Response{}, openAIInvalid("an OpenAI-compatible request needs a model", nil)
	}
	access, err := p.access(request.Credential)
	if err != nil {
		return core.Response{}, err
	}
	path, contentType, body := openAITranscriptionsPath, request.ContentType, request.Body
	if request.Surface == core.ModelSurfaceAudioSpeech {
		payload, err := openAIPayload(request)
		if err != nil {
			return core.Response{}, err
		}
		payload["model"] = request.Model
		if body, err = openAIEncode(payload); err != nil {
			return core.Response{}, openAIInvalid("the "+p.label+" request could not be encoded", err)
		}
		path, contentType = openAISpeechPath, core.ContentTypeJSON
	} else if err := openAITranscriptionUpload(request); err != nil {
		return core.Response{}, err
	}
	// The body's own content type is the transport's here, which neither
	// the configured headers nor the credential's replace.
	header := p.header(access, "", false)
	header.Set("Content-Type", contentType)
	response, err := p.send(ctx, path, header, body)
	if err != nil {
		return core.Response{}, err
	}
	raw, err := p.read(ctx, response)
	if err != nil {
		return core.Response{}, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		return core.Response{}, p.refused(response, raw)
	}
	if request.Surface == core.ModelSurfaceAudioSpeech && len(raw) == 0 {
		return core.Response{}, unusableResponse("the "+p.label+" speech answer has no audio", nil)
	}
	answerType := strings.TrimSpace(response.Header.Get("Content-Type"))
	if answerType == "" {
		answerType = openAIUnknownContentType
	}
	return core.Response{Body: raw, ContentType: answerType}, nil
}

// openAITranscriptionUpload refuses a transcription that is not the
// multipart upload the endpoint takes, before anything is sent.
func openAITranscriptionUpload(request core.Request) error {
	mediaType, params, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != "multipart/form-data" || strings.TrimSpace(params["boundary"]) == "" {
		return openAIInvalid("an audio transcription request must be multipart form data", err)
	}
	if len(request.Body) == 0 {
		return openAIInvalid("an audio transcription request needs a body", nil)
	}
	return nil
}

// audioStreamUnsupported refuses to stream an audio surface, as Google
// refuses to stream the surfaces it serves: the surface is native, so the
// refusal is no *core.SurfaceError, but it permits failover to a target
// that streams, and nothing is sent.
func (p *OpenAICompatible) audioStreamUnsupported(surface core.ModelSurface) error {
	return &core.ProviderError{
		Message: p.label + ": streaming is not implemented for " + string(surface) + "; use a non-streaming request",
		Class:   core.ProviderErrorUnsupported, Classification: core.ProviderErrorClassification{FailoverEligible: true},
	}
}
