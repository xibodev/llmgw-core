package providers

import (
	"context"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// Invoke speaks one OpenAI speech request: its input, in the voice its
// model names, at the rate its speed asks for. The answer is MP3, so a
// response_format other than mp3 is refused before anything is sent. The
// body's other fields, such as its voice, are reported in Response.Losses.
func (p *EdgeTTS) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	if request.Surface != core.ModelSurfaceAudioSpeech {
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	speech, losses, err := p.speechRequest(request)
	if err != nil {
		return core.Response{}, err
	}
	audio, err := p.Synthesize(ctx, request.Credential, speech.voice, speech.input, speech.rate)
	if err != nil {
		return core.Response{}, err
	}
	return core.Response{Body: audio, ContentType: "audio/mpeg", Losses: losses}, nil
}

// Stream refuses every request, because Edge TTS answers with the whole
// audio. Nothing is sent, and another target may stream.
func (p *EdgeTTS) Stream(_ context.Context, request core.Request) (core.StreamIter, error) {
	if request.Surface != core.ModelSurfaceAudioSpeech {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	return nil, &core.ProviderError{
		Message: "Edge TTS does not stream speech", Class: core.ProviderErrorUnsupported,
		Classification: core.ProviderErrorClassification{FailoverEligible: true},
	}
}

// Synthesize speaks text in voice at rate, a prosody rate such as "+0%" or
// "-25%", as the gateway's EdgeTTSProvider.SynthesizeContext does, and
// returns the audio in EdgeTTSOutputFormat. A blank voice is the default
// voice, and a blank rate "+0%". A voice or rate the SSML cannot carry,
// and text with nothing to speak, are refused before anything is sent. The
// audio of all the chunks is bounded as a complete answer is, to 64 MiB.
func (p *EdgeTTS) Synthesize(ctx context.Context, credential *core.Credential, voice, text, rate string) ([]byte, error) {
	token, err := edgeTTSToken(credential)
	if err != nil {
		return nil, err
	}
	voice = strings.TrimSpace(voice)
	if voice == "" {
		voice = p.voice
	}
	if strings.TrimSpace(rate) == "" {
		rate = "+0%"
	}
	cleaned := edgeTTSSanitize(text)
	switch {
	case !edgeTTSVoicePattern.MatchString(voice):
		return nil, &core.ProviderError{Message: "the Edge TTS voice is not a voice name", Class: core.ProviderErrorInvalidRequest}
	case !edgeTTSRatePattern.MatchString(rate):
		return nil, &core.ProviderError{Message: "the Edge TTS rate must be a signed percentage, such as +0%", Class: core.ProviderErrorInvalidRequest}
	case strings.TrimSpace(cleaned) == "":
		return nil, &core.ProviderError{Message: "Edge TTS has no text to speak", Class: core.ProviderErrorInvalidRequest}
	}
	var audio []byte
	for _, chunk := range edgeTTSSplit(edgeTTSEscapeXML(cleaned), edgeTTSMaxMessageSize) {
		if err := ctx.Err(); err != nil {
			return nil, transportFailure(ctx, "the Edge TTS request was abandoned", err)
		}
		if audio, err = p.speak(ctx, token, voice, chunk, rate, audio); err != nil {
			return nil, err
		}
	}
	if len(audio) == 0 {
		// Another target may serve the request, but the service answered,
		// so it counts against nothing, as in the gateway.
		return nil, &core.ProviderError{
			Message: "the Edge TTS service returned no audio", Class: core.ProviderErrorUpstream,
			Classification: core.ProviderErrorClassification{FailoverEligible: true},
		}
	}
	return audio, nil
}

// speak sends one chunk over its own connection, as the gateway does, and
// appends the audio the service sends back until its turn ends. Text
// messages other than turn.end, and binary ones that carry no audio, are
// skipped.
func (p *EdgeTTS) speak(ctx context.Context, token, voice, text, rate string, audio []byte) ([]byte, error) {
	conn, err := p.open(ctx, token)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	exchange, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	stop := context.AfterFunc(exchange, func() { _ = conn.Close() })
	defer stop()

	now := p.now()
	if err := conn.WriteText(exchange, edgeTTSSpeechConfig(now)); err != nil {
		return nil, transportFailure(ctx, "Edge TTS could not send its speech configuration", err)
	}
	if err := conn.WriteText(exchange, edgeTTSSSML(p.newID(), now, voice, rate, text)); err != nil {
		return nil, transportFailure(ctx, "Edge TTS could not send its text", err)
	}
	for {
		messageType, data, err := conn.Read(exchange)
		if err != nil {
			return nil, transportFailure(ctx, "the Edge TTS answer broke off", err)
		}
		switch messageType {
		case WebSocketTextMessage:
			if edgeTTSHeaders(data)["Path"] == "turn.end" {
				return audio, nil
			}
		case WebSocketBinaryMessage:
			if payload, ok := edgeTTSAudio(data); ok {
				if len(audio)+len(payload) > p.maxAudioBytes {
					return nil, unusableResponse("the Edge TTS audio exceeds the size limit", nil)
				}
				audio = append(audio, payload...)
			}
		}
	}
}

// edgeTTSToken reads the access token from a credential: its API key, or
// else its token, and without either the public token. The token is sent
// in URLs as it is, so one that is not URL-safe is a configuration error.
func edgeTTSToken(credential *core.Credential) (string, error) {
	token := ""
	if credential != nil {
		if token = strings.TrimSpace(credential.APIKey); token == "" {
			token = strings.TrimSpace(credential.Token)
		}
	}
	switch {
	case token == "":
		return edgeTTSDefaultToken, nil
	case !edgeTTSTokenPattern.MatchString(token):
		return "", core.NewConfigurationError("the Edge TTS access token may hold only letters, digits and -._~", nil)
	}
	return token, nil
}
