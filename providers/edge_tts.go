package providers

import (
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// EdgeTTSOutputFormat is the audio Edge TTS asks the service for, as the
// gateway does: MP3 at 24 kHz, 48 kbit/s, mono.
const EdgeTTSOutputFormat = "audio-24khz-48kbitrate-mono-mp3"

// EdgeTTSConfig configures EdgeTTS. Dial is required.
type EdgeTTSConfig struct {
	// Dial opens the synthesis websocket. Nil is a configuration error.
	Dial WebSocketDialer
	// BaseURL is the service's host and path. Empty uses Microsoft's,
	// api.msedgeservices.com/tts/cognitiveservices. A scheme is optional:
	// http:// or ws:// opts into plaintext, for a local relay or a test,
	// and any other base uses TLS.
	BaseURL string
	// DefaultVoice is the voice of the model "default". Empty uses
	// en-US-EmmaMultilingualNeural.
	DefaultVoice string
	// Client lists voices. Nil uses a client that times out after Timeout.
	Client *http.Client
	// Timeout bounds each handshake, and each chunk's exchange after it.
	// Zero or less uses 60 seconds, the gateway's default.
	Timeout time.Duration
	// Now is the local clock, which Edge TTS corrects by the skew it
	// learns from the service. Nil uses time.Now.
	Now func() time.Time
	// NewID makes each ConnectionId and X-RequestId. Nil uses 32 random
	// hex digits.
	NewID func() string
}

// EdgeTTS implements core.Provider for the speech service behind
// Microsoft Edge's read-aloud feature, as the gateway's EdgeTTSProvider
// does, over the websocket its WebSocketDialer opens.
//
// Its only surface is audio speech, and a model is a voice, such as
// en-US-EmmaMultilingualNeural; the model "default" is the default voice.
// Invoke takes an OpenAI speech request and answers with MP3. The body's
// voice is not read, because the model names the voice: a product that
// lets a caller choose one in the body makes it the request's model. Edge
// TTS does not stream. ListModels lists the service's voices.
//
// The frames are the gateway's, byte for byte: the speech configuration,
// then SSML with the voice, the rate and the text, cleaned of control
// characters, escaped and split into messages of at most 4096 bytes, each
// spoken over its own connection. A split falls at whitespace where it
// can and never inside an entity, nor, unlike the gateway's, inside a
// character. Every request is signed with Sec-MS-GEC from the service's
// time. A handshake refused with 403 usually means the local clock drifted
// from the service's, so Edge TTS learns the skew from the refusal's Date
// and retries once. The skew is this instance's, where the gateway keeps
// one for its whole process.
//
// A credential's API key, or its token, is the service's access token.
// Without one, Edge TTS sends the public token the read-aloud feature
// sends.
type EdgeTTS struct {
	dial        WebSocketDialer
	base, voice string
	insecure    bool
	client      *http.Client
	timeout     time.Duration
	now         func() time.Time
	newID       func() string
	// maxAudioBytes bounds a synthesis's audio.
	maxAudioBytes int

	// mu guards skew: how many seconds the service's clock is ahead of
	// the local one, as learned from the Date of a refused request.
	mu   sync.Mutex
	skew float64
}

var _ core.Provider = (*EdgeTTS)(nil)

// NewEdgeTTS returns an Edge TTS provider. The base URL is read as the
// gateway reads it, and a base with a query, a fragment or no host is a
// configuration error, as is a default voice the SSML cannot carry.
func NewEdgeTTS(config EdgeTTSConfig) (*EdgeTTS, error) {
	if config.Dial == nil {
		return nil, core.NewConfigurationError("Edge TTS needs a websocket dialer", nil)
	}
	base := strings.TrimSpace(config.BaseURL)
	insecure := strings.HasPrefix(base, "http://") || strings.HasPrefix(base, "ws://")
	for _, prefix := range []string{"https://", "wss://", "http://", "ws://"} {
		base = strings.TrimPrefix(base, prefix)
	}
	base = strings.TrimRight(base, "/")
	if base == "" {
		base = edgeTTSDefaultBase
	}
	if parsed, err := url.Parse("https://" + base); err != nil || parsed.Host == "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, core.NewConfigurationError("the Edge TTS base URL must be a host and a path", err)
	}
	voice := strings.TrimSpace(config.DefaultVoice)
	if voice == "" {
		voice = edgeTTSDefaultVoice
	}
	if !edgeTTSVoicePattern.MatchString(voice) {
		return nil, core.NewConfigurationError("the Edge TTS default voice is not a voice name", nil)
	}
	p := &EdgeTTS{
		dial: config.Dial, base: base, voice: voice, insecure: insecure, client: config.Client,
		timeout: config.Timeout, now: config.Now, newID: config.NewID, maxAudioBytes: inferenceMaxResponseBytes,
	}
	if p.timeout <= 0 {
		p.timeout = 60 * time.Second
	}
	if p.client == nil {
		p.client = &http.Client{Timeout: p.timeout}
	}
	if p.now == nil {
		p.now = time.Now
	}
	if p.newID == nil {
		p.newID = edgeTTSRandomID
	}
	return p, nil
}

// edgeTTSRandomID is 16 random bytes in hex, as the gateway makes an ID.
func edgeTTSRandomID() string {
	var raw [16]byte
	_, _ = rand.Read(raw[:])
	return hex.EncodeToString(raw[:])
}

// NativeSurfaces reports audio speech for every voice.
func (p *EdgeTTS) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceAudioSpeech}
}

// DefaultVoice is the voice of the model "default", and of a blank voice.
func (p *EdgeTTS) DefaultVoice() string { return p.voice }
