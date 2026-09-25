package providers

import (
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// The goldens below are what the gateway's EdgeTTSProvider computes for the
// same clock, IDs and token.
const (
	edgeTTSFixtureToken     = "fixture-token"
	edgeTTSFixtureSignature = "ADB9188F52EF5B80F8AD80F06FA7A17D58DE9EE72945BD4FE3C329599B5CBD02"
	edgeTTSFixtureURL       = "wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1?Ocp-Apim-Subscription-Key=fixture-token" +
		"&Sec-MS-GEC=" + edgeTTSFixtureSignature + "&Sec-MS-GEC-Version=1-140.0.3485.14&ConnectionId=0123456789abcdef0123456789abcdef"
	edgeTTSFixtureConfig = "X-Timestamp:Wed Mar 04 2026 05:06:07 GMT+0000 (Coordinated Universal Time)\r\n" +
		"Content-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n" +
		`{"context":{"synthesis":{"audio":{"metadataoptions":{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"true"},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}`
	edgeTTSFixtureSSML = "X-RequestId:fedcba9876543210fedcba9876543210\r\nContent-Type:application/ssml+xml\r\n" +
		"X-Timestamp:Wed Mar 04 2026 05:06:07 GMT+0000 (Coordinated Universal Time)Z\r\nPath:ssml\r\n\r\n" +
		"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='en-US-TestNeural'>" +
		"<prosody pitch='+0Hz' rate='+10%' volume='+0%'>Hello &lt;world&gt; &amp; friends</prosody></voice></speak>"
)

var edgeTTSFixtureNow = time.Date(2026, time.March, 4, 5, 6, 7, 0, time.UTC)

// newFixtureEdgeTTS speaks to service with a fixed clock and the IDs of
// the goldens, then numbered ones.
func newFixtureEdgeTTS(t *testing.T, service *edgeTTSService, adjust ...func(*EdgeTTSConfig)) *EdgeTTS {
	t.Helper()
	var mu sync.Mutex
	ids := []string{"0123456789abcdef0123456789abcdef", "fedcba9876543210fedcba9876543210"}
	next := 0
	config := EdgeTTSConfig{
		Dial: service.dial, Now: func() time.Time { return edgeTTSFixtureNow },
		NewID: func() string {
			mu.Lock()
			defer mu.Unlock()
			next++
			if next <= len(ids) {
				return ids[next-1]
			}
			return fmt.Sprintf("%032x", next)
		},
	}
	for _, change := range adjust {
		change(&config)
	}
	provider, err := NewEdgeTTS(config)
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func edgeTTSSpeechRequest(model, body string) core.Request {
	return core.Request{
		Surface: core.ModelSurfaceAudioSpeech, Model: model, Body: []byte(body), ContentType: core.ContentTypeJSON,
		Credential: &core.Credential{APIKey: edgeTTSFixtureToken},
	}
}

// Ported from the gateway's TestNewEdgeTTSDefaultsAndOverrides, with what
// the gateway accepts and core refuses.
func TestNewEdgeTTSDefaultsAndOverrides(t *testing.T) {
	t.Parallel()
	dial := (&edgeTTSService{}).dial
	defaults, err := NewEdgeTTS(EdgeTTSConfig{Dial: dial})
	if err != nil || defaults.base != edgeTTSDefaultBase || defaults.voice != edgeTTSDefaultVoice || defaults.DefaultVoice() != edgeTTSDefaultVoice ||
		defaults.insecure || defaults.timeout != 60*time.Second || defaults.client.Timeout != 60*time.Second {
		t.Fatalf("defaults = %+v, err = %v", defaults, err)
	}
	if surfaces := defaults.NativeSurfaces("any"); !reflect.DeepEqual(surfaces, []core.ModelSurface{core.ModelSurfaceAudioSpeech}) {
		t.Fatalf("surfaces = %v", surfaces)
	}
	overridden, err := NewEdgeTTS(EdgeTTSConfig{Dial: dial, BaseURL: "https://relay.example.com/tts/", DefaultVoice: " en-GB-SoniaNeural ", Timeout: 5 * time.Second})
	if err != nil || overridden.base != "relay.example.com/tts" || overridden.voice != "en-GB-SoniaNeural" || overridden.insecure || overridden.timeout != 5*time.Second {
		t.Fatalf("overridden = %+v, err = %v", overridden, err)
	}
	for _, base := range []string{"http://127.0.0.1:9999/base", "ws://127.0.0.1:9999/base/"} {
		if insecure, err := NewEdgeTTS(EdgeTTSConfig{Dial: dial, BaseURL: base}); err != nil || !insecure.insecure || insecure.base != "127.0.0.1:9999/base" {
			t.Fatalf("%s: provider = %+v, err = %v", base, insecure, err)
		}
	}
	for name, config := range map[string]EdgeTTSConfig{
		"no dialer":       {},
		"a query":         {Dial: dial, BaseURL: "https://relay.example.com/tts?key=x"},
		"a fragment":      {Dial: dial, BaseURL: "relay.example.com/tts#x"},
		"no host":         {Dial: dial, BaseURL: "https:///tts"},
		"a markup voice":  {Dial: dial, DefaultVoice: "en-US-Neural'><break/>"},
		"a spoofed voice": {Dial: dial, DefaultVoice: "-x"},
	} {
		var failure *core.ProviderError
		if provider, err := NewEdgeTTS(config); provider != nil || !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration {
			t.Fatalf("%s: provider = %+v, err = %v, want a configuration error", name, provider, err)
		}
	}
}

func TestEdgeTTSSendsTheGatewaysFramesByteForByte(t *testing.T) {
	t.Parallel()
	service := &edgeTTSService{answers: []edgeTTSAnswer{{frames: []edgeTTSFrame{
		edgeTTSTextFrame("X-RequestId:fixture\r\nPath:turn.start\r\n\r\n{}"), edgeTTSAudioFrame("FAKE-MP3-"), edgeTTSAudioFrame("BYTES"), edgeTTSTurnEnd(),
	}}}}
	provider := newFixtureEdgeTTS(t, service)
	response, err := provider.Invoke(t.Context(), edgeTTSSpeechRequest("en-US-TestNeural", `{"model":"tts-1","input":"Hello <world> & friends","voice":"alloy","speed":1.1}`))
	if err != nil || string(response.Body) != "FAKE-MP3-BYTES" || response.ContentType != "audio/mpeg" {
		t.Fatalf("response = %q %q, err = %v", response.Body, response.ContentType, err)
	}
	if losses := lossPaths(response.Losses); !slices.Equal(losses, []string{"advisory dropped voice"}) {
		t.Fatalf("losses = %q", losses)
	}
	dials, conns, bodies := service.recorded()
	want := edgeTTSDialed{url: edgeTTSFixtureURL, subprotocols: []string{"synthesize"}, header: http.Header{
		"User-Agent":    {"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0"},
		"Origin":        {"chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold"},
		"Pragma":        {"no-cache"},
		"Cache-Control": {"no-cache"},
	}}
	if !reflect.DeepEqual(dials, []edgeTTSDialed{want}) || len(conns) != 1 || bodies[0].count() != 1 {
		t.Fatalf("dials = %+v", dials)
	}
	written, closes := conns[0].state()
	if len(written) != 2 || string(written[0]) != edgeTTSFixtureConfig || string(written[1]) != edgeTTSFixtureSSML || closes == 0 {
		t.Fatalf("frames = %q, closes = %d", written, closes)
	}
	if strings.Contains(edgeTTSFixtureSSML, "alloy") {
		t.Fatal("the body's voice reached the SSML")
	}
}
