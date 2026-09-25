package providers

import (
	"slices"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// edgeTTSSpoken is the SSML a scripted service received, after its headers.
func edgeTTSSpoken(t *testing.T, service *edgeTTSService) []string {
	t.Helper()
	_, conns, _ := service.recorded()
	spoken := make([]string, 0, len(conns))
	for _, conn := range conns {
		written, _ := conn.state()
		if len(written) != 2 {
			t.Fatalf("frames = %q", written)
		}
		_, ssml, _ := strings.Cut(string(written[1]), "\r\n\r\n")
		spoken = append(spoken, ssml)
	}
	return spoken
}

// A speech request's model names the voice, and the body's other fields
// are read as the gateway reads them for a native synthesizer.
func TestEdgeTTSReadsTheSpeechRequest(t *testing.T) {
	t.Parallel()
	answer := []edgeTTSAnswer{{frames: []edgeTTSFrame{edgeTTSAudioFrame("MP3"), edgeTTSTurnEnd()}}}
	for _, tc := range []struct {
		name, model, body, voice, rate string
		losses                         []string
	}{
		{name: "default", model: "Default", body: `{"input":"hi","voice":"default"}`, voice: edgeTTSDefaultVoice, rate: "+0%"},
		{name: "the model's voice", model: "en-GB-SoniaNeural", body: `{"input":"hi","voice":"en-gb-sonianeural","speed":0.5,"response_format":"MP3"}`, voice: "en-GB-SoniaNeural", rate: "-50%"},
		{name: "a default voice by name", model: "default", body: `{"input":"hi","voice":"en-US-EmmaMultilingualNeural","speed":"fast"}`, voice: edgeTTSDefaultVoice, rate: "+0%"},
		{
			name: "fields it does not read", model: "en-US-TestNeural", voice: "en-US-TestNeural", rate: "+25%",
			body:   `{"input":"hi","voice":{"id":"voice_fixture"},"speed":1.25,"instructions":"Whisper.","stream_format":"audio","extra":null}`,
			losses: []string{"advisory dropped instructions", "advisory dropped stream_format", "advisory dropped voice"},
		},
		{name: "a stream", model: "en-US-TestNeural", body: `{"input":"hi","stream_format":"sse"}`, voice: "en-US-TestNeural", rate: "+0%", losses: []string{"material dropped stream_format"}},
	} {
		service := &edgeTTSService{answers: answer}
		response, err := newFixtureEdgeTTS(t, service).Invoke(t.Context(), edgeTTSSpeechRequest(tc.model, tc.body))
		if err != nil || string(response.Body) != "MP3" {
			t.Fatalf("%s: response = %q, err = %v", tc.name, response.Body, err)
		}
		if losses := lossPaths(response.Losses); !slices.Equal(losses, tc.losses) {
			t.Fatalf("%s: losses = %q, want %q", tc.name, losses, tc.losses)
		}
		want := "<voice name='" + tc.voice + "'><prosody pitch='+0Hz' rate='" + tc.rate + "' volume='+0%'>hi</prosody>"
		if spoken := edgeTTSSpoken(t, service); len(spoken) != 1 || !strings.Contains(spoken[0], want) {
			t.Fatalf("%s: SSML = %q, want %q", tc.name, spoken, want)
		}
	}
}

// Text longer than a service message is spoken a chunk per connection, in
// order, as the gateway speaks it, and the audio joined.
func TestEdgeTTSSpeaksLongTextAChunkPerConnection(t *testing.T) {
	t.Parallel()
	var answers []edgeTTSAnswer
	for _, part := range []string{"one ", "two ", "three ", "four"} {
		answers = append(answers, edgeTTSAnswer{frames: []edgeTTSFrame{edgeTTSAudioFrame(part), edgeTTSTurnEnd()}})
	}
	service := &edgeTTSService{answers: answers}
	text := strings.Repeat("Hello <world> & friends. ", 400)
	audio, err := newFixtureEdgeTTS(t, service).Synthesize(t.Context(), &core.Credential{Token: edgeTTSFixtureToken}, "en-US-TestNeural", text, "+5%")
	if err != nil || string(audio) != "one two three four" {
		t.Fatalf("audio = %q, err = %v", audio, err)
	}
	chunks := edgeTTSSplit(edgeTTSEscapeXML(text), edgeTTSMaxMessageSize)
	spoken := edgeTTSSpoken(t, service)
	if len(chunks) != 4 || len(spoken) != 4 {
		t.Fatalf("chunks = %d, spoken = %d", len(chunks), len(spoken))
	}
	for index, ssml := range spoken {
		want := "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='en-US-TestNeural'>" +
			"<prosody pitch='+0Hz' rate='+5%' volume='+0%'>" + chunks[index] + "</prosody></voice></speak>"
		if ssml != want {
			t.Fatalf("chunk %d: SSML = %q", index, ssml)
		}
	}
	dials, _, _ := service.recorded()
	if query := edgeTTSQuery(t, dials[3]); query.Get("Ocp-Apim-Subscription-Key") != edgeTTSFixtureToken {
		t.Fatalf("a credential's token was not the access token: %v", query)
	}
}
