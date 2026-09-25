package providers

import (
	"context"
	"errors"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

func TestEdgeTTSRefusesWhatItCannotSpeakBeforeDialing(t *testing.T) {
	t.Parallel()
	service := &edgeTTSService{answers: []edgeTTSAnswer{{frames: []edgeTTSFrame{edgeTTSAudioFrame("MP3"), edgeTTSTurnEnd()}}}}
	provider := newFixtureEdgeTTS(t, service)
	chat := edgeTTSSpeechRequest("en-US-TestNeural", `{"messages":[]}`)
	chat.Surface = core.ModelSurfaceChatCompletions
	var surfaceErr *core.SurfaceError
	_, invokeErr := provider.Invoke(t.Context(), chat)
	_, streamErr := provider.Stream(t.Context(), chat)
	if !errors.As(invokeErr, &surfaceErr) || !errors.As(streamErr, &surfaceErr) {
		t.Fatalf("chat: %v, %v, want surface errors", invokeErr, streamErr)
	}
	_, err := provider.Stream(t.Context(), edgeTTSSpeechRequest("en-US-TestNeural", `{"input":"hi"}`))
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorUnsupported || core.ClassifyError(err).Disposition() != core.DispositionFailover {
		t.Fatalf("stream: err = %#v, want a refusal that permits failover", err)
	}
	plain := edgeTTSSpeechRequest("en-US-TestNeural", `{"input":"hi"}`)
	plain.ContentType = "text/plain"
	badToken := edgeTTSSpeechRequest("en-US-TestNeural", `{"input":"hi"}`)
	badToken.Credential = &core.Credential{APIKey: "fixture&Sec-MS-GEC=forged"}
	for name, tc := range map[string]struct {
		request core.Request
		class   core.ProviderErrorClass
	}{
		"not JSON":        {plain, core.ProviderErrorInvalidRequest},
		"not an object":   {edgeTTSSpeechRequest("en-US-TestNeural", `["hi"]`), core.ProviderErrorInvalidRequest},
		"null":            {edgeTTSSpeechRequest("en-US-TestNeural", `null`), core.ProviderErrorInvalidRequest},
		"trailing data":   {edgeTTSSpeechRequest("en-US-TestNeural", `{"input":"hi"} {}`), core.ProviderErrorInvalidRequest},
		"no input":        {edgeTTSSpeechRequest("en-US-TestNeural", `{"voice":"alloy"}`), core.ProviderErrorInvalidRequest},
		"blank input":     {edgeTTSSpeechRequest("en-US-TestNeural", `{"input":" \n "}`), core.ProviderErrorInvalidRequest},
		"number input":    {edgeTTSSpeechRequest("en-US-TestNeural", `{"input":5}`), core.ProviderErrorInvalidRequest},
		"control input":   {edgeTTSSpeechRequest("en-US-TestNeural", `{"input":"\u0001\u0002"}`), core.ProviderErrorInvalidRequest},
		"a markup voice":  {edgeTTSSpeechRequest("x' onload='y", `{"input":"hi"}`), core.ProviderErrorInvalidRequest},
		"an element":      {edgeTTSSpeechRequest("<voice>", `{"input":"hi"}`), core.ProviderErrorInvalidRequest},
		"a huge speed":    {edgeTTSSpeechRequest("en-US-TestNeural", `{"input":"hi","speed":1e300}`), core.ProviderErrorInvalidRequest},
		"a wav answer":    {edgeTTSSpeechRequest("en-US-TestNeural", `{"input":"hi","response_format":"wav"}`), core.ProviderErrorUnsupported},
		"an unsafe token": {badToken, core.ProviderErrorConfiguration},
	} {
		_, err := provider.Invoke(t.Context(), tc.request)
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != tc.class {
			t.Fatalf("%s: err = %#v, want %s", name, err, tc.class)
		}
	}
	for name, rate := range map[string]string{"a word": "fast", "markup": "+10%'/><break/>", "unsigned": "10%"} {
		if _, err := provider.Synthesize(t.Context(), nil, "", "hello", rate); core.ClassifyError(err).Disposition() != core.DispositionTerminal || err == nil {
			t.Fatalf("rate %s: err = %v, want an invalid request", name, err)
		}
	}
	if dials, _, _ := service.recorded(); len(dials) != 0 {
		t.Fatalf("dials = %+v, want none", dials)
	}
}

// The service's answer is read as the gateway reads it: messages it does
// not expect are skipped, and a turn without audio is an answer another
// target may serve.
func TestEdgeTTSReadsTheAnswerAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	service := &edgeTTSService{answers: []edgeTTSAnswer{{frames: []edgeTTSFrame{
		{kind: WebSocketBinaryMessage, data: []byte{1}},
		{kind: WebSocketBinaryMessage, data: []byte{0, 200, 'x'}},
		edgeTTSBinaryFrame("Path:audio.metadata\r\n", "{}"),
		edgeTTSTextFrame("Path:audio.metadata\r\n\r\n{}"),
		{kind: 9, data: []byte("Path:turn.end\r\n\r\n")},
		edgeTTSAudioFrame("A"), edgeTTSAudioFrame(""), edgeTTSAudioFrame("B"), edgeTTSTurnEnd(), edgeTTSAudioFrame("late"),
	}}}}
	if audio, err := newFixtureEdgeTTS(t, service).Synthesize(t.Context(), nil, "", "hello", ""); err != nil || string(audio) != "AB" {
		t.Fatalf("audio = %q, err = %v", audio, err)
	}
	silent := &edgeTTSService{answers: []edgeTTSAnswer{{frames: []edgeTTSFrame{edgeTTSTurnEnd()}}}}
	_, err := newFixtureEdgeTTS(t, silent).Synthesize(t.Context(), nil, "", "hello", "")
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorUpstream || failure.Classification != (core.ProviderErrorClassification{FailoverEligible: true}) {
		t.Fatalf("silent: err = %#v", err)
	}
	loud := &edgeTTSService{answers: []edgeTTSAnswer{{frames: []edgeTTSFrame{edgeTTSAudioFrame("ABC"), edgeTTSAudioFrame("DE"), edgeTTSTurnEnd()}}}}
	provider := newFixtureEdgeTTS(t, loud)
	provider.maxAudioBytes = 4
	if _, err := provider.Synthesize(t.Context(), nil, "", "hello", ""); !errors.As(err, &failure) ||
		failure.Classification != (core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}) {
		t.Fatalf("loud: err = %#v, want audio over the bound refused", err)
	}
}

// A read that waits past the exchange's deadline ends when Edge TTS closes
// the connection, and repeats as the gateway's read timeout does; one the
// caller gave up on permits nothing.
func TestEdgeTTSGivesUpWhenItsDeadlineOrTheCallerDoes(t *testing.T) {
	t.Parallel()
	stalled := []edgeTTSAnswer{{frames: []edgeTTSFrame{edgeTTSAudioFrame("A")}}}
	deadline := &edgeTTSService{answers: stalled}
	provider := newFixtureEdgeTTS(t, deadline, func(config *EdgeTTSConfig) { config.Timeout = 20 * time.Millisecond })
	_, err := provider.Synthesize(t.Context(), nil, "", "hello", "")
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorTransport ||
		failure.Classification != (core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}) {
		t.Fatalf("deadline: err = %#v", err)
	}
	caller := &edgeTTSService{answers: stalled}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	_, err = newFixtureEdgeTTS(t, caller).Synthesize(ctx, nil, "", "hello", "")
	if !errors.As(err, &failure) || failure.Classification != (core.ProviderErrorClassification{}) || !errors.Is(ctx.Err(), context.DeadlineExceeded) {
		t.Fatalf("caller: err = %#v", err)
	}
	for _, service := range []*edgeTTSService{deadline, caller} {
		if _, conns, _ := service.recorded(); len(conns) != 1 {
			t.Fatalf("conns = %d", len(conns))
		} else if _, closes := conns[0].state(); closes == 0 {
			t.Fatal("the connection was left open")
		}
	}
}
