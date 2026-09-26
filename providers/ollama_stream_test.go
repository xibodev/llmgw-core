package providers

import (
	"errors"
	"io"
	"slices"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// ollamaFrames drains a stream: its frames, and the error that ended it
// unless that was io.EOF.
func ollamaFrames(t *testing.T, stream core.StreamIter) ([]string, error) {
	t.Helper()
	var frames []string
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			return frames, nil
		}
		if err != nil {
			if _, again := stream.Next(); again != err {
				t.Fatalf("the stream failed with %v, then with %v", err, again)
			}
			return frames, err
		}
		frames = append(frames, string(frame))
	}
}

func ollamaFrame(chunk string) string { return "data: " + chunk + "\n\n" }

// The chunks are what the gateway's ollamaStreamIter returns for the same
// NDJSON, each framed as the gateway's Chat facade writes it, with the
// [DONE] the facade ends a stream with.
func TestOllamaStreamsChunksAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	const prefix, suffix = `{"choices":[{"delta":`, `,"index":0}],"id":"chatcmpl-ollama","model":"qwen3:8b","object":"chat.completion.chunk"}`
	content := func(text string) string {
		return ollamaFrame(prefix + `{"content":"` + text + `"},"finish_reason":null` + suffix)
	}
	stop := ollamaFrame(prefix + `{},"finish_reason":"stop"` + suffix)
	const done = "data: [DONE]\n\n"
	for _, tc := range []struct {
		name, ndjson string
		want         []string
	}{
		{
			name: "content, tools and done",
			ndjson: `{"model":"m","message":{"role":"assistant","content":"Hel"},"done":false}` + "\n" +
				`{"model":"m","message":{"role":"assistant","content":"lo <b>"},"done":false}` + "\n" + "not json\n\n" +
				`{"model":"m","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"lookup","arguments":{"q":"x"}}}]},"done":false}` + "\n" +
				`{"model":"m","message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":3,"eval_count":2}` + "\n",
			want: []string{
				content("Hel"), content(`lo \u003cb\u003e`),
				ollamaFrame(prefix + `{"tool_calls":[{"function":{"arguments":"{\"q\":\"x\"}","name":"lookup"},"id":"call_0","index":0,"type":"function"}]},"finish_reason":null` + suffix),
				stop, done,
			},
		},
		// The gateway ends a stream whose done event carries content with
		// that content, and so with no finish reason.
		{name: "done with content", ndjson: `{"message":{"content":"Hi"},"done":false}` + "\n" + `{"message":{"content":"!"},"done":true}` + "\n" + `{"message":{"content":"late"}}`, want: []string{content("Hi"), content("!"), done}},
		{name: "done only", ndjson: `{"done":true}` + "\n", want: []string{stop, done}},
		{name: "closed before done", ndjson: `{"message":{"content":"Hi"}}` + "\n" + `null`, want: []string{content("Hi"), done}},
	} {
		provider, _ := ollamaDaemon(t, 200, tc.ndjson)
		stream, err := provider.Stream(t.Context(), ollamaChatRequest("qwen3:8b", `{"messages":[{"role":"user","content":"hi"}],"seed":1}`))
		if err != nil {
			t.Fatal(err)
		}
		frames, err := ollamaFrames(t, stream)
		if err != nil || !slices.Equal(frames, tc.want) {
			t.Fatalf("%s: frames = %q, err = %v\nwant %q", tc.name, frames, err, tc.want)
		}
		if losses := lossPaths(core.StreamLosses(stream)); !slices.Equal(losses, []string{"advisory dropped seed"}) || stream.Close() != nil {
			t.Fatalf("%s: losses = %q", tc.name, losses)
		}
	}
}

// Ported from the gateway's TestOllamaStreamNormalAndOversizedRecords, with
// how each failure is classified.
func TestOllamaStreamFailures(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, ndjson string
		frames       int
		class        core.ProviderErrorClass
		want         core.ProviderErrorClassification
	}{
		{name: "oversized", ndjson: `{"message":{"content":"Hi"}}` + "\n" + strings.Repeat("x", maxStreamRecordWireSize+1) + "\n", frames: 1, class: core.ProviderErrorUpstream},
		{name: "empty", ndjson: `{"message":{"content":""},"done":false}` + "\n", class: core.ProviderErrorUpstream, want: core.ProviderErrorClassification{FailoverEligible: true}},
		{name: "nothing", ndjson: "", class: core.ProviderErrorUpstream, want: core.ProviderErrorClassification{FailoverEligible: true}},
	} {
		provider, _ := ollamaDaemon(t, 200, tc.ndjson)
		stream, err := provider.Stream(t.Context(), ollamaChatRequest("m", `{"messages":[]}`))
		if err != nil {
			t.Fatal(err)
		}
		frames, err := ollamaFrames(t, stream)
		_ = stream.Close()
		var failure *core.ProviderError
		if len(frames) != tc.frames || !errors.As(err, &failure) || failure.Class != tc.class || failure.Classification != tc.want {
			t.Fatalf("%s: frames = %q, err = %#v", tc.name, frames, err)
		}
		var tooLarge *streamRecordTooLargeError
		if tc.name == "oversized" && (!errors.As(err, &tooLarge) || tooLarge.format != "NDJSON" || strings.Contains(err.Error(), "xxxx")) {
			t.Fatalf("oversized: err = %#v", err)
		}
	}
	broken := newOllamaStream(t.Context(), io.NopCloser(&partialErrorReader{}), "m", nil)
	_, err := ollamaFrames(t, broken)
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorTransport ||
		failure.Classification != (core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}) {
		t.Fatalf("broken stream: err = %#v", err)
	}
}
