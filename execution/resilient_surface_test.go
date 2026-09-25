package execution_test

import (
	"context"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

// The port of TestV043NativeModalitiesPreserveResiliencePolicy: speech
// repeats, while a request that may pay for a second result is tried once
// and still guarded.
func TestResilientRepeatsOnlyRepeatableSurfaces(t *testing.T) {
	t.Parallel()
	for surface, want := range map[core.ModelSurface]int{
		core.ModelSurfaceAudioSpeech:         2,
		core.ModelSurfaceImages:              1,
		core.ModelSurfaceEmbeddings:          1,
		core.ModelSurfaceVideos:              1,
		core.ModelSurfaceAudioTranscriptions: 1,
	} {
		t.Run(string(surface), func(t *testing.T) {
			t.Parallel()
			health := gatewayCircuit(newClock(), 1)
			inner := &scripted{outcomes: []error{&invocation{retryable: true, circuitFailure: true}}}
			provider := execution.Resilient(inner, "provider", execution.Policy{Retry: execution.Retry{Attempts: 2}, Health: health})
			_, err := provider.Invoke(context.Background(), request(surface, `{}`))
			if inner.invokes != want || (err == nil) != (want == 2) {
				t.Fatalf("invokes=%d want=%d err=%v", inner.invokes, want, err)
			}
			if available, _ := health.Available("provider"); available == (want == 1) {
				t.Fatalf("available=%v after the operation", available)
			}
		})
	}
	inner := &scripted{always: &invocation{retryable: true, circuitFailure: true}}
	provider := execution.Resilient(inner, "provider", execution.Policy{
		Retry: execution.Retry{Attempts: 3}, Repeatable: func(core.Request) bool { return true },
	})
	if _, _ = provider.Invoke(context.Background(), request(core.ModelSurfaceImages, `{}`)); inner.invokes != 3 {
		t.Fatalf("Repeatable did not replace the default: invokes=%d", inner.invokes)
	}
}

func TestRepeatableRequestReadsResponsesState(t *testing.T) {
	t.Parallel()
	for body, want := range map[string]bool{
		`{"input":"hello","store":false}`:                             true,
		`{"input":[{"type":"message","role":"user","content":"hi"}]}`: true,
		`{"input":"hello","tools":[{"type":"function","name":"f"}]}`:  true,
		`{"input":"hello","prompt":{"id":"pmpt_fixture"}}`:            false,
		`{"input":[{"type":"item_reference","id":"msg_fixture"}]}`:    false,
		`{"input":[{"type":"input_file","file_id":"file_fixture"}]}`:  false,
		`{"input":"hello","tools":[{"container":{"type":"auto"}}]}`:   false,
		`not json`: false,
		`null`:     false,
	} {
		if got := execution.RepeatableRequest(request(core.ModelSurfaceResponses, body)); got != want {
			t.Errorf("RepeatableRequest(%s)=%v want %v", body, got, want)
		}
	}
}

// The wrapper invents no capability and hides none: it counts tokens only
// when the provider it wraps can, through decorators too, and
// core.PreservesWire reads the provider's own declaration.
func TestResilientKeepsTheProvidersDeclarations(t *testing.T) {
	t.Parallel()
	if execution.Resilient(nil, "provider", execution.Policy{}) != nil {
		t.Fatal("a nil provider was wrapped")
	}
	plain := execution.Resilient(&scripted{}, "provider", execution.Policy{})
	if _, ok := plain.(core.TokenCounter); ok {
		t.Fatal("the wrapper invented token counting")
	}
	if _, err := core.CountTokens(context.Background(), plain, core.TokenCountRequest{Request: messagesRequest}); err != core.ErrTokenCountUnsupported {
		t.Fatalf("count err=%v", err)
	}
	inner := &scripted{}
	counting := execution.Resilient(decorator{counter{inner}}, "provider", execution.Policy{})
	count, err := core.CountTokens(context.Background(), counting, core.TokenCountRequest{Request: messagesRequest})
	if err != nil || count.InputTokens != 7 || inner.counts != 1 {
		t.Fatalf("count=%+v err=%v", count, err)
	}
	wrapped := execution.Resilient(preserver{&scripted{}}, "provider", execution.Policy{})
	if !core.PreservesWire(wrapped, "model", core.ModelSurfaceChatCompletions) ||
		core.PreservesWire(wrapped, "model", core.ModelSurfaceResponses) {
		t.Fatal("PreservesWire did not read through the wrapper")
	}
	if unwrapped := wrapped.(interface{ Unwrap() core.Provider }).Unwrap(); unwrapped == nil {
		t.Fatal("the wrapper does not unwrap")
	}
}

// preserver declares that it forwards Chat Completions unchanged.
type preserver struct{ *scripted }

func (preserver) PreservesWire(_ string, surface core.ModelSurface) bool {
	return surface == core.ModelSurfaceChatCompletions
}
