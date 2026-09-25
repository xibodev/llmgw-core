package runtime_test

import (
	"context"
	"errors"
	"net/http/httptest"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// Gateway: the OpenCode Zen refusal in exactNativeTransparentTarget, and
// TestAnonymousZenSurfacesAreNotWireNative. Anonymous access reshapes every
// request, so its high-confidence models.dev rows never make it
// transparent, and its answers are labelled translated.
func TestTransparentRefusesAnonymousZen(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	backend := &zenRuntimeBackend{}
	server := httptest.NewServer(backend)
	defer server.Close()
	runtime := newZenRuntime(t, server, core.NewMemoryCredentialStore())
	visitor := core.Caller{ID: "visitor", Kind: core.CallerHuman}
	record, err := runtime.ListModels(ctx, visitor, "zen")
	if err != nil || len(record.Evidence.Models) != 2 ||
		record.Evidence.Models[0].Capabilities.Provenance.Confidence != core.ModelCapabilityConfidenceHigh {
		t.Fatalf("anonymous catalog = %+v, err = %v", record, err)
	}
	backend.take()
	_, err = runtime.Transparent(ctx, visitor, "zen", core.Request{
		Surface: core.ModelSurfaceChatCompletions, Model: "chat-fixture-free", ContentType: core.ContentTypeJSON,
		Body: []byte(`{"model":"chat-fixture-free","messages":[{"role":"user","content":"Say hello"}]}`),
	})
	var rejected *core.TransportRejectError
	if !errors.As(err, &rejected) || rejected.Reason != core.TransportRejectNativeUnconfirmed || rejected.NativeInterface {
		t.Fatalf("err = %v, want a refusal without a native interface", err)
	}
	if calls := backend.take(); len(calls) != 0 {
		t.Fatalf("a refused request reached Zen: %+v", calls)
	}
	if mode := runtime.TransportMode(ctx, visitor, "zen", "chat-fixture-free", core.ModelSurfaceChatCompletions); mode != core.TransportModeTranslated {
		t.Fatalf("anonymous mode = %q, want translated", mode)
	}
}
