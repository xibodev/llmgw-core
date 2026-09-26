package runtime_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalog"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
)

// wireProvider serves Chat and Responses natively, preserves only
// Responses, and lists one model whose capabilities it reports with the
// given confidence.
type wireProvider struct {
	confidence core.ModelCapabilityConfidence
	mu         sync.Mutex
	seen       []core.Request
}

func (p *wireProvider) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses}
}

func (p *wireProvider) PreservesWire(_ string, surface core.ModelSurface) bool {
	return surface == core.ModelSurfaceResponses
}

func (p *wireProvider) Invoke(_ context.Context, request core.Request) (core.Response, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.seen = append(p.seen, request)
	return core.Response{Body: []byte(`{"id":"resp_fixture"}`), ContentType: core.ContentTypeJSON}, nil
}

func (p *wireProvider) Stream(context.Context, core.Request) (core.StreamIter, error) {
	return nil, errors.New("not streamed")
}

func (p *wireProvider) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	capabilities := core.AdaptModelCapabilities(map[string]any{"chat": true}, []string{"/v1/responses", "/v1/chat/completions"}, time.Time{}, time.Time{})
	capabilities.Provenance = core.ModelCapabilityProvenance{Source: core.ModelCapabilitySourceUpstreamReported, Confidence: p.confidence}
	return []core.ModelInfo{{ID: "model", SupportedAPIs: []string{"/v1/responses", "/v1/chat/completions"}, Capabilities: capabilities}}, nil
}

func TestTransparentServesOnlyWhatTheCatalogConfirms(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Unix(1_800_000_000, 0).UTC()
	clock := func() time.Time { return now }
	for _, confidence := range []core.ModelCapabilityConfidence{core.ModelCapabilityConfidenceHigh, core.ModelCapabilityConfidenceMedium} {
		provider := &wireProvider{confidence: confidence}
		runtime := mustRuntime(t, coreruntime.Options[settings]{
			Providers:      func(settings, string) (core.Provider, error) { return provider, nil },
			CatalogService: catalog.New(catalog.Options{TTL: 3 * time.Hour, Now: clock}), Now: clock,
		})
		request := func(surface core.ModelSurface, body string) core.Request {
			return core.Request{Surface: surface, Model: "model", Body: []byte(body), ContentType: core.ContentTypeJSON}
		}
		responses := request(core.ModelSurfaceResponses, `{"model":"fixture/model",  "input":"hi"}`)
		refusal := func(request core.Request) *core.TransportRejectError {
			t.Helper()
			_, err := runtime.Transparent(ctx, core.LocalCaller(), "p", request)
			var rejected *core.TransportRejectError
			if !errors.As(err, &rejected) || rejected.Surface != request.Surface || rejected.Model != "model" {
				t.Fatalf("err = %v, want a *core.TransportRejectError", err)
			}
			return rejected
		}
		if rejected := refusal(responses); rejected.Reason != core.TransportRejectNotCataloged {
			t.Fatalf("before discovery: %+v", rejected)
		}
		if _, err := runtime.ListModels(ctx, core.LocalCaller(), "p"); err != nil {
			t.Fatal(err)
		}
		if confidence != core.ModelCapabilityConfidenceHigh {
			if rejected := refusal(responses); rejected.Reason != core.TransportRejectNativeUnconfirmed || !rejected.NativeInterface {
				t.Fatalf("medium confidence: %+v", rejected)
			}
			continue
		}
		response, err := runtime.Transparent(ctx, core.LocalCaller(), "p", responses)
		if err != nil || string(response.Body) != `{"id":"resp_fixture"}` || len(provider.seen) != 1 ||
			string(provider.seen[0].Body) != string(responses.Body) || provider.seen[0].Surface != core.ModelSurfaceResponses {
			t.Fatalf("response = %s, err = %v, seen = %+v", response.Body, err, provider.seen)
		}
		if rejected := refusal(request(core.ModelSurfaceChatCompletions, `{}`)); rejected.Reason != core.TransportRejectNativeUnconfirmed || rejected.NativeInterface {
			t.Fatalf("a surface the provider converts: %+v", rejected)
		}
		if rejected := refusal(request(core.ModelSurfaceResponses, `{"stream":true}`)); rejected.Reason != core.TransportRejectStreaming {
			t.Fatalf("a stream: %+v", rejected)
		}
		if mode := runtime.TransportMode(ctx, core.LocalCaller(), "p", "model", core.ModelSurfaceResponses); mode != core.TransportModeNative {
			t.Fatalf("responses mode = %q", mode)
		}
		now = now.Add(time.Hour)
		if rejected := refusal(responses); rejected.Reason != core.TransportRejectNativeUnconfirmed || !rejected.NativeInterface {
			t.Fatalf("an hour after discovery: %+v", rejected)
		}
		if len(provider.seen) != 1 {
			t.Fatalf("a refused request reached the provider: %+v", provider.seen)
		}
	}
}
