package translation_test

import (
	"context"
	"slices"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/translation"
)

// declaringProvider serves Chat natively, forwards it in its own protocol
// and counts tokens.
type declaringProvider struct {
	fakeProvider
	counted int
}

func (p *declaringProvider) PreservesWire(_ string, surface core.ModelSurface) bool {
	return surface == core.ModelSurfaceChatCompletions
}

func (p *declaringProvider) CountTokens(context.Context, core.TokenCountRequest) (core.TokenCount, error) {
	p.counted++
	return core.TokenCount{InputTokens: 7}, nil
}

// The adapter hands native requests over unchanged, so what the provider
// declares holds through it. A surface the adapter translates is never
// preserved.
func TestAdapterExposesTheProviderDeclarations(t *testing.T) {
	provider := &declaringProvider{fakeProvider: fakeProvider{native: core.ModelSurfaceChatCompletions}}
	adapter := translation.Adapter{Provider: provider}
	if adapter.Unwrap() != core.Provider(provider) {
		t.Fatal("Unwrap does not return the wrapped provider")
	}
	if !core.PreservesWire(adapter, "m", core.ModelSurfaceChatCompletions) {
		t.Fatal("a preserved native surface is not preserved through the adapter")
	}
	for _, translated := range []core.ModelSurface{core.ModelSurfaceMessages, core.ModelSurfaceResponses} {
		if !slices.Contains(adapter.Surfaces("m"), translated) || core.PreservesWire(adapter, "m", translated) {
			t.Fatalf("the translated %s surface reads as preserved, or is not served", translated)
		}
	}
	count, err := core.CountTokens(context.Background(), adapter, core.TokenCountRequest{
		Request: core.Request{Surface: core.ModelSurfaceMessages, Model: "m"},
	})
	if err != nil || count.InputTokens != 7 || provider.counted != 1 {
		t.Fatalf("count = %+v, err = %v, counted %d times", count, err, provider.counted)
	}
}
