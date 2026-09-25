package core_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// surfaceProvider serves the surfaces it lists natively and answers nothing.
type surfaceProvider struct{ native []core.ModelSurface }

func (p surfaceProvider) NativeSurfaces(string) []core.ModelSurface { return p.native }

func (surfaceProvider) Invoke(context.Context, core.Request) (core.Response, error) {
	return core.Response{}, errors.New("not invoked in these tests")
}

func (surfaceProvider) Stream(context.Context, core.Request) (core.StreamIter, error) {
	return nil, errors.New("not streamed in these tests")
}

func (surfaceProvider) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	return nil, nil
}

// preservingProvider declares the surfaces in preserved.
type preservingProvider struct {
	surfaceProvider
	preserved []core.ModelSurface
}

func (p preservingProvider) PreservesWire(_ string, surface core.ModelSurface) bool {
	return slices.Contains(p.preserved, surface)
}

// decorator wraps a provider and unwraps to it. A nil native forwards the
// wrapped provider's native surfaces; any other narrows them.
type decorator struct {
	core.Provider
	native []core.ModelSurface
}

func (d decorator) NativeSurfaces(model string) []core.ModelSurface {
	if d.native != nil {
		return d.native
	}
	return d.Provider.NativeSurfaces(model)
}

func (d decorator) Unwrap() core.Provider { return d.Provider }

// declaringDecorator changes what it forwards, so it declares for itself.
type declaringDecorator struct {
	decorator
	preserves bool
}

func (d declaringDecorator) PreservesWire(string, core.ModelSurface) bool { return d.preserves }

func TestPreservesWireReadsOnlyTheNearestDeclaration(t *testing.T) {
	chat, responses, messages := core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses, core.ModelSurfaceMessages
	both := []core.ModelSurface{chat, responses}
	// Serves Chat natively by converting it to Responses, as Codex does.
	converting := preservingProvider{surfaceProvider{both}, []core.ModelSurface{responses}}
	for name, check := range map[string]struct {
		provider core.Provider
		surface  core.ModelSurface
		want     bool
	}{
		"declared surface":                 {converting, responses, true},
		"native surface it converts":       {converting, chat, false},
		"surface it does not serve":        {converting, messages, false},
		"declared but not native":          {preservingProvider{surfaceProvider{[]core.ModelSurface{chat}}, both}, responses, false},
		"no declaration":                   {surfaceProvider{both}, chat, false},
		"through a decorator":              {decorator{Provider: converting}, responses, true},
		"through two decorators":           {decorator{Provider: decorator{Provider: converting}}, responses, true},
		"decorator narrowing its surfaces": {decorator{Provider: converting, native: []core.ModelSurface{chat}}, responses, false},
		"decorator declaring for itself":   {declaringDecorator{decorator: decorator{Provider: converting}}, responses, false},
		"decorator over no declaration":    {decorator{Provider: surfaceProvider{both}}, chat, false},
		"no provider":                      {nil, chat, false},
	} {
		t.Run(name, func(t *testing.T) {
			if got := core.PreservesWire(check.provider, "m", check.surface); got != check.want {
				t.Fatalf("PreservesWire(%s) = %v, want %v", check.surface, got, check.want)
			}
		})
	}
}
