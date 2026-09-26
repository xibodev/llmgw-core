package core_test

import (
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// keyedPreserver preserves every native surface for a request with a key.
type keyedPreserver struct{ surfaceProvider }

func (keyedPreserver) PreservesWireFor(credential *core.Credential, _ string, _ core.ModelSurface) bool {
	return credential != nil && credential.APIKey != ""
}

func TestPreservesWireForAsksForTheCredential(t *testing.T) {
	chat, responses := core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses
	keyed := keyedPreserver{surfaceProvider{[]core.ModelSurface{chat}}}
	withKey := &core.Credential{APIKey: "fixture"}
	for name, check := range map[string]struct {
		provider   core.Provider
		credential *core.Credential
		surface    core.ModelSurface
		want       bool
	}{
		"with a key":                     {keyed, withKey, chat, true},
		"without one":                    {keyed, nil, chat, false},
		"a surface it does not serve":    {keyed, withKey, responses, false},
		"through a decorator":            {decorator{Provider: keyed}, withKey, chat, true},
		"a decorator declaring for all":  {declaringDecorator{decorator: decorator{Provider: keyed}, preserves: true}, nil, chat, true},
		"a credential-blind declaration": {preservingProvider{surfaceProvider{[]core.ModelSurface{chat}}, []core.ModelSurface{chat}}, nil, chat, true},
		"no declaration":                 {surfaceProvider{[]core.ModelSurface{chat}}, withKey, chat, false},
		"no provider":                    {nil, withKey, chat, false},
	} {
		t.Run(name, func(t *testing.T) {
			if got := core.PreservesWireFor(check.provider, check.credential, "m", check.surface); got != check.want {
				t.Fatalf("PreservesWireFor = %v, want %v", got, check.want)
			}
		})
	}
	// Without the credential, the declaration that needs one is not read.
	if core.PreservesWire(keyed, "m", chat) {
		t.Fatal("PreservesWire read a declaration that depends on the credential")
	}
}
