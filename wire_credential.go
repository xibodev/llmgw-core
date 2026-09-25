package core

// CredentialWirePreserver is an optional Provider interface for a provider
// whose wire preservation depends on the credential a request carries. A
// provider that reshapes every anonymous request, as OpenCode Zen's
// admission does, preserves nothing for a request without a key, whatever
// it preserves for one with a key.
type CredentialWirePreserver interface {
	// PreservesWireFor reports whether the provider forwards surface for
	// model in its own wire protocol when the request carries credential,
	// which may be nil. It is consulted only for a surface the provider
	// serves natively.
	PreservesWireFor(credential *Credential, model string, surface ModelSurface) bool
}

// PreservesWireFor is PreservesWire for a request that carries credential:
// the nearest layer that declares either interface decides, and a
// CredentialWirePreserver answers for the credential.
func PreservesWireFor(provider Provider, credential *Credential, model string, surface ModelSurface) bool {
	for provider != nil {
		if !ServesNatively(provider, model, surface) {
			return false
		}
		if preserver, ok := provider.(CredentialWirePreserver); ok {
			return preserver.PreservesWireFor(credential, model, surface)
		}
		if preserver, ok := provider.(WirePreserver); ok {
			return preserver.PreservesWire(model, surface)
		}
		provider = unwrapProvider(provider)
	}
	return false
}
