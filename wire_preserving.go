package core

// WirePreserver is an optional Provider interface. A provider implements it
// to declare the surfaces it forwards in their own wire protocol: the
// request reaches the upstream as that surface's request, and the answer
// comes back as that surface's response or frames, with no conversion
// through another surface inside the provider. Setting the model or the
// stream flag, or encoding the JSON again, keeps a surface preserved;
// translating it does not.
//
// Preserving a surface is a stronger claim than serving it natively. Codex
// and Antigravity list Chat Completions among their native surfaces but
// convert it internally, so they serve it without preserving it, and the
// gateway labels such an answer translated. A product's response transport
// label reads PreservesWire, never NativeSurfaces, and preservation is
// never inferred from any other method.
type WirePreserver interface {
	// PreservesWire reports whether the provider forwards surface for
	// model in its own wire protocol. It is consulted only for a surface
	// the provider serves natively.
	PreservesWire(model string, surface ModelSurface) bool
}

// PreservesWire reports whether provider forwards surface for model in its
// own wire protocol: each layer serves the surface natively, and the
// nearest WirePreserver says so.
//
// A decorator that hands the requests of its native surfaces to the
// provider it wraps unchanged, as translation.Adapter does, exposes that
// provider through an Unwrap() Provider method, and PreservesWire looks
// through it. A decorator that changes those requests declares for itself
// instead. A provider that neither declares nor unwraps preserves nothing.
func PreservesWire(provider Provider, model string, surface ModelSurface) bool {
	for provider != nil {
		if !ServesNatively(provider, model, surface) {
			return false
		}
		if preserver, ok := provider.(WirePreserver); ok {
			return preserver.PreservesWire(model, surface)
		}
		provider = unwrapProvider(provider)
	}
	return false
}

// unwrapProvider returns the provider a decorator wraps, or nil.
func unwrapProvider(provider Provider) Provider {
	if wrapper, ok := provider.(interface{ Unwrap() Provider }); ok {
		return wrapper.Unwrap()
	}
	return nil
}
