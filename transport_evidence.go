package core

import (
	"errors"
	"strings"
	"time"
)

// DefaultTransportFreshness is how long after its catalog was refreshed a
// row counts as fresh evidence for transport planning: the gateway's hour,
// whatever TTL the catalog itself has.
const DefaultTransportFreshness = time.Hour

// TransportEvidence is what a stored catalog says about one of its rows for
// transport planning.
type TransportEvidence struct {
	// Row is the catalog row.
	Row ModelInfo
	// RefreshedAt is when the catalog holding Row was refreshed, its
	// CatalogEvidence.ObservedAt. Zero means never.
	RefreshedAt time.Time
	// FreshFor is how long after RefreshedAt the evidence stays fresh. Zero
	// or less uses DefaultTransportFreshness.
	FreshFor time.Duration
}

// Capabilities returns the capabilities to plan with, as the gateway plans:
// the row's reported Capabilities, or those InferCapabilities infers from
// what it lists, with their freshness replaced by the catalog's. They were
// discovered at RefreshedAt and expire FreshFor later, or carry no
// freshness at all when the catalog was never refreshed, so evidence is
// never fresher than the catalog that holds it. Provenance and confidence
// stay the row's: freshness never upgrades evidence.
func (e TransportEvidence) Capabilities() ModelCapabilities {
	capabilities := InferCapabilities(e.Row, e.RefreshedAt, time.Time{})
	if e.Row.Capabilities != nil {
		reported := *e.Row.Capabilities
		capabilities = &reported
	}
	if e.RefreshedAt.IsZero() {
		capabilities.Freshness = ModelCapabilityFreshness{}
		return *capabilities
	}
	freshFor := e.FreshFor
	if freshFor <= 0 {
		freshFor = DefaultTransportFreshness
	}
	discovered := e.RefreshedAt.UTC()
	expires := discovered.Add(freshFor)
	capabilities.Freshness.DiscoveredAt = &discovered
	capabilities.Freshness.ExpiresAt = &expires
	return *capabilities
}

// TransportInterfaces returns the interfaces a catalog row offers for
// transport planning: one for each chat surface in surfaces, the row's
// SupportedAPIs, that ParseSurfacePath recognizes, in their order. An
// interface is native when provider preserves the surface's wire for model
// (see PreservesWire), and adapted otherwise. A surface the row does not
// list gets no interface, so a request for it is adapted or refused. A nil
// provider, one that could not be built, offers none.
func TransportInterfaces(provider Provider, model string, surfaces []string) []TransportInterface {
	if provider == nil {
		return nil
	}
	interfaces := make([]TransportInterface, 0, len(surfaces))
	for _, candidate := range surfaces {
		surface := ParseSurfacePath(candidate)
		if surface == "" {
			continue
		}
		native := SupportUnsupported
		if PreservesWire(provider, model, surface) {
			native = SupportSupported
		}
		interfaces = append(interfaces, TransportInterface{Surface: surface, Native: native})
	}
	return interfaces
}

// ParseSurfacePath returns the chat surface a request path or a catalog
// endpoint names, with or without the /v1 prefix: "/v1/responses" and
// "/responses" are Responses. Anything else, a non-chat surface included,
// is the empty surface, as in the gateway's transport planning.
func ParseSurfacePath(path string) ModelSurface {
	switch strings.ToLower(strings.TrimPrefix(strings.TrimSpace(path), "/v1")) {
	case "/chat/completions":
		return ModelSurfaceChatCompletions
	case "/responses":
		return ModelSurfaceResponses
	case "/messages":
		return ModelSurfaceMessages
	}
	return ""
}

// ErrInvalidTransportRequirement reports a requested transport other than
// transparent.
var ErrInvalidTransportRequirement = errors.New("core: a requested transport must be transparent when present")

// ParseTransportRequirement reads the transport a client requests, such as
// the gateway's X-LLMGW-Transport-Mode header: blank is
// TransportRequirementAny, and transparent, in any case, is
// TransportRequirementTransparent. Anything else, native included, is
// ErrInvalidTransportRequirement, because a client may ask only for
// transparency.
func ParseTransportRequirement(value string) (TransportRequirement, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "":
		return TransportRequirementAny, nil
	case string(TransportRequirementTransparent):
		return TransportRequirementTransparent, nil
	}
	return "", ErrInvalidTransportRequirement
}
