package extension

import (
	"context"
	"slices"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// Provider adapts one provider a daemon serves to core.Provider. It serves
// the surfaces its ProviderInfo lists; a request for another surface fails
// with a *core.SurfaceError before it reaches the daemon.
type Provider struct {
	client *Client
	info   ProviderInfo
}

var _ core.Provider = (*Provider)(nil)

// NewProvider returns the Provider of info.ID on client's daemon, such as a
// provider from the daemon's Info.
func NewProvider(client *Client, info ProviderInfo) *Provider {
	info.Surfaces = slices.Clone(info.Surfaces)
	info.OAuthMethods = slices.Clone(info.OAuthMethods)
	return &Provider{client: client, info: info}
}

// Info returns what the provider was created from.
func (p *Provider) Info() ProviderInfo {
	info := p.info
	info.Surfaces = slices.Clone(info.Surfaces)
	info.OAuthMethods = slices.Clone(info.OAuthMethods)
	return info
}

// NativeSurfaces returns the provider's surfaces, whatever the model.
func (p *Provider) NativeSurfaces(string) []core.ModelSurface {
	return slices.Clone(p.info.Surfaces)
}

// Invoke performs one non-streaming operation on the daemon.
func (p *Provider) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	if err := p.serves(request); err != nil {
		return core.Response{}, err
	}
	return p.client.Invoke(ctx, p.info.ID, request)
}

// Stream performs one streaming operation on the daemon.
func (p *Provider) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	if err := p.serves(request); err != nil {
		return nil, err
	}
	return p.client.Stream(ctx, p.info.ID, request)
}

// ListModels returns the provider's catalog for credential. A model that
// reports nothing about what it serves, neither SupportedAPIs nor the
// support of an operation or a surface in its Capabilities or
// LegacyCapabilities, serves the provider's surfaces: its SupportedAPIs
// become their paths, as core.SurfacePath gives them, so core infers those
// surfaces for it wherever it reads a row. A surface core does not define
// has no path and is left out. A model that reports what it serves is kept
// as the daemon sent it.
func (p *Provider) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	models, err := p.client.ListModels(ctx, p.info.ID, credential)
	if err != nil {
		return nil, err
	}
	paths := surfacePaths(p.info.Surfaces)
	if len(paths) == 0 {
		return models, nil
	}
	for index := range models {
		if !reportsWhatItServes(models[index]) {
			// Each row gets its own copy, so changing one row changes no other.
			models[index].SupportedAPIs = slices.Clone(paths)
		}
	}
	return models, nil
}

// surfacePaths returns the path of each of surfaces that core defines, once
// and in order.
func surfacePaths(surfaces []core.ModelSurface) []string {
	var paths []string
	for _, surface := range surfaces {
		if path := core.SurfacePath(surface); path != "" && !slices.Contains(paths, path) {
			paths = append(paths, path)
		}
	}
	return paths
}

// reportsWhatItServes reports whether model says anything about what it
// serves: it lists SupportedAPIs, or its Capabilities, or those core infers
// from its LegacyCapabilities, give the support of an operation or a
// surface, whether supported or not.
func reportsWhatItServes(model core.ModelInfo) bool {
	return len(model.SupportedAPIs) > 0 || knowsService(model.Capabilities) ||
		knowsService(core.InferCapabilities(model, time.Time{}, time.Time{}))
}

// knowsService reports whether capabilities give the support of any
// operation or surface.
func knowsService(capabilities *core.ModelCapabilities) bool {
	return capabilities != nil && (capabilities.Operations != (core.ModelOperationCapabilities{}) ||
		capabilities.Surfaces != (core.ModelSurfaceCapabilities{}))
}

func (p *Provider) serves(request core.Request) error {
	if !slices.Contains(p.info.Surfaces, request.Surface) {
		return &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	return nil
}
