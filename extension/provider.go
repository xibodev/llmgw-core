package extension

import (
	"context"
	"slices"

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

// ListModels returns the provider's catalog for credential.
func (p *Provider) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	return p.client.ListModels(ctx, p.info.ID, credential)
}

func (p *Provider) serves(request core.Request) error {
	if !slices.Contains(p.info.Surfaces, request.Surface) {
		return &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	return nil
}
