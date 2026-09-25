package runtime

import (
	"context"
	"errors"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalog"
)

// ReadCatalog is ListModels with the read's diagnostics. The catalog keeps
// its rows unless a discovery succeeds: a discovery that fails leaves them,
// and one that an invalidation fenced stores nothing (see
// catalog.Service.Invalidate). With a CatalogService that keeps stale rows,
// Record holds them beside the error of a failed discovery.
func (r *Runtime[S]) ReadCatalog(ctx context.Context, caller core.Caller, instance string) catalog.Read {
	b, err := r.bind(instance)
	if err != nil {
		return catalog.Failed(err)
	}
	c, err := r.credential(ctx, caller, instance, b)
	if err != nil {
		r.observe(ctx, caller, instance, c, core.EvidenceOperationListModels, "", "", err)
		return catalog.Failed(err)
	}
	return r.catalogs.Read(ctx, catalog.Request{
		Key:       core.CatalogKey{Instance: instance, CredentialKey: c.key},
		NotBefore: r.changedAt(),
		Discover: func(ctx context.Context) ([]core.ModelInfo, error) {
			models, err := b.provider.ListModels(ctx, c.value)
			if rejectedCredential(err, c) {
				if next, ok := refreshed(ctx, b, c); ok {
					c = next
					models, err = b.provider.ListModels(ctx, c.value)
				}
			}
			r.observe(ctx, caller, instance, c, core.EvidenceOperationListModels, "", "", err)
			return models, err
		},
	})
}

// CachedCatalog returns the stored catalog ListModels would serve caller,
// fresh or not. It discovers nothing and refreshes no credential, so a read
// path stays safe while an upstream is slow or down.
func (r *Runtime[S]) CachedCatalog(ctx context.Context, caller core.Caller, instance string) catalog.Read {
	if _, err := r.bind(instance); err != nil {
		return catalog.Failed(err)
	}
	key, err := r.credentialKey(ctx, caller, instance)
	if err != nil {
		return catalog.Failed(err)
	}
	return r.catalogs.Cached(ctx, core.CatalogKey{Instance: instance, CredentialKey: key})
}

// CachedModel returns the row of model in the stored catalog of instance
// that credentialKey discovered, without discovering it: the lookup a
// vertical's catalog hook, such as providers.ZenConfig.Models, and
// transport planning need.
func (r *Runtime[S]) CachedModel(ctx context.Context, instance, credentialKey, model string) (core.ModelInfo, bool) {
	return r.catalogs.CachedLookup(ctx, core.CatalogKey{Instance: instance, CredentialKey: credentialKey}, model)
}

// CatalogService returns the service that keeps this Runtime's catalogs,
// for cached reads and invalidation by key.
func (r *Runtime[S]) CatalogService() *catalog.Service { return r.catalogs }

// changedAt is when the settings last changed, or zero.
func (r *Runtime[S]) changedAt() time.Time {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.settingsChangedAt
}

// credentialKey resolves the key of caller's credential without loading or
// refreshing it. The empty key means the request goes without one.
func (r *Runtime[S]) credentialKey(ctx context.Context, caller core.Caller, instance string) (string, error) {
	if r.options.Credentials == nil {
		return "", nil
	}
	key, err := r.options.Credentials.Resolve(ctx, caller, instance)
	if errors.Is(err, core.ErrNoCredential) {
		return "", nil
	}
	if err != nil {
		return "", credentialUnavailable(err)
	}
	return key, nil
}
