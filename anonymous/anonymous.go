// Package anonymous runs the automation that connects the reviewed
// anonymous providers, as the gateway's does: it enrolls each profile's
// provider, discovers what its catalog admits, probes the models with a
// completion and publishes the exact targets that answered.
//
// Policy and persistence stay product code, behind Hooks: whether the
// automation runs, enrolling a provider in the product's configuration,
// the durable claim that checks a provider once a day, the evidence
// generation, and keeping the results. Catalogs and probes go through
// the product's Catalog and Invoker, typically its runtime.Runtime, so
// each provider is read and invoked by its own vertical, with the
// vertical's anonymous admission and the instance's resilience policy.
//
// It replaces providers.AutoConnectAnonymousProviders and
// providers.NewAnonymousOpenAICompatibleAdapter, which read catalogs and
// send probes with their own HTTP client.
package anonymous

import (
	"context"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

// Hooks is what the product decides and keeps.
type Hooks interface {
	// Enabled reports whether the automation runs now. A run consults it
	// before each provider and again once the provider is claimed, so
	// turning the automation off stops a run at the next provider.
	Enabled(ctx context.Context) bool
	// Enroll makes profile's provider one the product serves and the
	// automation manages. It creates a provider that is missing, and never
	// changes or enables one configured otherwise. It returns the
	// provider's instance and StatusManaged, or a status that says why the
	// automation leaves the provider alone, such as a collision.
	Enroll(ctx context.Context, profile providers.AnonymousProviderProfile) (providerID, status string)
	// Claim claims the check of providerID at at. It succeeds at most once
	// every every, across processes, which is how the automation checks a
	// provider once a day however often it runs. A product that reads the
	// provider again under the claim declines one that changed since
	// Enroll.
	Claim(ctx context.Context, providerID string, at time.Time, every time.Duration) (bool, error)
	// Generation returns the evidence generation a check of providerID
	// records under. A check that cannot get one fails with
	// FailureEvidenceUnavailable and is not recorded.
	Generation(ctx context.Context, providerID string) (int64, error)
	// Record keeps a check's result. Its Connect carries the whole
	// evidence: the catalog, each probe and the published targets.
	Record(ctx context.Context, providerID string, generation int64, result Result) error
}

// Catalog discovers the models an enrolled instance admits to anonymous
// access, as the instance's vertical admits them. A check reports what it
// discovered, so an implementation reads the catalog as it is now, not as
// a cache kept it.
type Catalog interface {
	Discover(ctx context.Context, caller core.Caller, instance string) ([]core.ModelInfo, error)
}

// CatalogFunc adapts a function to Catalog.
type CatalogFunc func(ctx context.Context, caller core.Caller, instance string) ([]core.ModelInfo, error)

// Discover calls f.
func (f CatalogFunc) Discover(ctx context.Context, caller core.Caller, instance string) ([]core.ModelInfo, error) {
	return f(ctx, caller, instance)
}

// Invoker performs one operation on an instance. runtime.Runtime
// implements it, so a probe reaches the upstream through the instance's
// vertical and resilience policy, as any request does.
type Invoker interface {
	Invoke(ctx context.Context, caller core.Caller, instance string, request core.Request) (core.Response, error)
}

// InvokerFunc adapts a function to Invoker.
type InvokerFunc func(ctx context.Context, caller core.Caller, instance string, request core.Request) (core.Response, error)

// Invoke calls f.
func (f InvokerFunc) Invoke(ctx context.Context, caller core.Caller, instance string, request core.Request) (core.Response, error) {
	return f(ctx, caller, instance, request)
}
