// Package runtime provides Runtime, the one value that owns an embedding
// product's provider state: the provider registry, the provider cache, the
// credential refresh coordinators, the catalog cache and provider health.
//
// The package keeps no package state and reads no environment. A product
// supplies everything through Options, including its own settings type. The
// Runtime never interprets settings; it notices when their generation changes
// and hands the snapshot to the product's factories.
//
// Import it under another name, such as coreruntime, in a file that also uses
// the standard library's runtime package.
package runtime

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalog"
	"github.com/xibodev/llmgw-core/providers"
)

// DefaultCatalogTTL is how long a stored catalog is served before the Runtime
// rediscovers it, when Options.CatalogTTL is zero.
const DefaultCatalogTTL = 15 * time.Minute

// SettingsSource supplies a product's current settings and a generation that
// changes whenever they do. After a generation change the Runtime rebuilds
// providers and refresh coordinators on their next use, and rediscovers
// catalogs observed before the change, so hot-reloaded settings take effect
// without a restart.
type SettingsSource[S any] interface {
	Snapshot() (settings S, generation uint64)
}

// MemorySettings is the in-memory reference SettingsSource.
type MemorySettings[S any] struct {
	mu         sync.RWMutex
	settings   S
	generation uint64
}

// NewMemorySettings returns settings at generation 1.
func NewMemorySettings[S any](settings S) *MemorySettings[S] {
	return &MemorySettings[S]{settings: settings, generation: 1}
}

// Snapshot implements SettingsSource.
func (m *MemorySettings[S]) Snapshot() (S, uint64) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.settings, m.generation
}

// Update replaces the settings and returns the new generation.
func (m *MemorySettings[S]) Update(settings S) uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.settings = settings
	m.generation++
	return m.generation
}

// ProviderFactory builds the provider of one instance from a settings
// snapshot. It constructs and must not block: the Runtime calls it under its
// lock, once per instance and settings generation.
type ProviderFactory[S any] func(settings S, instance string) (core.Provider, error)

// RefreshFactory returns how one instance refreshes an OAuth credential, or
// nil when its credentials never refresh.
type RefreshFactory[S any] func(settings S, instance string) tokenstore.RefreshFunc

// Options configures a Runtime. Settings and Providers are required.
type Options[S any] struct {
	Settings  SettingsSource[S]
	Providers ProviderFactory[S]
	// Credentials resolves and stores credentials. Nil sends every request
	// without one.
	Credentials core.CredentialStore
	// Refresh supplies each instance's OAuth refresh. Nil means credentials
	// never refresh.
	Refresh RefreshFactory[S]
	// Policy supplies each instance's retry and circuit policy, which the
	// Runtime applies to the provider it binds. Nil applies none.
	Policy PolicyFactory[S]
	// Catalogs stores discovered catalogs. Nil keeps them in this Runtime's
	// memory only.
	Catalogs core.CatalogStore
	// Evidence receives account evidence. Nil discards it.
	Evidence core.EvidenceSink
	// Registry is the product's effective provider registry. Nil uses
	// providers.DefaultRegistry.
	Registry *providers.Registry
	// CatalogTTL bounds how long a stored catalog is served. Zero uses
	// DefaultCatalogTTL.
	CatalogTTL time.Duration
	// CatalogService keeps catalogs instead, with its own store and TTLs,
	// and may keep stale rows. Catalogs and CatalogTTL are then unused.
	CatalogService *catalog.Service
	// Now returns the current time. Nil uses time.Now.
	Now func() time.Time
}

// Runtime performs provider operations for callers. It is safe for concurrent
// use; keep one per process.
type Runtime[S any] struct {
	options Options[S]

	mu                sync.Mutex
	started           bool
	generation        uint64
	settings          S
	settingsChangedAt time.Time
	providers         map[string]core.Provider
	coordinators      map[string]*tokenstore.Coordinator
	health            map[string]core.ProviderHealthEvidence
	catalogs          *catalog.Service
	circuits          circuits
}

var errRefreshUnavailable = errors.New("runtime: this instance has no credential refresh")

// New returns a Runtime.
func New[S any](options Options[S]) (*Runtime[S], error) {
	if options.Settings == nil || options.Providers == nil {
		return nil, errors.New("runtime: Settings and Providers are required")
	}
	if options.Catalogs == nil {
		options.Catalogs = core.NewMemoryCatalogStore()
	}
	if options.Evidence == nil {
		options.Evidence = core.EvidenceSinkFunc(func(context.Context, core.AccountEvidence) {})
	}
	if options.Registry == nil {
		options.Registry = providers.DefaultRegistry()
	}
	if options.CatalogTTL <= 0 {
		options.CatalogTTL = DefaultCatalogTTL
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	catalogs := options.CatalogService
	if catalogs == nil {
		catalogs = catalog.New(catalog.Options{Store: options.Catalogs, TTL: options.CatalogTTL, Now: options.Now})
	}
	return &Runtime[S]{
		options:      options,
		providers:    map[string]core.Provider{},
		coordinators: map[string]*tokenstore.Coordinator{},
		health:       map[string]core.ProviderHealthEvidence{},
		catalogs:     catalogs,
	}, nil
}

// Registry returns the product's effective provider registry.
func (r *Runtime[S]) Registry() *providers.Registry { return r.options.Registry }

// Health returns the last observed health of an instance.
func (r *Runtime[S]) Health(instance string) core.ProviderHealthEvidence {
	r.mu.Lock()
	defer r.mu.Unlock()
	if evidence, ok := r.health[instance]; ok {
		return evidence
	}
	return core.ProviderHealthEvidence{Status: core.ProviderHealthUnknown, ErrorClass: core.ProviderErrorNone}
}

// binding is what one operation needs for an instance, taken at one settings
// generation.
type binding struct {
	provider    core.Provider
	coordinator *tokenstore.Coordinator
}

func (r *Runtime[S]) bind(instance string) (binding, error) {
	settings, generation := r.options.Settings.Snapshot()
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.started || generation != r.generation {
		if r.started {
			r.settingsChangedAt = r.options.Now()
		}
		r.started, r.generation, r.settings = true, generation, settings
		clear(r.providers)
		clear(r.coordinators)
	}
	provider, ok := r.providers[instance]
	if !ok {
		built, err := r.options.Providers(r.settings, instance)
		if err != nil {
			return binding{}, err
		}
		if built == nil {
			return binding{}, core.NewConfigurationError("provider instance "+instance+" is not configured", nil)
		}
		provider = r.resilient(built, instance)
		r.providers[instance] = provider
	}
	if r.options.Credentials == nil {
		return binding{provider: provider}, nil
	}
	coordinator, ok := r.coordinators[instance]
	if !ok {
		var refresh tokenstore.RefreshFunc
		if r.options.Refresh != nil {
			refresh = r.options.Refresh(r.settings, instance)
		}
		if refresh == nil {
			refresh = func(context.Context, tokenstore.Record) (tokenstore.Record, error) {
				return tokenstore.Record{}, errRefreshUnavailable
			}
		}
		var err error
		coordinator, err = tokenstore.NewCoordinator(r.options.Credentials, refresh, tokenstore.WithClock(r.options.Now))
		if err != nil {
			return binding{}, err
		}
		r.coordinators[instance] = coordinator
	}
	return binding{provider: provider, coordinator: coordinator}, nil
}

// credential is the credential one operation uses, and the record it came
// from. A zero credential means the request is sent without one.
type credential struct {
	key    string
	record tokenstore.Record
	value  *core.Credential
}

func (r *Runtime[S]) credential(ctx context.Context, caller core.Caller, instance string, b binding) (credential, error) {
	if r.options.Credentials == nil {
		return credential{}, nil
	}
	key, err := r.options.Credentials.Resolve(ctx, caller, instance)
	if errors.Is(err, core.ErrNoCredential) {
		return credential{}, nil
	}
	if err != nil {
		return credential{}, credentialUnavailable(err)
	}
	record, err := b.coordinator.Token(ctx, key)
	if err != nil {
		return credential{}, credentialUnavailable(err)
	}
	return credential{key: key, record: record, value: core.CredentialFromRecord(key, record)}, nil
}

func credentialUnavailable(err error) error {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	return &core.ProviderError{
		Message:        "the credential for this provider is unavailable",
		Class:          core.ProviderErrorAuth,
		Classification: core.ProviderErrorClassification{FailoverEligible: true},
		Cause:          err,
	}
}

// rejectedCredential reports a failure the upstream attributed to an OAuth
// credential, which one refresh may fix. Static API keys never refresh.
func rejectedCredential(err error, c credential) bool {
	return err != nil && c.key != "" && c.record.TokenType != core.TokenTypeAPIKey &&
		core.ClassifyError(err).StatusCode == http.StatusUnauthorized
}

// refreshed refreshes a rejected credential once, sharing the refresh with
// every concurrent caller that saw the same revision.
func refreshed(ctx context.Context, b binding, c credential) (credential, bool) {
	record, err := b.coordinator.Rejected(ctx, c.key, c.record)
	if err != nil {
		return c, false
	}
	return credential{key: c.key, record: record, value: core.CredentialFromRecord(c.key, record)}, true
}

func invalidRequest(err error) error {
	return &core.ProviderError{Message: err.Error(), Class: core.ProviderErrorInvalidRequest, Cause: err}
}

// Invoke performs one non-streaming operation for caller on instance. An
// OAuth credential the upstream rejects is refreshed once and the request
// replayed.
func (r *Runtime[S]) Invoke(ctx context.Context, caller core.Caller, instance string, request core.Request) (core.Response, error) {
	if err := request.Validate(); err != nil {
		return core.Response{}, invalidRequest(err)
	}
	b, err := r.bind(instance)
	if err != nil {
		return core.Response{}, err
	}
	c, err := r.credential(ctx, caller, instance, b)
	if err != nil {
		r.observe(ctx, caller, instance, c, core.EvidenceOperationInvoke, request.Surface, request.Model, err)
		return core.Response{}, err
	}
	request.Credential = c.value
	response, err := b.provider.Invoke(ctx, request)
	if rejectedCredential(err, c) {
		if next, ok := refreshed(ctx, b, c); ok {
			c = next
			request.Credential = c.value
			response, err = b.provider.Invoke(ctx, request)
		}
	}
	r.observe(ctx, caller, instance, c, core.EvidenceOperationInvoke, request.Surface, request.Model, err)
	return response, err
}

// Stream performs one streaming operation for caller on instance. A rejected
// OAuth credential is refreshed and the stream reopened only while no frame
// has been delivered, that is when opening the stream fails.
func (r *Runtime[S]) Stream(ctx context.Context, caller core.Caller, instance string, request core.Request) (core.StreamIter, error) {
	if err := request.Validate(); err != nil {
		return nil, invalidRequest(err)
	}
	b, err := r.bind(instance)
	if err != nil {
		return nil, err
	}
	c, err := r.credential(ctx, caller, instance, b)
	if err != nil {
		r.observe(ctx, caller, instance, c, core.EvidenceOperationStream, request.Surface, request.Model, err)
		return nil, err
	}
	request.Credential = c.value
	stream, err := b.provider.Stream(ctx, request)
	if rejectedCredential(err, c) {
		if next, ok := refreshed(ctx, b, c); ok {
			c = next
			request.Credential = c.value
			stream, err = b.provider.Stream(ctx, request)
		}
	}
	r.observe(ctx, caller, instance, c, core.EvidenceOperationStream, request.Surface, request.Model, err)
	return stream, err
}

// ListModels returns the catalog of instance for caller. A stored catalog is
// served while it is fresh: younger than the catalog TTL, observed after the
// last settings change, and not a failure. Another process's discovery counts,
// because catalogs are shared through the CatalogStore and keyed by the
// credential that discovered them. Discoveries of one catalog are serialized
// within this Runtime, and fenced against invalidation (see ReadCatalog).
func (r *Runtime[S]) ListModels(ctx context.Context, caller core.Caller, instance string) (core.CatalogRecord, error) {
	read := r.ReadCatalog(ctx, caller, instance)
	return read.Record, read.Err
}

// observe records account evidence for an operation and updates the
// instance's health. Only upstream outcomes, an HTTP status or a transport
// failure, change health: a configuration, request, surface or credential
// error says nothing about the provider. A caller that gave up records
// nothing.
func (r *Runtime[S]) observe(ctx context.Context, caller core.Caller, instance string, c credential, operation string, surface core.ModelSurface, model string, err error) {
	if ctx.Err() != nil {
		return
	}
	now := r.options.Now()
	outcome := core.ProviderHealthEvidence{Status: core.ProviderHealthHealthy, ErrorClass: core.ProviderErrorNone, ObservedAt: now}
	affectsHealth := true
	if err != nil {
		classification := core.ClassifyError(err)
		switch {
		case classification.StatusCode != 0:
			outcome = core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: classification.StatusCode, ObservedAt: now})
		case classification.CircuitFailure:
			outcome = core.ClassifyProviderFailure(core.ProviderFailure{ObservedAt: now, Err: err})
		default:
			affectsHealth = false
			outcome = core.ProviderHealthEvidence{Status: core.ProviderHealthUnknown, ErrorClass: core.ProviderErrorNone, ObservedAt: now}
			var providerErr *core.ProviderError
			if errors.As(err, &providerErr) && providerErr.Class != "" {
				outcome.ErrorClass = providerErr.Class
			}
		}
		if classification.RetryAfter > outcome.RetryAfter {
			outcome.RetryAfter = classification.RetryAfter
		}
	}
	if affectsHealth {
		r.mu.Lock()
		r.health[instance] = outcome
		r.mu.Unlock()
	}
	r.options.Evidence.Record(ctx, core.AccountEvidence{
		Caller: caller, Instance: instance,
		CredentialKey: c.key, CredentialRevision: c.record.Revision, AccountID: c.record.AccountID,
		Operation: operation, Surface: surface, Model: model, Outcome: outcome,
	})
}
