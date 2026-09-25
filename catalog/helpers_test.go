package catalog_test

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalog"
	"github.com/xibodev/llmgw-core/providers"
)

var fixtureNow = time.Date(2026, time.September, 21, 12, 0, 0, 0, time.UTC)

const fixtureTTL = time.Hour

// clock is a settable time source.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

// discoverer lists fixed rows, or fails, and counts its calls.
type discoverer struct {
	calls  atomic.Int32
	models []core.ModelInfo
	err    error
	during func()
}

func (d *discoverer) Discover(context.Context) ([]core.ModelInfo, error) {
	d.calls.Add(1)
	if d.during != nil {
		d.during()
	}
	return d.models, d.err
}

func rows(ids ...string) []core.ModelInfo {
	models := make([]core.ModelInfo, len(ids))
	for index, id := range ids {
		models[index] = core.ModelInfo{ID: id, Object: "model"}
	}
	return models
}

func discovered(observedAt time.Time, schema int, ids ...string) core.CatalogEvidence {
	return core.CatalogEvidence{Status: core.CatalogDiscovered, Models: rows(ids...), ObservedAt: observedAt, SchemaVersion: schema}
}

// catalogFailure is a catalog failure as core's providers report one: a
// safe message, and a cause whose detail may quote the upstream.
func catalogFailure(status int, detail string) error {
	return &core.ProviderError{
		Message: "Provider credential was rejected by the catalog API.",
		Class:   core.ProviderErrorForbidden,
		Cause:   &providers.CatalogError{Code: providers.CatalogCodeAuthenticationFailed, Status: status, Detail: detail},
	}
}

func newService(store core.CatalogStore, keepStale bool) (*catalog.Service, *clock) {
	c := &clock{now: fixtureNow}
	return catalog.New(catalog.Options{Store: store, TTL: fixtureTTL, KeepStale: keepStale, Now: c.Now, SchemaVersion: 6}), c
}

// hookStore runs afterSave once, after the next conditional save succeeds.
type hookStore struct {
	*core.MemoryCatalogStore
	afterSave func()
}

func (s *hookStore) SaveIf(ctx context.Context, key core.CatalogKey, evidence core.CatalogEvidence, expected string) (string, error) {
	revision, err := s.MemoryCatalogStore.SaveIf(ctx, key, evidence, expected)
	if hook := s.afterSave; err == nil && hook != nil {
		s.afterSave = nil
		hook()
	}
	return revision, err
}

// plainStore hides the conditional methods of the store it wraps.
type plainStore struct{ store core.CatalogStore }

func (s plainStore) Load(ctx context.Context, key core.CatalogKey) (core.CatalogRecord, error) {
	return s.store.Load(ctx, key)
}

func (s plainStore) Save(ctx context.Context, key core.CatalogKey, evidence core.CatalogEvidence) (string, error) {
	return s.store.Save(ctx, key, evidence)
}
