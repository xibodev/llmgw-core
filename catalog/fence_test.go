package catalog_test

import (
	"context"
	"errors"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalog"
)

// invalidating discovers rows after invalidating key with each mode.
func invalidating(service *catalog.Service, key core.CatalogKey, modes ...catalog.Invalidation) *discoverer {
	return &discoverer{models: rows("discovered"), during: func() {
		for _, mode := range modes {
			_ = service.Invalidate(context.Background(), key, mode)
		}
	}}
}

// Gateway: TestCatalogInvalidationRejectsInFlightStaleWrite, against a
// conditional store and a plain one.
func TestHardInvalidationFencesADiscoveryInFlight(t *testing.T) {
	t.Parallel()
	for name, store := range map[string]core.CatalogStore{
		"conditional": core.NewMemoryCatalogStore(), "plain": plainStore{core.NewMemoryCatalogStore()},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			ctx := context.Background()
			key := core.CatalogKey{Instance: "copilot", CredentialKey: "owner"}
			service, _ := newService(store, true)
			if _, err := service.Refresh(ctx, catalog.Request{Key: key, Discover: invalidating(service, key, catalog.Hard).Discover}); !errors.Is(err, catalog.ErrStateChanged) {
				t.Fatalf("err = %v, want ErrStateChanged", err)
			}
			if read := service.Cached(ctx, key); len(read.Record.Evidence.Models) != 0 || read.Diagnostics.Status != catalog.StatusNotSynced {
				t.Fatalf("the stale discovery was stored: %+v", read)
			}
			// A discovery that begins after the invalidation stores.
			if record, err := service.Refresh(ctx, catalog.Request{Key: key, Discover: (&discoverer{models: rows("fresh")}).Discover}); err != nil || len(record.Evidence.Models) != 1 {
				t.Fatalf("record = %+v, err = %v", record, err)
			}
		})
	}
}

// Gateway: TestCatalogProviderPersistenceRebasesWithoutWeakeningHardFence.
func TestSoftInvalidationFencesNothingButAHardOneStillDoes(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	key := core.CatalogKey{Instance: "antigravity", CredentialKey: "owner"}
	service, _ := newService(nil, true)
	record, err := service.Refresh(ctx, catalog.Request{Key: key, Discover: invalidating(service, key, catalog.Soft).Discover})
	if err != nil || len(record.Evidence.Models) != 1 {
		t.Fatalf("a soft invalidation fenced the operation that caused it: %+v, %v", record, err)
	}
	if row, ok := service.CachedLookup(ctx, key, "discovered"); !ok || row.ID != "discovered" {
		t.Fatal("the discovery a soft invalidation did not fence was not stored")
	}
	for _, mutation := range []string{"revoke", "reauthorization", "provider config"} {
		t.Run(mutation, func(t *testing.T) {
			discover := invalidating(service, key, catalog.Soft, catalog.Hard).Discover
			if _, err := service.Refresh(ctx, catalog.Request{Key: key, Discover: discover}); !errors.Is(err, catalog.ErrStateChanged) {
				t.Fatalf("a hard invalidation was mistaken for a soft one: err = %v", err)
			}
		})
	}
}

// The gateway's rebase: an operation whose own credential write caused one
// hard invalidation keeps its result, and two still fence it.
func TestRebaseAcceptsExactlyOneHardInvalidation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	key := core.CatalogKey{Instance: "codex", CredentialKey: "owner"}
	service, _ := newService(nil, true)
	accept := func() bool { return true }
	record, err := service.Refresh(ctx, catalog.Request{Key: key, Discover: invalidating(service, key, catalog.Hard).Discover, Rebase: accept})
	if err != nil || len(record.Evidence.Models) != 1 {
		t.Fatalf("record = %+v, err = %v", record, err)
	}
	discover := invalidating(service, key, catalog.Hard, catalog.Hard).Discover
	if _, err := service.Refresh(ctx, catalog.Request{Key: key, Discover: discover, Rebase: accept}); !errors.Is(err, catalog.ErrStateChanged) {
		t.Fatalf("two hard invalidations rebased: err = %v", err)
	}
	refused := func() bool { return false }
	discover = invalidating(service, key, catalog.Hard).Discover
	if _, err := service.Refresh(ctx, catalog.Request{Key: key, Discover: discover, Rebase: refused}); !errors.Is(err, catalog.ErrStateChanged) {
		t.Fatalf("a refused rebase stored: err = %v", err)
	}
}

// Another process's write during a discovery wins, and the read serves it.
func TestADiscoveryNeverOverwritesAnotherWrite(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCatalogStore()
	service, _ := newService(store, true)
	upstream := &discoverer{models: rows("ours"), during: func() {
		_, _ = store.Save(ctx, readKey, discovered(fixtureNow, 6, "theirs"))
	}}
	read := service.Read(ctx, catalog.Request{Key: readKey, Discover: upstream.Discover})
	if row, ok := catalog.Find(read.Record, "theirs"); read.Err != nil || !ok || row.ID != "theirs" {
		t.Fatalf("read = %+v", read)
	}
}
