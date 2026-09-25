package catalog_test

import (
	"context"
	"errors"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalog"
)

// Gateway: TestCatalogReadFailedRefreshCannotRestoreInvalidatedRows.
func TestFailedDiscoveryCannotRestoreInvalidatedRows(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCatalogStore()
	if _, err := store.Save(ctx, readKey, discovered(fixtureNow.Add(-2*fixtureTTL), 6, "old")); err != nil {
		t.Fatal(err)
	}
	service, _ := newService(store, true)
	started, release := make(chan struct{}), make(chan struct{})
	upstream := &discoverer{err: errors.New("fixture failure"), during: func() { close(started); <-release }}
	done := make(chan catalog.Read, 1)
	go func() { done <- service.Read(ctx, catalog.Request{Key: readKey, Discover: upstream.Discover}) }()
	<-started
	if err := service.Invalidate(ctx, readKey, catalog.Hard); err != nil {
		t.Fatal(err)
	}
	close(release)
	read := <-done
	if read.Err == nil || len(read.Record.Evidence.Models) != 0 || read.Diagnostics.FromCache || !read.Record.Evidence.ObservedAt.IsZero() {
		t.Fatalf("the invalidation restored stale rows: %+v", read)
	}
}

// Gateway: TestCatalogReadSuccessfulRefreshUsesAuthoritativeSnapshot. The
// store interleaves a write or an invalidation after the discovery's save,
// before the read takes its final snapshot, without scheduler timing.
func TestReadServesTheAuthoritativeSnapshot(t *testing.T) {
	t.Parallel()
	for _, scenario := range []string{"replaced", "empty", "invalidated"} {
		t.Run(scenario, func(t *testing.T) {
			t.Parallel()
			ctx := context.Background()
			key := core.CatalogKey{Instance: "catalog-read", CredentialKey: "fixture-owner"}
			store := &hookStore{MemoryCatalogStore: core.NewMemoryCatalogStore()}
			service, _ := newService(store, true)
			replacement := discovered(fixtureNow.Add(time.Second), 6, "replacement-model")
			if scenario == "empty" {
				replacement = core.CatalogEvidence{Status: core.CatalogEmpty, ObservedAt: fixtureNow.Add(time.Second), SchemaVersion: 6}
			}
			var stale string
			store.afterSave = func() {
				saved, _ := store.MemoryCatalogStore.Load(ctx, key)
				stale = saved.Revision
				if scenario == "invalidated" {
					_ = store.MemoryCatalogStore.Delete(ctx, key)
					_, _ = store.MemoryCatalogStore.Save(ctx, readKey, discovered(fixtureNow, 6, "other-scope-model"))
					return
				}
				_, _ = store.MemoryCatalogStore.Save(ctx, key, replacement)
			}
			read := service.Read(ctx, catalog.Request{Key: key, Discover: (&discoverer{models: rows("fetched-model")}).Discover})
			if read.Diagnostics.FromCache || read.Diagnostics.Stale {
				t.Fatalf("diagnostics: %+v", read)
			}
			if scenario == "invalidated" {
				if !errors.Is(read.Err, catalog.ErrStateChanged) || read.Diagnostics.Status != catalog.StatusError ||
					read.Diagnostics.FailureCode != catalog.FailureStateChanged || len(read.Record.Evidence.Models) != 0 ||
					!read.Record.Evidence.ObservedAt.IsZero() {
					t.Fatalf("the invalidation restored rows or borrowed another key's: %+v", read)
				}
				if _, err := store.SaveIf(ctx, key, discovered(fixtureNow, 6, "stale"), stale); !errors.Is(err, core.ErrCatalogConflict) {
					t.Fatalf("the invalidation accepted a stale write: %v", err)
				}
				return
			}
			if read.Err != nil || !read.Record.Evidence.ObservedAt.Equal(replacement.ObservedAt) ||
				len(read.Record.Evidence.Models) != len(replacement.Models) {
				t.Fatalf("snapshot: %+v, want %+v", read, replacement)
			}
			if want := map[string]catalog.Status{"replaced": catalog.StatusSynced, "empty": catalog.StatusEmpty}[scenario]; read.Diagnostics.Status != want {
				t.Fatalf("status = %q, want %q", read.Diagnostics.Status, want)
			}
		})
	}
}

// Gateway: TestCachedCatalogReadNeverContactsUpstream.
func TestCachedReadNeverDiscovers(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	service, _ := newService(nil, true)
	upstream := &discoverer{models: rows("network-model")}
	read := service.Cached(ctx, readKey)
	if read.Err != nil || len(read.Record.Evidence.Models) != 0 || read.Diagnostics.Status != catalog.StatusNotSynced || !read.Diagnostics.FromCache {
		t.Fatalf("empty cached read: %+v", read)
	}
	if _, err := service.Refresh(ctx, catalog.Request{Key: readKey, Discover: upstream.Discover}); err != nil {
		t.Fatal(err)
	}
	read = service.Cached(ctx, readKey)
	if row, ok := service.CachedLookup(ctx, readKey, "network-model"); !ok || row.ID != "network-model" ||
		len(read.Record.Evidence.Models) != 1 || upstream.calls.Load() != 1 {
		t.Fatalf("synced cached read: calls=%d read=%+v", upstream.calls.Load(), read)
	}
	if _, ok := service.CachedLookup(ctx, readKey, "missing-model"); ok {
		t.Fatal("a cached lookup found a model the catalog lacks")
	}
}
