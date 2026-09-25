package catalog_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalog"
	"github.com/xibodev/llmgw-core/providers"
)

var readKey = core.CatalogKey{Instance: "catalog-read"}

// Gateway: TestCatalogReadEmptyReplacesStaleRowsAndCachesSuccess.
func TestReadEmptyDiscoveryReplacesStaleRowsAndIsCached(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCatalogStore()
	if _, err := store.Save(ctx, readKey, discovered(fixtureNow.Add(-2*fixtureTTL), 6, "old")); err != nil {
		t.Fatal(err)
	}
	service, _ := newService(store, true)
	upstream := &discoverer{models: []core.ModelInfo{}}
	for index := range 2 {
		read := service.Read(ctx, catalog.Request{Key: readKey, Discover: upstream.Discover})
		if read.Err != nil || len(read.Record.Evidence.Models) != 0 || read.Record.Evidence.ObservedAt.IsZero() ||
			read.Diagnostics.Status != catalog.StatusEmpty || read.Diagnostics.Stale || read.Diagnostics.FromCache != (index == 1) {
			t.Fatalf("read %d: %+v", index, read)
		}
	}
	if upstream.calls.Load() != 1 {
		t.Fatalf("an empty catalog was discovered %d times", upstream.calls.Load())
	}
}

// Gateway: TestCatalogReadStaleFailureIsScopedAndRedacted.
func TestReadStaleFailureKeepsOnlyItsOwnRowsAndQuotesNothing(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCatalogStore()
	owner := core.CatalogKey{Instance: "catalog-read", CredentialKey: "fixture-owner"}
	other := core.CatalogKey{Instance: "catalog-read", CredentialKey: "fixture-other"}
	refreshed := fixtureNow.Add(-2 * fixtureTTL)
	if _, err := store.Save(ctx, owner, discovered(refreshed, 6, "cached-model")); err != nil {
		t.Fatal(err)
	}
	service, _ := newService(store, true)
	upstream := &discoverer{err: catalogFailure(403, "api_key=fixture-secret Authorization: Bearer fixture-token")}
	read := service.Read(ctx, catalog.Request{Key: owner, Discover: upstream.Discover})
	if read.Err == nil || len(read.Record.Evidence.Models) != 1 || !read.Record.Evidence.ObservedAt.Equal(refreshed) ||
		!read.Diagnostics.Stale || !read.Diagnostics.FromCache || read.Diagnostics.UpstreamStatus != 403 ||
		read.Diagnostics.FailureCode != providers.CatalogCodeAuthenticationFailed || read.Diagnostics.Status != catalog.StatusError {
		t.Fatalf("stale failure: %+v", read)
	}
	second := service.Read(ctx, catalog.Request{Key: other, Discover: upstream.Discover})
	if second.Err == nil || len(second.Record.Evidence.Models) != 0 || second.Diagnostics.FromCache || !second.Record.Evidence.ObservedAt.IsZero() {
		t.Fatalf("another key borrowed the stale rows: %+v", second)
	}
	raw, _ := json.Marshal(read.Diagnostics)
	for _, private := range []string{"fixture-secret", "fixture-token", owner.CredentialKey, other.CredentialKey} {
		if strings.Contains(string(raw), private) {
			t.Fatalf("diagnostics disclosed %q: %s", private, raw)
		}
	}
}

// Without KeepStale a failed discovery returns only its error, as a Runtime
// has always done.
func TestReadFailureWithoutKeepStaleReturnsOnlyTheError(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCatalogStore()
	if _, err := store.Save(ctx, readKey, discovered(fixtureNow.Add(-2*fixtureTTL), 6, "old")); err != nil {
		t.Fatal(err)
	}
	service, _ := newService(store, false)
	failed := errors.New("fixture failure")
	read := service.Read(ctx, catalog.Request{Key: readKey, Discover: (&discoverer{err: failed}).Discover})
	if !errors.Is(read.Err, failed) || read.Record.Revision != "" || len(read.Record.Evidence.Models) != 0 ||
		read.Diagnostics.FailureCode != catalog.FailureFailed || read.Diagnostics.Detail != "Provider catalog failed." {
		t.Fatalf("read = %+v", read)
	}
}

// Gateway: TestCatalogReadServiceProjectCacheBoundary.
func TestReadKeysAreIndependent(t *testing.T) {
	t.Parallel()
	service, _ := newService(nil, true)
	upstream := &discoverer{models: []core.ModelInfo{}}
	first := core.CatalogKey{Instance: "catalog-read", CredentialKey: "service#first"}
	second := core.CatalogKey{Instance: "catalog-read", CredentialKey: "service#second"}
	for _, key := range []core.CatalogKey{first, second, first} {
		if read := service.Read(context.Background(), catalog.Request{Key: key, Discover: upstream.Discover}); read.Err != nil {
			t.Fatalf("read: %+v", read)
		}
	}
	if upstream.calls.Load() != 2 {
		t.Fatalf("discoveries = %d, want one per key", upstream.calls.Load())
	}
}

// Gateway: TestCatalogReadUsesDiagnosticSanitizer.
func TestDiagnosticsNeverQuoteAnError(t *testing.T) {
	t.Parallel()
	service, _ := newService(nil, true)
	failed := &providers.CatalogError{Code: providers.CatalogCodeHTTPError, Status: 502,
		Detail: "Authorization: Bearer fixture-private-token api_key=fixture-private-key"}
	read := service.Read(context.Background(), catalog.Request{Key: readKey, Discover: (&discoverer{err: failed}).Discover})
	if read.Err == nil || read.Diagnostics.FailureCode != providers.CatalogCodeHTTPError || read.Diagnostics.UpstreamStatus != 502 {
		t.Fatalf("read = %+v", read)
	}
	if strings.Contains(read.Diagnostics.Detail, "fixture-private") {
		t.Fatalf("diagnostics quoted the error: %s", read.Diagnostics.Detail)
	}
}
