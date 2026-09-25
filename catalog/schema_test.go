package catalog_test

import (
	"context"
	"slices"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalog"
)

// Gateway: TestPreRenameCatalogEntriesAreDiscarded,
// TestPreAnonymousZenFilterCatalogEntriesAreDiscarded and
// TestPreVertexEmptyActionsCatalogEntriesAreDiscarded. A catalog other
// rules wrote is neither served nor kept, and the next discovery replaces it.
func TestSchemaStampRefusesCatalogsOtherRulesWrote(t *testing.T) {
	t.Parallel()
	for _, schema := range []int{0, 1, 4} {
		ctx := context.Background()
		store := core.NewMemoryCatalogStore()
		key := core.CatalogKey{Instance: "vertex_ai"}
		if _, err := store.Save(ctx, key, discovered(fixtureNow, schema, "gemini-1.0-pro")); err != nil {
			t.Fatal(err)
		}
		service, _ := newService(store, true)
		read := service.Cached(ctx, key)
		if len(read.Record.Evidence.Models) != 0 || !read.Record.Evidence.ObservedAt.IsZero() || read.Diagnostics.Status != catalog.StatusNotSynced {
			t.Fatalf("schema %d: the stamped catalog was served: %+v", schema, read)
		}
		upstream := &discoverer{models: rows("gemini-2.5-pro"), err: nil}
		read = service.Read(ctx, catalog.Request{Key: key, Discover: upstream.Discover})
		if read.Err != nil || upstream.calls.Load() != 1 || read.Record.Evidence.SchemaVersion != 6 || read.Record.Evidence.Models[0].ID != "gemini-2.5-pro" {
			t.Fatalf("schema %d: read = %+v", schema, read)
		}
	}
}

// Gateway: TestCurrentSchemaCatalogEntriesSurviveAReload.
func TestCurrentSchemaCatalogsSurviveANewService(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCatalogStore()
	key := core.CatalogKey{Instance: "copilot", CredentialKey: "prn_owner"}
	first, _ := newService(store, true)
	upstream := &discoverer{models: []core.ModelInfo{{ID: "gpt-5.5", SupportedAPIs: []string{"/responses"}}}}
	if _, err := first.Refresh(ctx, catalog.Request{Key: key, Discover: upstream.Discover}); err != nil {
		t.Fatal(err)
	}
	restarted, _ := newService(store, true)
	read := restarted.Cached(ctx, key)
	if len(read.Record.Evidence.Models) != 1 || !slices.Equal(read.Record.Evidence.Models[0].SupportedAPIs, []string{"/responses"}) ||
		read.Record.Evidence.ObservedAt.IsZero() || read.Diagnostics.Status != catalog.StatusSynced {
		t.Fatalf("a new service lost the catalog: %+v", read)
	}
}

func TestTTLIsPerInstanceAndNotBeforeExpiresOlderCatalogs(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCatalogStore()
	for _, instance := range []string{"slow", "fast"} {
		if _, err := store.Save(ctx, core.CatalogKey{Instance: instance}, discovered(fixtureNow.Add(-30*time.Minute), 0, "m")); err != nil {
			t.Fatal(err)
		}
	}
	service := catalog.New(catalog.Options{
		Store: store, TTL: 10 * time.Minute, Now: func() time.Time { return fixtureNow },
		InstanceTTL: func(instance string) time.Duration { return map[string]time.Duration{"slow": time.Hour}[instance] },
	})
	if service.TTL("slow") != time.Hour || service.TTL("fast") != 10*time.Minute {
		t.Fatalf("TTL: slow=%v fast=%v", service.TTL("slow"), service.TTL("fast"))
	}
	upstream := &discoverer{models: rows("m")}
	for instance, wantCalls := range map[string]int32{"slow": 0, "fast": 1} {
		before := upstream.calls.Load()
		read := service.Read(ctx, catalog.Request{Key: core.CatalogKey{Instance: instance}, Discover: upstream.Discover})
		if read.Err != nil || upstream.calls.Load()-before != wantCalls {
			t.Fatalf("%s: read = %+v, discoveries = %d", instance, read, upstream.calls.Load()-before)
		}
	}
	if stale := service.Cached(ctx, core.CatalogKey{Instance: "slow"}); stale.Diagnostics.Stale {
		t.Fatalf("a catalog within its instance TTL is stale: %+v", stale)
	}
	notBefore := fixtureNow.Add(-time.Minute)
	read := service.Read(ctx, catalog.Request{Key: core.CatalogKey{Instance: "slow"}, Discover: upstream.Discover, NotBefore: notBefore})
	if read.Err != nil || upstream.calls.Load() != 2 {
		t.Fatalf("a catalog observed before NotBefore was served: %+v", read)
	}
}
