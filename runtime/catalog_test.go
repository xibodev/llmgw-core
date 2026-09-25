package runtime_test

import (
	"context"
	"errors"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalog"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
)

// listingHook runs during before the provider it wraps lists its models.
type listingHook struct {
	core.Provider
	during func()
}

func (p listingHook) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	p.during()
	return p.Provider.ListModels(ctx, credential)
}

func TestFailedDiscoveryKeepsStaleRowsOnlyWhenOptedIn(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Unix(1_800_000_000, 0).UTC()
	clock := func() time.Time { return now }
	unavailable := &core.ProviderError{Message: "unavailable", Class: core.ProviderErrorUpstream,
		Classification: core.ProviderErrorClassification{StatusCode: 503, Retryable: true}}
	for _, keepStale := range []bool{false, true} {
		var built *fakeProvider
		options := coreruntime.Options[settings]{Providers: factory(&counters{}, func(p *fakeProvider) { built = p }), Now: clock}
		if keepStale {
			options.CatalogService = catalog.New(catalog.Options{TTL: time.Hour, KeepStale: true, Now: clock})
		} else {
			options.CatalogTTL = time.Hour
		}
		runtime := mustRuntime(t, options)
		if _, err := runtime.ListModels(ctx, core.LocalCaller(), "p"); err != nil {
			t.Fatal(err)
		}
		built.failWith = unavailable
		now = now.Add(2 * time.Hour)
		record, err := runtime.ListModels(ctx, core.LocalCaller(), "p")
		if !errors.Is(err, unavailable) || (len(record.Evidence.Models) == 1) != keepStale {
			t.Fatalf("keepStale=%v: record=%+v err=%v", keepStale, record, err)
		}
		read := runtime.ReadCatalog(ctx, core.LocalCaller(), "p")
		if keepStale && (!read.Diagnostics.Stale || !read.Diagnostics.FromCache || read.Diagnostics.UpstreamStatus != 503 ||
			read.Diagnostics.Status != catalog.StatusError) {
			t.Fatalf("stale diagnostics = %+v", read.Diagnostics)
		}
	}
}

func TestCachedReadsNeverDiscover(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	calls := &counters{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{Providers: factory(calls, nil)})
	if _, ok := runtime.CachedModel(ctx, "p", "", "model-a"); ok {
		t.Fatal("a cached lookup found a model before any discovery")
	}
	if read := runtime.CachedCatalog(ctx, core.LocalCaller(), "p"); read.Err != nil || read.Diagnostics.Status != catalog.StatusNotSynced {
		t.Fatalf("cached catalog = %+v", read)
	}
	if calls.lists.Load() != 0 {
		t.Fatalf("a cached read discovered %d times", calls.lists.Load())
	}
	if _, err := runtime.ListModels(ctx, core.LocalCaller(), "p"); err != nil {
		t.Fatal(err)
	}
	if row, ok := runtime.CachedModel(ctx, "p", "", "model-a"); !ok || row.ID != "model-a" {
		t.Fatalf("cached model = %+v, %v", row, ok)
	}
	if read := runtime.CachedCatalog(ctx, core.LocalCaller(), "p"); read.Diagnostics.Status != catalog.StatusSynced || calls.lists.Load() != 1 {
		t.Fatalf("cached catalog = %+v, lists = %d", read, calls.lists.Load())
	}
}

// A credential revoked while its catalog was being discovered never leaves
// rows behind: the invalidation fences the discovery.
func TestAnInvalidationFencesTheRuntimesDiscovery(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	var runtime *coreruntime.Runtime[settings]
	key := core.CatalogKey{Instance: "p"}
	runtime = mustRuntime(t, coreruntime.Options[settings]{
		Providers: func(s settings, instance string) (core.Provider, error) {
			inner, _ := factory(&counters{}, nil)(s, instance)
			return listingHook{Provider: inner, during: func() {
				_ = runtime.CatalogService().Invalidate(ctx, key, catalog.Hard)
			}}, nil
		},
	})
	read := runtime.ReadCatalog(ctx, core.LocalCaller(), "p")
	if !errors.Is(read.Err, catalog.ErrStateChanged) || read.Diagnostics.FailureCode != catalog.FailureStateChanged ||
		len(read.Record.Evidence.Models) != 0 {
		t.Fatalf("read = %+v", read)
	}
	if _, ok := runtime.CachedModel(ctx, "p", "", "model-a"); ok {
		t.Fatal("the fenced discovery was stored")
	}
}
