package catalogtest

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// conditional opens a handle as a ConditionalCatalogStore, or skips.
func conditional(t *testing.T, open Opener) core.ConditionalCatalogStore {
	t.Helper()
	store, ok := open(t).(core.ConditionalCatalogStore)
	if !ok {
		t.Skip("the store is not a core.ConditionalCatalogStore")
	}
	return store
}

// runConditional verifies the ConditionalCatalogStore guarantees.
func runConditional(t *testing.T, newBackend NewBackend) {
	t.Helper()
	ctx := context.Background()
	key := core.CatalogKey{Instance: "provider", CredentialKey: "credential"}

	t.Run("conditional saves fence on the revision", func(t *testing.T) {
		store := conditional(t, newBackend(t))
		if record, err := store.Load(ctx, key); !errors.Is(err, core.ErrCatalogNotFound) || record.Revision != "" {
			t.Fatalf("a key never saved: revision=%q err=%v, want no revision", record.Revision, err)
		}
		first, err := store.SaveIf(ctx, key, evidence("a", 1), "")
		if err != nil || first == "" {
			t.Fatalf("SaveIf on a key never saved: revision=%q err=%v", first, err)
		}
		if _, err := store.SaveIf(ctx, key, evidence("b", 2), ""); !errors.Is(err, core.ErrCatalogConflict) {
			t.Fatalf("SaveIf expecting no record: err=%v, want ErrCatalogConflict", err)
		}
		second, err := store.SaveIf(ctx, key, evidence("c", 3), first)
		if err != nil || second == first {
			t.Fatalf("SaveIf with the current revision: revision=%q err=%v", second, err)
		}
		if _, err := store.SaveIf(ctx, key, evidence("d", 4), first); !errors.Is(err, core.ErrCatalogConflict) {
			t.Fatalf("SaveIf with a replaced revision: err=%v, want ErrCatalogConflict", err)
		}
		got, err := store.Load(ctx, key)
		if err != nil || got.Revision != second {
			t.Fatalf("Load: revision=%q err=%v, want %q", got.Revision, err, second)
		}
		assertEvidence(t, got.Evidence, evidence("c", 3))
	})

	t.Run("delete leaves a revision that fences earlier ones", func(t *testing.T) {
		store := conditional(t, newBackend(t))
		saved, err := store.Save(ctx, key, evidence("a", 2))
		if err != nil {
			t.Fatal(err)
		}
		if err := store.Delete(ctx, key); err != nil {
			t.Fatal(err)
		}
		deleted, err := store.Load(ctx, key)
		if !errors.Is(err, core.ErrCatalogNotFound) || deleted.Revision == "" || deleted.Revision == saved || len(deleted.Evidence.Models) != 0 {
			t.Fatalf("Load after Delete: record=%+v err=%v, want ErrCatalogNotFound with a new revision", deleted, err)
		}
		for _, expected := range []string{saved, ""} {
			if _, err := store.SaveIf(ctx, key, evidence("stale", 1), expected); !errors.Is(err, core.ErrCatalogConflict) {
				t.Fatalf("SaveIf expecting %q after Delete: err=%v, want ErrCatalogConflict", expected, err)
			}
		}
		if _, err := store.SaveIf(ctx, key, evidence("b", 3), deleted.Revision); err != nil {
			t.Fatalf("SaveIf expecting the deletion: %v", err)
		}
		got, err := store.Load(ctx, key)
		if err != nil {
			t.Fatal(err)
		}
		assertEvidence(t, got.Evidence, evidence("b", 3))

		// Deleting nothing still fences a writer that saw nothing.
		other := core.CatalogKey{Instance: "other"}
		if err := store.Delete(ctx, other); err != nil {
			t.Fatalf("Delete of a key never saved: %v", err)
		}
		if _, err := store.SaveIf(ctx, other, evidence("stale", 1), ""); !errors.Is(err, core.ErrCatalogConflict) {
			t.Fatalf("SaveIf expecting no record after Delete: err=%v, want ErrCatalogConflict", err)
		}
	})

	t.Run("conditional writes are visible across handles", func(t *testing.T) {
		open := newBackend(t)
		writer, reader := conditional(t, open), conditional(t, open)
		revision, err := writer.SaveIf(ctx, key, evidence("a", 1), "")
		if err != nil {
			t.Fatal(err)
		}
		if err := reader.Delete(ctx, key); err != nil {
			t.Fatal(err)
		}
		if _, err := writer.SaveIf(ctx, key, evidence("stale", 1), revision); !errors.Is(err, core.ErrCatalogConflict) {
			t.Fatalf("the first handle missed the second's Delete: err=%v", err)
		}
		if _, err := writer.Load(ctx, key); !errors.Is(err, core.ErrCatalogNotFound) {
			t.Fatalf("Load after another handle's Delete: err=%v, want ErrCatalogNotFound", err)
		}
	})

	t.Run("concurrent conditional saves admit one writer", func(t *testing.T) {
		open := newBackend(t)
		stores := []core.ConditionalCatalogStore{conditional(t, open), conditional(t, open)}
		base, err := stores[0].Save(ctx, key, evidence("base", 1))
		if err != nil {
			t.Fatal(err)
		}
		var won, lost atomic.Int32
		var writers sync.WaitGroup
		for index := range 16 {
			writers.Add(1)
			go func() {
				defer writers.Done()
				_, err := stores[index%2].SaveIf(ctx, key, evidence("w", 2), base)
				switch {
				case err == nil:
					won.Add(1)
				case errors.Is(err, core.ErrCatalogConflict):
					lost.Add(1)
				default:
					t.Error(err)
				}
			}()
		}
		writers.Wait()
		if won.Load() != 1 || lost.Load() != 15 {
			t.Fatalf("won=%d lost=%d, want exactly one writer", won.Load(), lost.Load())
		}
	})
}
