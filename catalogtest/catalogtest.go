// Package catalogtest is the conformance suite for core.CatalogStore
// implementations. Run it from each implementation's tests.
package catalogtest

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// Opener returns one handle on shared backing storage. Separate handles model
// separate processes: a file store opens the same path again.
type Opener func(t *testing.T) core.CatalogStore

// NewBackend creates fresh backing storage for one subtest and returns its
// opener.
type NewBackend func(t *testing.T) Opener

// Run verifies the CatalogStore guarantees against fresh backing storage for
// every subtest.
func Run(t *testing.T, newBackend NewBackend) {
	t.Helper()
	ctx := context.Background()
	key := core.CatalogKey{Instance: "provider"}

	t.Run("missing key", func(t *testing.T) {
		store := newBackend(t)(t)
		if _, err := store.Load(ctx, key); !errors.Is(err, core.ErrCatalogNotFound) {
			t.Fatalf("Load of a missing key: err=%v, want ErrCatalogNotFound", err)
		}
	})

	t.Run("round trip", func(t *testing.T) {
		store := newBackend(t)(t)
		want := evidence("a", 3)
		revision, err := store.Save(ctx, key, want)
		if err != nil || revision == "" {
			t.Fatalf("Save: revision=%q err=%v", revision, err)
		}
		got, err := store.Load(ctx, key)
		if err != nil || got.Revision != revision {
			t.Fatalf("Load: revision=%q err=%v, want revision %q", got.Revision, err, revision)
		}
		assertEvidence(t, got.Evidence, want)
	})

	t.Run("revision changes on every save", func(t *testing.T) {
		store := newBackend(t)(t)
		first, err := store.Save(ctx, key, evidence("a", 1))
		if err != nil {
			t.Fatal(err)
		}
		second, err := store.Save(ctx, key, evidence("a", 1))
		if err != nil || second == first {
			t.Fatalf("identical saves share revision %q (err=%v)", first, err)
		}
		if got, err := store.Load(ctx, key); err != nil || got.Revision != second {
			t.Fatalf("Load revision=%q err=%v, want the latest %q", got.Revision, err, second)
		}
	})

	t.Run("keys are independent", func(t *testing.T) {
		store := newBackend(t)(t)
		personal := core.CatalogKey{Instance: key.Instance, CallerID: "user-1"}
		if _, err := store.Save(ctx, key, evidence("shared", 2)); err != nil {
			t.Fatal(err)
		}
		if _, err := store.Save(ctx, personal, evidence("personal", 4)); err != nil {
			t.Fatal(err)
		}
		shared, err := store.Load(ctx, key)
		if err != nil {
			t.Fatal(err)
		}
		assertEvidence(t, shared.Evidence, evidence("shared", 2))
		mine, err := store.Load(ctx, personal)
		if err != nil {
			t.Fatal(err)
		}
		assertEvidence(t, mine.Evidence, evidence("personal", 4))
		if _, err := store.Load(ctx, core.CatalogKey{Instance: "other"}); !errors.Is(err, core.ErrCatalogNotFound) {
			t.Fatalf("another instance: err=%v, want ErrCatalogNotFound", err)
		}
	})

	t.Run("writes are visible across handles", func(t *testing.T) {
		open := newBackend(t)
		writer, reader := open(t), open(t)
		revision, err := writer.Save(ctx, key, evidence("a", 2))
		if err != nil {
			t.Fatal(err)
		}
		got, err := reader.Load(ctx, key)
		if err != nil || got.Revision != revision {
			t.Fatalf("other handle: revision=%q err=%v, want %q", got.Revision, err, revision)
		}
		assertEvidence(t, got.Evidence, evidence("a", 2))
		revision, err = reader.Save(ctx, key, evidence("b", 5))
		if err != nil {
			t.Fatal(err)
		}
		if got, err := writer.Load(ctx, key); err != nil || got.Revision != revision {
			t.Fatalf("first handle missed the second's write: revision=%q err=%v", got.Revision, err)
		}
	})

	t.Run("loads return copies", func(t *testing.T) {
		store := newBackend(t)(t)
		if _, err := store.Save(ctx, key, evidence("a", 2)); err != nil {
			t.Fatal(err)
		}
		got, err := store.Load(ctx, key)
		if err != nil {
			t.Fatal(err)
		}
		got.Evidence.Models[0].ID = "mutated"
		again, err := store.Load(ctx, key)
		if err != nil {
			t.Fatal(err)
		}
		assertEvidence(t, again.Evidence, evidence("a", 2))
	})

	t.Run("concurrent writes never tear", func(t *testing.T) {
		open := newBackend(t)
		writer, reader := open(t), open(t)
		sizes := map[string]int{"a": 3, "b": 5, "c": 7}
		if _, err := writer.Save(ctx, key, evidence("a", sizes["a"])); err != nil {
			t.Fatal(err)
		}
		var writers sync.WaitGroup
		errs := make(chan error, 64)
		for tag, size := range sizes {
			writers.Add(1)
			go func() {
				defer writers.Done()
				for range 25 {
					if _, err := writer.Save(ctx, key, evidence(tag, size)); err != nil {
						errs <- fmt.Errorf("save %s: %w", tag, err)
						return
					}
				}
			}()
		}
		done := make(chan struct{})
		var readers sync.WaitGroup
		for range 2 {
			readers.Add(1)
			go func() {
				defer readers.Done()
				for {
					select {
					case <-done:
						return
					default:
					}
					record, err := reader.Load(ctx, key)
					if err != nil {
						errs <- fmt.Errorf("load: %w", err)
						return
					}
					if err := untorn(record.Evidence, sizes); err != nil {
						errs <- err
						return
					}
				}
			}()
		}
		writers.Wait()
		close(done)
		readers.Wait()
		close(errs)
		for err := range errs {
			t.Error(err)
		}
	})
}

// evidence returns a catalog whose models all carry tag, so a mix of two saves
// is detectable.
func evidence(tag string, models int) core.CatalogEvidence {
	rows := make([]core.ModelInfo, models)
	for index := range rows {
		rows[index] = core.ModelInfo{ID: fmt.Sprintf("%s-%d", tag, index), Object: "model", OwnedBy: tag}
	}
	return core.CatalogEvidence{Status: core.CatalogDiscovered, Models: rows, ObservedAt: time.Unix(1_800_000_000, 0).UTC()}
}

func assertEvidence(t *testing.T, got, want core.CatalogEvidence) {
	t.Helper()
	if got.Status != want.Status || !got.ObservedAt.Equal(want.ObservedAt) || len(got.Models) != len(want.Models) {
		t.Fatalf("evidence=%+v, want %+v", got, want)
	}
	for index := range want.Models {
		if got.Models[index].ID != want.Models[index].ID || got.Models[index].OwnedBy != want.Models[index].OwnedBy {
			t.Fatalf("model %d=%+v, want %+v", index, got.Models[index], want.Models[index])
		}
	}
}

func untorn(got core.CatalogEvidence, sizes map[string]int) error {
	if len(got.Models) == 0 {
		return fmt.Errorf("a load observed an empty catalog")
	}
	tag := got.Models[0].OwnedBy
	if size, ok := sizes[tag]; !ok || len(got.Models) != size {
		return fmt.Errorf("a load observed %d models tagged %q: a torn write", len(got.Models), tag)
	}
	for _, model := range got.Models {
		if model.OwnedBy != tag || !strings.HasPrefix(model.ID, tag+"-") {
			return fmt.Errorf("a load mixed models of %q and %q: a torn write", tag, model.OwnedBy)
		}
	}
	return nil
}
