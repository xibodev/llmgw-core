package oauthflow_test

import (
	"context"
	"errors"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/oauthflow"
	"github.com/xibodev/llmgw-core/oauthflow/oauthflowtest"
)

func TestCappedMemoryFlowStoreConformance(t *testing.T) {
	oauthflowtest.Run(t, func(now func() time.Time) oauthflow.FlowStore {
		return oauthflow.NewMemoryFlowStore(oauthflow.MemoryFlowStoreOptions{Now: now, MaxFlowsPerCaller: 8})
	})
}

func TestMemoryFlowStoreCapsPendingFlowsPerCaller(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	clock := &testClock{now: time.Unix(1_800_000_000, 0).UTC()}
	store := oauthflow.NewMemoryFlowStore(oauthflow.MemoryFlowStoreOptions{Now: clock.Now, MaxFlowsPerCaller: 2})
	me := owner()
	sameIDOtherProject := core.Caller{ID: me.ID, Kind: me.Kind, ProjectID: "project-2"}
	now := clock.Now()
	create := func(caller core.Caller, id string, created time.Time) {
		t.Helper()
		flow := oauthflow.Flow{
			ID: id, Caller: caller, Instance: fixtureInstance, Method: oauthflow.MethodBrowser,
			CreatedAt: created, ExpiresAt: clock.Now().Add(10 * time.Minute),
			Secrets: oauthflow.Secrets{State: "state-" + id, Verifier: "verifier-" + id},
		}
		if err := store.Create(ctx, flow); err != nil {
			t.Fatalf("Create %s: %v", id, err)
		}
	}
	present := func(caller core.Caller, want bool, ids ...string) {
		t.Helper()
		for _, id := range ids {
			_, err := store.Get(ctx, caller, id)
			if want && err != nil {
				t.Fatalf("flow %s was evicted: %v", id, err)
			}
			if !want && !errors.Is(err, oauthflow.ErrFlowNotFound) {
				t.Fatalf("flow %s: err=%v, want ErrFlowNotFound after eviction", id, err)
			}
		}
	}

	create(me, "a", now)
	create(me, "b", now)
	create(sameIDOtherProject, "x", now)
	create(sameIDOtherProject, "y", now)
	create(me, "c", now) // a and b tie on CreatedAt; a was inserted first
	present(me, false, "a")
	present(me, true, "b", "c")
	present(sameIDOtherProject, true, "x", "y")
	if _, _, err := store.ResolveState(ctx, "state-a"); !errors.Is(err, oauthflow.ErrFlowNotFound) {
		t.Fatalf("an evicted flow's state resolved: err=%v", err)
	}

	create(me, "d", now.Add(-time.Hour)) // evicts b; d is older than c by CreatedAt
	create(me, "e", now)                 // evicts d, inserted after c but created first
	present(me, false, "b", "d")
	present(me, true, "c", "e")

	if _, err := store.Consume(ctx, me, "c"); err != nil {
		t.Fatal(err)
	}
	create(me, "f", now) // a consumed flow does not count
	present(me, true, "c", "e", "f")

	clock.Advance(10 * time.Minute) // e and f expire and stop counting
	create(me, "g", clock.Now())
	create(me, "h", clock.Now())
	present(me, true, "g", "h")
	for _, id := range []string{"e", "f"} {
		if _, err := store.Get(ctx, me, id); !errors.Is(err, oauthflow.ErrFlowExpired) {
			t.Fatalf("expired flow %s: err=%v, want ErrFlowExpired, not an eviction", id, err)
		}
	}
	if _, err := store.Get(ctx, sameIDOtherProject, "x"); !errors.Is(err, oauthflow.ErrFlowExpired) {
		t.Fatalf("another caller's flow: err=%v, want ErrFlowExpired: it was never evicted", err)
	}
}

func TestUncappedMemoryFlowStoreKeepsEveryFlow(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := oauthflow.NewMemoryFlowStore(oauthflow.MemoryFlowStoreOptions{})
	for index := range 50 {
		flow := oauthflow.Flow{ID: string(rune('A' + index)), Caller: owner(), ExpiresAt: time.Now().Add(time.Hour)}
		if err := store.Create(ctx, flow); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.Get(ctx, owner(), "A"); err != nil {
		t.Fatalf("an uncapped store evicted a flow: %v", err)
	}
}
