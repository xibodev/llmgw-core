package oauthflowtest

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/oauthflow"
)

func testConsumeIsSingleUse(t *testing.T, f *fixture) {
	ctx := context.Background()
	created := f.create(t, "flow-once")
	before, err := f.store.Get(ctx, owner(), "flow-once")
	if err != nil {
		t.Fatal(err)
	}
	consumed, err := f.store.Consume(ctx, owner(), "flow-once")
	if err != nil {
		t.Fatalf("Consume: %v", err)
	}
	if !consumed.Consumed() || consumed.Revision == before.Revision || consumed.Secrets.Verifier != created.Secrets.Verifier ||
		consumed.Secrets.State != created.Secrets.State || consumed.Secrets.DeviceCode != created.Secrets.DeviceCode {
		t.Fatal("Consume must return the flow with its secrets, marked consumed, at a new revision")
	}
	assertSameFlow(t, consumed, created)
	if _, err := f.store.Consume(ctx, owner(), "flow-once"); !errors.Is(err, oauthflow.ErrFlowNotFound) {
		t.Fatalf("replayed Consume: err=%v, want ErrFlowNotFound", err)
	}
	tombstone, err := f.store.Get(ctx, owner(), "flow-once")
	if err != nil {
		t.Fatalf("Get of a consumed flow: %v", err)
	}
	if !tombstone.Consumed() || tombstone.Secrets.Verifier != "" || tombstone.Secrets.State != "" ||
		tombstone.Secrets.DeviceCode != "" || tombstone.Secrets.RedirectURI != "" {
		t.Fatal("a consumed flow must keep no secrets")
	}
	if _, _, err := f.store.ResolveState(ctx, created.Secrets.State); !errors.Is(err, oauthflow.ErrFlowNotFound) {
		t.Fatalf("ResolveState of a consumed flow: err=%v, want ErrFlowNotFound", err)
	}
}

func testConcurrentConsumes(t *testing.T, f *fixture) {
	created := f.create(t, "flow-race")
	const workers = 32
	var (
		start sync.WaitGroup
		done  sync.WaitGroup
		mu    sync.Mutex
		wins  []oauthflow.Flow
		other []error
	)
	start.Add(1)
	for range workers {
		done.Add(1)
		go func() {
			defer done.Done()
			start.Wait()
			flow, err := f.store.Consume(context.Background(), owner(), "flow-race")
			mu.Lock()
			defer mu.Unlock()
			switch {
			case err == nil:
				wins = append(wins, flow)
			case !errors.Is(err, oauthflow.ErrFlowNotFound):
				other = append(other, err)
			}
		}()
	}
	start.Done()
	done.Wait()
	if len(other) > 0 {
		t.Fatalf("losing consumes must get ErrFlowNotFound, got %v", other)
	}
	if len(wins) != 1 {
		t.Fatalf("%d of %d concurrent consumes succeeded, want exactly 1", len(wins), workers)
	}
	if wins[0].Secrets.Verifier != created.Secrets.Verifier {
		t.Fatal("the winning consume did not receive the flow's secrets")
	}
}

func testOtherCallers(t *testing.T, f *fixture) {
	ctx := context.Background()
	f.create(t, "flow-owned")
	expiring := f.flow("flow-expiring")
	expiring.ExpiresAt = f.clock.Now().Add(time.Minute)
	if err := f.store.Create(ctx, expiring); err != nil {
		t.Fatal(err)
	}
	f.clock.Advance(2 * time.Minute)
	me := owner()
	others := map[string]core.Caller{
		"another id":           {ID: "fixture-other", Kind: me.Kind, ProjectID: me.ProjectID},
		"same id, other kind":  {ID: me.ID, Kind: core.CallerService, ProjectID: me.ProjectID},
		"same id, other proj":  {ID: me.ID, Kind: me.Kind, ProjectID: "fixture-other-project"},
		"same id, no project":  {ID: me.ID, Kind: me.Kind},
		"anonymous":            {Kind: core.CallerAnonymous},
		"local single-user id": core.LocalCaller(),
	}
	current, err := f.store.Get(ctx, me, "flow-owned")
	if err != nil {
		t.Fatal(err)
	}
	for name, caller := range others {
		for _, id := range []string{"flow-owned", "flow-expiring"} {
			if _, err := f.store.Get(ctx, caller, id); !errors.Is(err, oauthflow.ErrFlowNotFound) {
				t.Fatalf("%s: Get %s: err=%v, want ErrFlowNotFound", name, id, err)
			}
			if _, err := f.store.Consume(ctx, caller, id); !errors.Is(err, oauthflow.ErrFlowNotFound) {
				t.Fatalf("%s: Consume %s: err=%v, want ErrFlowNotFound", name, id, err)
			}
		}
		if _, err := f.store.Update(ctx, caller, current); !errors.Is(err, oauthflow.ErrFlowNotFound) {
			t.Fatalf("%s: Update: err=%v, want ErrFlowNotFound", name, err)
		}
	}
	if _, err := f.store.Consume(ctx, me, "flow-owned"); err != nil {
		t.Fatalf("other callers' attempts spent the owner's flow: %v", err)
	}
}
