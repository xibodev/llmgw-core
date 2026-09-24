// Package oauthflowtest is the conformance suite for oauthflow.FlowStore
// implementations. Run it from each implementation's tests:
//
//	func TestConformance(t *testing.T) {
//		oauthflowtest.Run(t, func(now func() time.Time) oauthflow.FlowStore {
//			return mystore.Open(dsn, mystore.WithClock(now))
//		})
//	}
//
// Each subtest gets a fresh store and its own clock, which the suite
// advances to test expiry. Stores must read time only from that clock.
package oauthflowtest

import (
	"context"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/oauthflow"
)

// NewStore returns an empty store that reads time from now.
type NewStore func(now func() time.Time) oauthflow.FlowStore

// Run verifies the FlowStore guarantees.
func Run(t *testing.T, newStore NewStore) {
	t.Helper()
	cases := []struct {
		name string
		run  func(*testing.T, *fixture)
	}{
		{"CreateAndGet", testCreateAndGet},
		{"DuplicateCreateIsRejected", testDuplicateCreate},
		{"ConsumeIsSingleUse", testConsumeIsSingleUse},
		{"ConcurrentConsumesHaveOneWinner", testConcurrentConsumes},
		{"OtherCallersAreRejected", testOtherCallers},
		{"ExpiryRejectsGetAndConsume", testExpiry},
		{"ReturnedFlowsAreCopies", testCopies},
		{"CanceledContextsAreRejected", testCanceledContext},
		{"UpdateComparesRevisions", testUpdate},
		{"ResolveStateFindsOnlyPendingFlows", testResolveState},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clock := &clock{now: time.Unix(1_800_000_000, 0).UTC()}
			tc.run(t, &fixture{store: newStore(clock.Now), clock: clock})
		})
	}
}

type fixture struct {
	store oauthflow.FlowStore
	clock *clock
}

// clock is the time source a subtest controls.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func owner() core.Caller {
	return core.Caller{ID: "fixture-user", Kind: core.CallerHuman, ProjectID: "fixture-project"}
}

// flow returns a pending browser flow of owner that expires in ten minutes,
// with every secret set to a recognisable fixture value.
func (f *fixture) flow(id string) oauthflow.Flow {
	now := f.clock.Now()
	return oauthflow.Flow{
		ID: id, Caller: owner(), Instance: "fixture-provider", Method: oauthflow.MethodBrowser,
		CreatedAt: now, ExpiresAt: now.Add(10 * time.Minute),
		AuthorizationURL: "https://auth.example.test/authorize?state=fixture-state-" + id,
		Progress:         oauthflow.Progress{Interval: 5 * time.Second, NextPollAt: now},
		Secrets: oauthflow.Secrets{
			Verifier: "fixture-verifier-" + id, State: "fixture-state-" + id, DeviceCode: "fixture-device-" + id,
			RedirectURI: "https://app.example.test/oauth/callback",
			DriverData:  map[string]string{"client_id": "fixture-client"},
			Params:      map[string]string{"connection_name": "personal"},
		},
	}
}

// create stores flow(id) and fails the test on error.
func (f *fixture) create(t *testing.T, id string) oauthflow.Flow {
	t.Helper()
	flow := f.flow(id)
	if err := f.store.Create(context.Background(), flow); err != nil {
		t.Fatalf("Create %q: %v", id, err)
	}
	return flow
}
