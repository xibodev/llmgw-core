package oauthflowtest

import (
	"context"
	"errors"
	"testing"

	"github.com/xibodev/llmgw-core/oauthflow"
)

func testCreateAndGet(t *testing.T, f *fixture) {
	ctx := context.Background()
	want := f.create(t, "flow-a")
	got, err := f.store.Get(ctx, owner(), "flow-a")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Revision == "" || got.Consumed() {
		t.Fatalf("Get: revision=%q consumed=%v, want a revision and a pending flow", got.Revision, got.Consumed())
	}
	assertSameFlow(t, got, want)
	if got.Secrets.Verifier != want.Secrets.Verifier || got.Secrets.State != want.Secrets.State ||
		got.Secrets.DeviceCode != want.Secrets.DeviceCode || got.Secrets.RedirectURI != want.Secrets.RedirectURI ||
		got.Secrets.DriverData["client_id"] != "fixture-client" || got.Secrets.Params["connection_name"] != "personal" {
		t.Fatal("Get did not round-trip the flow's secrets")
	}
	if _, err := f.store.Get(ctx, owner(), "flow-missing"); !errors.Is(err, oauthflow.ErrFlowNotFound) {
		t.Fatalf("Get of an unknown id: err=%v, want ErrFlowNotFound", err)
	}
}

func testDuplicateCreate(t *testing.T, f *fixture) {
	ctx := context.Background()
	f.create(t, "flow-dup")
	duplicate := f.flow("flow-dup")
	duplicate.Secrets.Verifier = "fixture-verifier-replaced"
	duplicate.Secrets.State = "fixture-state-replaced"
	if err := f.store.Create(ctx, duplicate); !errors.Is(err, oauthflow.ErrFlowExists) {
		t.Fatalf("duplicate Create: err=%v, want ErrFlowExists", err)
	}
	got, err := f.store.Get(ctx, owner(), "flow-dup")
	if err != nil || got.Secrets.Verifier != "fixture-verifier-flow-dup" {
		t.Fatalf("duplicate Create replaced the stored flow: err=%v", err)
	}
}

func testCopies(t *testing.T, f *fixture) {
	ctx := context.Background()
	input := f.create(t, "flow-copy")
	input.Secrets.DriverData["client_id"] = "mutated-input"
	input.Secrets.Params["connection_name"] = "mutated-input"

	got, err := f.store.Get(ctx, owner(), "flow-copy")
	if err != nil {
		t.Fatal(err)
	}
	if got.Secrets.DriverData["client_id"] != "fixture-client" || got.Secrets.Params["connection_name"] != "personal" {
		t.Fatal("Create kept a reference to the caller's maps")
	}
	got.Secrets.DriverData["client_id"] = "mutated-get"
	got.Secrets.Params["connection_name"] = "mutated-get"
	got.Secrets.Verifier = "mutated-get"
	got.Progress.Interval = 0

	consumed, err := f.store.Consume(ctx, owner(), "flow-copy")
	if err != nil {
		t.Fatal(err)
	}
	if consumed.Secrets.DriverData["client_id"] != "fixture-client" || consumed.Secrets.Params["connection_name"] != "personal" ||
		consumed.Secrets.Verifier != "fixture-verifier-flow-copy" || consumed.Progress.Interval == 0 {
		t.Fatal("mutating a returned flow changed the stored flow")
	}
	consumed.Secrets.DriverData["client_id"] = "mutated-consume"
	after, err := f.store.Get(ctx, owner(), "flow-copy")
	if err != nil {
		t.Fatal(err)
	}
	if len(after.Secrets.DriverData) != 0 || len(after.Secrets.Params) != 0 {
		t.Fatal("a consumed flow still holds driver data or params")
	}
}

func testCanceledContext(t *testing.T, f *fixture) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	stored := f.create(t, "flow-live")
	if err := f.store.Create(canceled, f.flow("flow-canceled")); !errors.Is(err, context.Canceled) {
		t.Fatalf("Create: err=%v, want context.Canceled", err)
	}
	if _, err := f.store.Get(context.Background(), owner(), "flow-canceled"); !errors.Is(err, oauthflow.ErrFlowNotFound) {
		t.Fatalf("a canceled Create stored its flow: err=%v", err)
	}
	if _, err := f.store.Get(canceled, owner(), "flow-live"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Get: err=%v, want context.Canceled", err)
	}
	if _, err := f.store.Consume(canceled, owner(), "flow-live"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Consume: err=%v, want context.Canceled", err)
	}
	current, err := f.store.Get(context.Background(), owner(), "flow-live")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.store.Update(canceled, owner(), current); !errors.Is(err, context.Canceled) {
		t.Fatalf("Update: err=%v, want context.Canceled", err)
	}
	if _, _, err := f.store.ResolveState(canceled, stored.Secrets.State); !errors.Is(err, context.Canceled) {
		t.Fatalf("ResolveState: err=%v, want context.Canceled", err)
	}
	if _, err := f.store.Consume(context.Background(), owner(), "flow-live"); err != nil {
		t.Fatalf("a canceled Consume spent the flow: %v", err)
	}
}

func assertSameFlow(t *testing.T, got, want oauthflow.Flow) {
	t.Helper()
	if got.ID != want.ID || got.Caller != want.Caller || got.Instance != want.Instance || got.Method != want.Method ||
		!got.CreatedAt.Equal(want.CreatedAt) || !got.ExpiresAt.Equal(want.ExpiresAt) ||
		got.AuthorizationURL != want.AuthorizationURL || got.Progress.Interval != want.Progress.Interval ||
		!got.Progress.NextPollAt.Equal(want.Progress.NextPollAt) {
		t.Fatalf("stored flow %v differs from %v", got, want)
	}
}
