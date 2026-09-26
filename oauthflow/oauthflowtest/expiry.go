package oauthflowtest

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/xibodev/llmgw-core/oauthflow"
)

func testExpiry(t *testing.T, f *fixture) {
	ctx := context.Background()
	created := f.create(t, "flow-expiry")
	f.create(t, "flow-finished")
	if _, err := f.store.Consume(ctx, owner(), "flow-finished"); err != nil {
		t.Fatal(err)
	}
	f.clock.Advance(10*time.Minute - time.Second)
	current, err := f.store.Get(ctx, owner(), "flow-expiry")
	if err != nil {
		t.Fatalf("Get one second before expiry: %v", err)
	}
	f.clock.Advance(time.Second)
	if _, err := f.store.Get(ctx, owner(), "flow-expiry"); !errors.Is(err, oauthflow.ErrFlowExpired) {
		t.Fatalf("Get at expiry: err=%v, want ErrFlowExpired", err)
	}
	if _, err := f.store.Consume(ctx, owner(), "flow-expiry"); !errors.Is(err, oauthflow.ErrFlowExpired) {
		t.Fatalf("Consume at expiry: err=%v, want ErrFlowExpired", err)
	}
	if _, err := f.store.Update(ctx, owner(), current); !errors.Is(err, oauthflow.ErrFlowExpired) {
		t.Fatalf("Update at expiry: err=%v, want ErrFlowExpired", err)
	}
	if _, _, err := f.store.ResolveState(ctx, created.Secrets.State); !errors.Is(err, oauthflow.ErrFlowExpired) {
		t.Fatalf("ResolveState at expiry: err=%v, want ErrFlowExpired", err)
	}
	if _, err := f.store.Get(ctx, owner(), "flow-finished"); !errors.Is(err, oauthflow.ErrFlowExpired) {
		t.Fatalf("Get of a consumed flow at expiry: err=%v, want ErrFlowExpired", err)
	}
	f.clock.Advance(time.Minute)
	if _, err := f.store.Consume(ctx, owner(), "flow-expiry"); err == nil {
		t.Fatal("an expired flow was consumed")
	}
}

func testUpdate(t *testing.T, f *fixture) {
	ctx := context.Background()
	created := f.create(t, "flow-update")
	current, err := f.store.Get(ctx, owner(), "flow-update")
	if err != nil {
		t.Fatal(err)
	}
	change := current.Clone()
	change.Progress.Interval = 10 * time.Second
	change.Progress.NextPollAt = f.clock.Now().Add(10 * time.Second)
	change.Instance, change.Method = "fixture-other-provider", oauthflow.MethodDevice
	change.ExpiresAt = f.clock.Now().Add(time.Hour)
	change.Secrets.Verifier, change.Secrets.DeviceCode = "fixture-verifier-injected", "fixture-device-injected"
	updated, err := f.store.Update(ctx, owner(), change)
	if err != nil {
		t.Fatalf("Update at the current revision: %v", err)
	}
	if updated.Revision == current.Revision || updated.Progress.Interval != 10*time.Second ||
		!updated.Progress.NextPollAt.Equal(change.Progress.NextPollAt) {
		t.Fatal("Update must store the progress at a new revision")
	}
	stored, err := f.store.Get(ctx, owner(), "flow-update")
	if err != nil {
		t.Fatal(err)
	}
	if stored.Instance != created.Instance || stored.Method != created.Method || !stored.ExpiresAt.Equal(created.ExpiresAt) ||
		stored.Secrets.Verifier != created.Secrets.Verifier || stored.Secrets.DeviceCode != created.Secrets.DeviceCode {
		t.Fatal("Update changed more than the flow's progress")
	}
	if _, err := f.store.Update(ctx, owner(), change); !errors.Is(err, oauthflow.ErrConflict) {
		t.Fatalf("Update at a stale revision: err=%v, want ErrConflict", err)
	}

	consumed, err := f.store.Consume(ctx, owner(), "flow-update")
	if err != nil {
		t.Fatal(err)
	}
	consumed.Progress.Outcome, consumed.Progress.CredentialKey = oauthflow.OutcomeComplete, "fixture-credential"
	if _, err := f.store.Update(ctx, owner(), consumed); err != nil {
		t.Fatalf("Update recording the outcome of a consumed flow: %v", err)
	}
	finished, err := f.store.Get(ctx, owner(), "flow-update")
	if err != nil {
		t.Fatal(err)
	}
	if finished.Progress.Outcome != oauthflow.OutcomeComplete || finished.Progress.CredentialKey != "fixture-credential" ||
		finished.Secrets.Verifier != "" || finished.Secrets.State != "" {
		t.Fatal("Update must record the outcome without restoring secrets")
	}
	missing := f.flow("flow-missing")
	if _, err := f.store.Update(ctx, owner(), missing); !errors.Is(err, oauthflow.ErrFlowNotFound) {
		t.Fatalf("Update of an unknown flow: err=%v, want ErrFlowNotFound", err)
	}
}

func testResolveState(t *testing.T, f *fixture) {
	ctx := context.Background()
	created := f.create(t, "flow-state")
	caller, id, err := f.store.ResolveState(ctx, created.Secrets.State)
	if err != nil || caller != owner() || id != "flow-state" {
		t.Fatalf("ResolveState: caller=%+v id=%q err=%v", caller, id, err)
	}
	for _, state := range []string{"", "fixture-state-unknown", created.Secrets.Verifier, "flow-state"} {
		if _, _, err := f.store.ResolveState(ctx, state); !errors.Is(err, oauthflow.ErrFlowNotFound) {
			t.Fatalf("ResolveState(%q): err=%v, want ErrFlowNotFound", state, err)
		}
	}
	device := f.flow("flow-device")
	device.Method, device.Secrets.State = oauthflow.MethodDevice, ""
	if err := f.store.Create(ctx, device); err != nil {
		t.Fatal(err)
	}
	if _, _, err := f.store.ResolveState(ctx, ""); !errors.Is(err, oauthflow.ErrFlowNotFound) {
		t.Fatalf("a flow without state resolved: err=%v", err)
	}
	reused := f.flow("flow-reused-state")
	reused.Secrets.State = created.Secrets.State
	if err := f.store.Create(ctx, reused); !errors.Is(err, oauthflow.ErrFlowExists) {
		t.Fatalf("Create reusing another flow's state: err=%v, want ErrFlowExists", err)
	}
}
