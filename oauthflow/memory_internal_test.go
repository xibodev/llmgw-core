package oauthflow

import (
	"context"
	"errors"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// TestMemoryFlowStoreWipesExpiredSecrets looks inside the store: an expired
// flow can never be consumed, so nothing may keep its verifier or state.
func TestMemoryFlowStoreWipesExpiredSecrets(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Unix(1_800_000_000, 0)
	store := NewMemoryFlowStore(MemoryFlowStoreOptions{Now: func() time.Time { return now }, ExpiredRetention: time.Hour})
	owner := core.Caller{ID: "user-1", Kind: core.CallerHuman}
	for _, id := range []string{"looked-up", "scanned"} {
		flow := Flow{
			ID: id, Caller: owner, ExpiresAt: now.Add(time.Minute),
			Secrets: Secrets{
				Verifier: "verifier-" + id, State: "state-" + id, DeviceCode: "device-" + id,
				RedirectURI: "https://app.example.test/oauth/callback",
				DriverData:  map[string]string{"k": "v"}, Params: map[string]string{"k": "v"},
			},
		}
		if err := store.Create(ctx, flow); err != nil {
			t.Fatal(err)
		}
	}
	now = now.Add(2 * time.Minute)

	if _, err := store.Get(ctx, owner, "looked-up"); !errors.Is(err, ErrFlowExpired) {
		t.Fatalf("Get: err=%v, want ErrFlowExpired", err)
	}
	assertWiped(t, store, "looked-up")
	if err := store.Create(ctx, Flow{ID: "trigger", Caller: owner, ExpiresAt: now.Add(time.Minute)}); err != nil {
		t.Fatal(err)
	}
	assertWiped(t, store, "scanned")

	// Only the state's hash stays indexed, so a late callback still learns
	// the flow expired, until the record itself is purged.
	if _, _, err := store.ResolveState(ctx, "state-looked-up"); !errors.Is(err, ErrFlowExpired) {
		t.Fatalf("ResolveState after the wipe: err=%v, want ErrFlowExpired", err)
	}
	now = now.Add(time.Hour)
	if _, _, err := store.ResolveState(ctx, "state-scanned"); !errors.Is(err, ErrFlowNotFound) {
		t.Fatalf("ResolveState after the retention: err=%v, want ErrFlowNotFound", err)
	}
	if err := store.Create(ctx, Flow{ID: "trigger-2", Caller: owner, ExpiresAt: now.Add(time.Minute)}); err != nil {
		t.Fatal(err)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if len(store.states) != 0 || store.flows["looked-up"] != nil || store.flows["scanned"] != nil {
		t.Fatalf("purged flows left %d state index entries or their records", len(store.states))
	}
}

func assertWiped(t *testing.T, store *MemoryFlowStore, id string) {
	t.Helper()
	store.mu.Lock()
	defer store.mu.Unlock()
	entry := store.flows[id]
	if entry == nil {
		t.Fatalf("flow %s was dropped before its retention ended", id)
	}
	secrets := entry.flow.Secrets
	if secrets.Verifier != "" || secrets.State != "" || secrets.DeviceCode != "" || secrets.RedirectURI != "" ||
		secrets.DriverData != nil || secrets.Params != nil {
		t.Fatalf("expired flow %s kept its secrets", id)
	}
	if !entry.indexed || store.states[entry.stateKey] != id {
		t.Fatalf("expired flow %s lost its state index entry early", id)
	}
}
