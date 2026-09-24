package core_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/xibodev/llm-provider-auth/tokenstore"
	"github.com/xibodev/llm-provider-auth/tokenstore/storetest"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalogtest"
)

func TestMemoryCredentialStoreIsATokenStore(t *testing.T) {
	storetest.Run(t, func(*testing.T) storetest.Opener {
		store := core.NewMemoryCredentialStore()
		return func(*testing.T) tokenstore.Store { return store }
	})
}

func TestMemoryCredentialStoreResolvesOwnThenShared(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCredentialStore()
	user := core.Caller{ID: "user-1", Kind: core.CallerHuman}
	store.BindShared("openai", "system-openai")
	store.Bind(user, " openai ", "user-1-openai")

	cases := map[string]struct {
		caller core.Caller
		want   string
	}{
		"own binding":         {user, "user-1-openai"},
		"another user":        {core.Caller{ID: "user-2", Kind: core.CallerHuman}, "system-openai"},
		"anonymous":           {core.Caller{Kind: core.CallerAnonymous}, "system-openai"},
		"same id, other kind": {core.Caller{ID: "user-1", Kind: core.CallerService}, "system-openai"},
		"local user":          {core.LocalCaller(), "system-openai"},
	}
	for name, tc := range cases {
		if key, err := store.Resolve(ctx, tc.caller, "openai"); err != nil || key != tc.want {
			t.Fatalf("%s: key=%q err=%v, want %q", name, key, err, tc.want)
		}
	}
	if _, err := store.Resolve(ctx, user, "anthropic"); !errors.Is(err, core.ErrNoCredential) {
		t.Fatalf("unbound instance: err=%v, want ErrNoCredential", err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := store.Resolve(canceled, user, "openai"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled resolve: err=%v", err)
	}
}

func TestAPIKeyRecordsNeverRefresh(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(ctx, "system-openai", core.APIKeyRecord("static-api-key")); err != nil {
		t.Fatal(err)
	}
	coordinator, err := tokenstore.NewCoordinator(store, func(context.Context, tokenstore.Record) (tokenstore.Record, error) {
		t.Error("an API key record was refreshed")
		return tokenstore.Record{}, errors.New("unexpected refresh")
	})
	if err != nil {
		t.Fatal(err)
	}
	record, err := coordinator.Token(ctx, "system-openai")
	if err != nil {
		t.Fatal(err)
	}
	credential := core.CredentialFromRecord("system-openai", record)
	if credential.APIKey != "static-api-key" || credential.Token != "" || credential.ConnectionID != "system-openai" {
		t.Fatalf("API key credential=%+v", credential)
	}
	oauth := core.CredentialFromRecord("user-1-codex", tokenstore.Record{AccessToken: "access-token", TokenType: "Bearer"})
	if oauth.Token != "access-token" || oauth.APIKey != "" || oauth.ConnectionID != "user-1-codex" {
		t.Fatalf("OAuth credential=%+v", oauth)
	}
}

func TestMemoryCatalogStoreConformance(t *testing.T) {
	catalogtest.Run(t, func(*testing.T) catalogtest.Opener {
		store := core.NewMemoryCatalogStore()
		return func(*testing.T) core.CatalogStore { return store }
	})
}

func TestMemoryCatalogStoreNormalizesKeysAndHonoursCancellation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCatalogStore()
	revision, err := store.Save(ctx, core.CatalogKey{Instance: " provider ", CredentialKey: " user-1-credential "}, core.CatalogEvidence{Status: core.CatalogEmpty})
	if err != nil {
		t.Fatal(err)
	}
	if record, err := store.Load(ctx, core.CatalogKey{Instance: "provider", CredentialKey: "user-1-credential"}); err != nil || record.Revision != revision {
		t.Fatalf("normalized key: record=%+v err=%v", record, err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := store.Load(canceled, core.CatalogKey{Instance: "provider"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled load: err=%v", err)
	}
	if _, err := store.Save(canceled, core.CatalogKey{Instance: "provider"}, core.CatalogEvidence{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled save: err=%v", err)
	}
}

func TestEvidenceSinks(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	var sink core.MemoryEvidenceSink
	var recorder core.EvidenceSink = &sink
	var group sync.WaitGroup
	for range 20 {
		group.Add(1)
		go func() {
			defer group.Done()
			recorder.Record(ctx, core.AccountEvidence{Instance: "openai", Operation: core.EvidenceOperationInvoke})
		}()
	}
	group.Wait()
	records := sink.Records()
	if len(records) != 20 {
		t.Fatalf("recorded %d, want 20", len(records))
	}
	records[0].Instance = "mutated"
	if sink.Records()[0].Instance != "openai" {
		t.Fatal("Records exposed the sink's storage")
	}

	var captured []core.AccountEvidence
	core.EvidenceSinkFunc(func(_ context.Context, evidence core.AccountEvidence) {
		captured = append(captured, evidence)
	}).Record(ctx, core.AccountEvidence{Model: "m", Operation: core.EvidenceOperationListModels})
	if len(captured) != 1 || captured[0].Model != "m" {
		t.Fatalf("captured=%+v", captured)
	}
}
