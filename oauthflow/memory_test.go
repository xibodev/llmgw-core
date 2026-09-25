package oauthflow_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/oauthflow"
	"github.com/xibodev/llmgw-core/oauthflow/oauthflowtest"
)

func TestMemoryFlowStoreConformance(t *testing.T) {
	oauthflowtest.Run(t, func(now func() time.Time) oauthflow.FlowStore {
		return oauthflow.NewMemoryFlowStore(oauthflow.MemoryFlowStoreOptions{Now: now})
	})
}

func TestMemoryFlowStorePurgesLongExpiredFlows(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Unix(1_800_000_000, 0)
	store := oauthflow.NewMemoryFlowStore(oauthflow.MemoryFlowStoreOptions{Now: func() time.Time { return now }})
	owner := core.Caller{ID: "user-1", Kind: core.CallerHuman}
	flow := oauthflow.Flow{ID: "flow-old", Caller: owner, ExpiresAt: now.Add(time.Minute), Secrets: oauthflow.Secrets{State: "state-old"}}
	if err := store.Create(ctx, flow); err != nil {
		t.Fatal(err)
	}
	now = now.Add(2 * time.Minute)
	if _, err := store.Get(ctx, owner, "flow-old"); !errors.Is(err, oauthflow.ErrFlowExpired) {
		t.Fatalf("recently expired: err=%v, want ErrFlowExpired", err)
	}
	now = now.Add(time.Hour)
	if err := store.Create(ctx, oauthflow.Flow{ID: "flow-new", Caller: owner, ExpiresAt: now.Add(time.Minute)}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Get(ctx, owner, "flow-old"); !errors.Is(err, oauthflow.ErrFlowNotFound) {
		t.Fatalf("long expired: err=%v, want ErrFlowNotFound after the purge", err)
	}
	reuse := oauthflow.Flow{ID: "flow-reuse", Caller: owner, ExpiresAt: now.Add(time.Minute), Secrets: oauthflow.Secrets{State: "state-old"}}
	if err := store.Create(ctx, reuse); err != nil {
		t.Fatalf("the purge kept the old flow's state indexed: %v", err)
	}
}

func TestMemoryFlowStoreValidatesFlows(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := oauthflow.NewMemoryFlowStore(oauthflow.MemoryFlowStoreOptions{})
	valid := oauthflow.Flow{ID: "flow-1", Caller: core.LocalCaller(), ExpiresAt: time.Now().Add(time.Minute)}
	invalid := map[string]oauthflow.Flow{
		"no id":         {Caller: valid.Caller, ExpiresAt: valid.ExpiresAt},
		"no caller id":  {ID: "flow-2", Caller: core.Caller{Kind: core.CallerHuman}, ExpiresAt: valid.ExpiresAt},
		"unknown kind":  {ID: "flow-3", Caller: core.Caller{ID: "user-1", Kind: "robot"}, ExpiresAt: valid.ExpiresAt},
		"no expiry set": {ID: "flow-4", Caller: valid.Caller},
	}
	for name, flow := range invalid {
		if err := store.Create(ctx, flow); err == nil {
			t.Fatalf("%s: Create accepted an invalid flow", name)
		}
	}
	if err := store.Create(ctx, valid); err != nil {
		t.Fatalf("valid flow: %v", err)
	}
}

func TestFlowFormattingOmitsSecrets(t *testing.T) {
	t.Parallel()
	flow := oauthflow.Flow{
		ID: "flow-1", Caller: core.LocalCaller(), Instance: "fixture", Method: oauthflow.MethodBrowser,
		AuthorizationURL: "https://auth.example.test/authorize?state=secret-state",
		Secrets: oauthflow.Secrets{
			Verifier: "secret-verifier", State: "secret-state", DeviceCode: "secret-device",
			RedirectURI: "https://app.example.test/secret-redirect",
			DriverData:  map[string]string{"k": "secret-driver"}, Params: map[string]string{"k": "secret-param"},
		},
	}
	for _, text := range []string{fmt.Sprint(flow), fmt.Sprintf("%v %+v %#v %s", flow, flow, flow, flow), fmt.Sprintf("%+v", &flow)} {
		if strings.Contains(text, "secret-") {
			t.Fatalf("formatted flow leaked a secret: %s", text)
		}
	}
}

func TestNewIDIsUnguessable(t *testing.T) {
	t.Parallel()
	seen := map[string]bool{}
	for range 64 {
		id, err := oauthflow.NewID()
		if err != nil {
			t.Fatal(err)
		}
		if len(id) != 43 || seen[id] {
			t.Fatalf("id %q is short or repeated", id)
		}
		seen[id] = true
	}
}
