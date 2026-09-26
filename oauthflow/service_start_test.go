package oauthflow_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/oauthflow"
)

func TestStartShowsTheOwnerOnlyPublicFields(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	start := h.clock.Now()

	device := h.start(t, oauthflow.MethodDevice)
	if device.ID == "" || device.Instance != fixtureInstance || device.Method != oauthflow.MethodDevice ||
		device.Status != oauthflow.StatusPending || device.UserCode != "WXYZ-1234" ||
		device.VerificationURI != "https://verify.example.test/device" || device.VerificationURIComplete == "" ||
		device.Interval != 5 || !device.ExpiresAt.Equal(start.Add(15*time.Minute)) || device.AuthorizationURL != "" {
		t.Fatalf("device view=%+v", device)
	}
	for _, method := range []oauthflow.Method{oauthflow.MethodBrowser, oauthflow.MethodManual} {
		view := h.start(t, method, oauthflow.WithRedirectURI("https://app.example.test/oauth/callback"))
		if view.Method != method || view.Status != oauthflow.StatusPending || view.AuthorizationURL == "" ||
			view.UserCode != "" || view.Interval != 0 || !view.ExpiresAt.Equal(start.Add(oauthflow.DefaultTTL)) {
			t.Fatalf("%s view=%+v", method, view)
		}
		got, err := h.service.Get(context.Background(), owner(), view.ID)
		if err != nil || got != view {
			t.Fatalf("Get %s: view=%+v err=%v, want %+v", method, got, err, view)
		}
		assertNoSecrets(t, got)
	}
	if device.ID == h.start(t, oauthflow.MethodDevice).ID {
		t.Fatal("two flows share an id")
	}
}

func TestStartPassesProductInputsToTheDriverOnly(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	params := map[string]string{"connection_name": "work", "client_secret": secretParam}
	view := h.start(t, oauthflow.MethodBrowser,
		oauthflow.WithRedirectURI(" https://app.example.test/oauth/callback "), oauthflow.WithParams(params))
	params["connection_name"] = "mutated"
	request := h.driver.starts[0]
	if request.Caller != owner() || request.Instance != fixtureInstance || request.Method != oauthflow.MethodBrowser ||
		request.RedirectURI != "https://app.example.test/oauth/callback" || request.Params["connection_name"] != "work" {
		t.Fatalf("driver received %+v", request)
	}
	if _, err := h.service.Complete(context.Background(), owner(), view.ID, oauthflow.CompleteInput{Code: fixtureCode}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.credentials.Load(context.Background(), "user-1/fixture-provider/work"); err != nil {
		t.Fatalf("the key hook did not receive the start params: %v", err)
	}
	if flow := h.driver.seen[0]; flow.Secrets.RedirectURI != "https://app.example.test/oauth/callback" {
		t.Fatalf("the flow kept redirect URI %q", flow.Secrets.RedirectURI)
	}
}

func TestStartRejectsWhatItCannotRun(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	if _, err := h.service.Start(ctx, owner(), fixtureInstance, "password"); err == nil {
		t.Fatal("started an unknown method")
	}
	if _, err := h.service.Start(ctx, owner(), "unknown-provider", oauthflow.MethodDevice); err == nil {
		t.Fatal("started a flow on an instance without a driver")
	}
	if _, err := h.service.Start(ctx, core.Caller{Kind: core.CallerHuman}, fixtureInstance, oauthflow.MethodDevice); err == nil {
		t.Fatal("started a flow for a caller without an identity")
	}

	codeOnly := oauthflow.BrowserPKCE{}
	incomplete := startFunc(func(context.Context, oauthflow.StartRequest) (oauthflow.Authorization, error) {
		return oauthflow.Authorization{AuthorizationURL: "https://auth.example.test/authorize"}, nil
	})
	for name, driver := range map[string]oauthflow.Driver{"code driver for device": codeOnly, "no state or verifier": incomplete} {
		service, err := oauthflow.New(oauthflow.Options{
			Store: oauthflow.NewMemoryFlowStore(oauthflow.MemoryFlowStoreOptions{}), Credentials: core.NewMemoryCredentialStore(),
			Drivers:       func(string, oauthflow.Method) (oauthflow.Driver, error) { return driver, nil },
			CredentialKey: func(context.Context, oauthflow.Completion) (string, error) { return "key", nil },
		})
		if err != nil {
			t.Fatal(err)
		}
		method := oauthflow.MethodDevice
		if name == "no state or verifier" {
			method = oauthflow.MethodBrowser
		}
		if _, err := service.Start(ctx, owner(), fixtureInstance, method); err == nil {
			t.Fatalf("%s: Start succeeded", name)
		}
	}
	if _, err := oauthflow.New(oauthflow.Options{}); err == nil {
		t.Fatal("New accepted empty options")
	}
}

// startFunc is a code driver whose Start is a function and that never
// exchanges.
type startFunc func(context.Context, oauthflow.StartRequest) (oauthflow.Authorization, error)

func (f startFunc) Start(ctx context.Context, request oauthflow.StartRequest) (oauthflow.Authorization, error) {
	return f(ctx, request)
}

func (startFunc) Exchange(context.Context, oauthflow.Flow, string) (tokenstore.Record, error) {
	return tokenstore.Record{}, errors.New("unexpected exchange")
}
