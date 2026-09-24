package oauthflow_test

import (
	"context"
	"testing"

	"github.com/xibodev/llmgw-core/oauthflow"
)

const fixtureRedirect = "https://app.example.test/oauth/callback"

func TestCallbackFindsTheFlowByItsState(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	view := h.start(t, oauthflow.MethodBrowser, oauthflow.WithRedirectURI(fixtureRedirect),
		oauthflow.WithParams(map[string]string{"connection_name": "work"}))
	state := stateOf(t, view)

	for name, input := range map[string]oauthflow.CompleteInput{
		"no state":      {Code: fixtureCode},
		"unknown state": {Code: fixtureCode, State: secretState + "-unknown"},
		"wrong path":    {Code: fixtureCode, State: state, RedirectURI: "https://app.example.test/oauth/callback/other"},
	} {
		_, err := h.service.Callback(ctx, input)
		wantErr(t, err, oauthflow.ErrFlowNotFound, name)
	}
	_, err := h.service.Callback(ctx, oauthflow.CompleteInput{State: state})
	wantErr(t, err, oauthflow.ErrCodeRequired, "callback without a code")
	if got, err := h.service.Get(ctx, owner(), view.ID); err != nil || got.Status != oauthflow.StatusPending {
		t.Fatalf("a refused callback consumed the flow: view=%+v err=%v", got, err)
	}

	done, err := h.service.Callback(ctx, oauthflow.CompleteInput{Code: fixtureCode, State: state, RedirectURI: fixtureRedirect})
	if err != nil || done.Status != oauthflow.StatusComplete || done.CredentialKey != "user-1/fixture-provider/work" {
		t.Fatalf("Callback: view=%+v err=%v", done, err)
	}
	assertNoSecrets(t, done)
	if record, err := h.credentials.Load(ctx, done.CredentialKey); err != nil || record.AccessToken != secretAccess {
		t.Fatalf("saved credential=%s err=%v", record, err)
	}
	_, err = h.service.Callback(ctx, oauthflow.CompleteInput{Code: fixtureCode, State: state})
	wantErr(t, err, oauthflow.ErrFlowNotFound, "replayed callback")
	if got, err := h.service.Get(ctx, owner(), view.ID); err != nil || got.Status != oauthflow.StatusComplete {
		t.Fatalf("the owner's Get after the callback: view=%+v err=%v", got, err)
	}
	if _, exchanges := h.driver.counts(); exchanges != 1 {
		t.Fatalf("exchanges=%d, want 1", exchanges)
	}
}

func TestCallbackWithAProviderErrorEndsTheFlow(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	view := h.start(t, oauthflow.MethodBrowser, oauthflow.WithRedirectURI(fixtureRedirect))
	ended, err := h.service.Callback(ctx, oauthflow.CompleteInput{State: stateOf(t, view), Error: "access_denied"})
	wantErr(t, err, oauthflow.ErrAccessDenied, "denied callback")
	if ended.Status != oauthflow.StatusFailed {
		t.Fatalf("denied callback view=%+v", ended)
	}
	if got, err := h.service.Get(ctx, owner(), view.ID); err != nil || got.Status != oauthflow.StatusFailed {
		t.Fatalf("Get after a denied callback: view=%+v err=%v", got, err)
	}
}
