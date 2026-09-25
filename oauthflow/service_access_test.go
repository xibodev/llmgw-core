package oauthflow_test

import (
	"context"
	"testing"
	"time"

	"github.com/xibodev/llmgw-core/oauthflow"
)

func TestOtherCallersCannotReachAFlow(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	device := h.start(t, oauthflow.MethodDevice)
	manual := h.start(t, oauthflow.MethodManual)
	h.clock.Advance(5 * time.Second)
	for name, intruder := range intruders() {
		for _, id := range []string{device.ID, manual.ID} {
			if _, err := h.service.Get(ctx, intruder, id); err == nil {
				t.Fatalf("%s: Get succeeded", name)
			}
		}
		_, err := h.service.Poll(ctx, intruder, device.ID)
		wantErr(t, err, oauthflow.ErrFlowNotFound, name+": Poll")
		_, err = h.service.Complete(ctx, intruder, manual.ID, oauthflow.CompleteInput{Code: fixtureCode, State: stateOf(t, manual)})
		wantErr(t, err, oauthflow.ErrFlowNotFound, name+": Complete")
	}
	if polls, exchanges := h.driver.counts(); polls != 0 || exchanges != 0 {
		t.Fatalf("other callers reached the provider: polls=%d exchanges=%d", polls, exchanges)
	}
	if _, err := h.service.Poll(ctx, owner(), device.ID); err != nil {
		t.Fatalf("owner's poll after the attempts: %v", err)
	}
	if done, err := h.service.Complete(ctx, owner(), manual.ID, oauthflow.CompleteInput{Code: fixtureCode}); err != nil || done.Status != oauthflow.StatusComplete {
		t.Fatalf("owner's completion after the attempts: view=%+v err=%v", done, err)
	}
}

func TestExpiredFlowsAreRejectedEverywhere(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	device := h.start(t, oauthflow.MethodDevice)
	browser := h.start(t, oauthflow.MethodBrowser, oauthflow.WithRedirectURI("https://app.example.test/oauth/callback"))
	manual := h.start(t, oauthflow.MethodManual)
	h.clock.Advance(15 * time.Minute)

	for _, id := range []string{device.ID, browser.ID, manual.ID} {
		view, err := h.service.Get(ctx, owner(), id)
		wantErr(t, err, oauthflow.ErrFlowExpired, "Get")
		if view.Status != oauthflow.StatusExpired || view.ID != id {
			t.Fatalf("expired Get view=%+v", view)
		}
	}
	_, err := h.service.Poll(ctx, owner(), device.ID)
	wantErr(t, err, oauthflow.ErrFlowExpired, "Poll")
	_, err = h.service.Complete(ctx, owner(), manual.ID, oauthflow.CompleteInput{Code: fixtureCode})
	wantErr(t, err, oauthflow.ErrFlowExpired, "Complete")
	_, err = h.service.Callback(ctx, oauthflow.CompleteInput{Code: fixtureCode, State: stateOf(t, browser)})
	wantErr(t, err, oauthflow.ErrFlowExpired, "Callback")
	for name, intruder := range intruders() {
		_, err := h.service.Get(ctx, intruder, device.ID)
		wantErr(t, err, oauthflow.ErrFlowNotFound, name+": Get of an expired flow")
	}
	if polls, exchanges := h.driver.counts(); polls != 0 || exchanges != 0 {
		t.Fatalf("expired flows reached the provider: polls=%d exchanges=%d", polls, exchanges)
	}
}

func TestOperationsCheckTheFlowsMethod(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	device := h.start(t, oauthflow.MethodDevice)
	manual := h.start(t, oauthflow.MethodManual)
	_, err := h.service.Poll(ctx, owner(), manual.ID)
	wantErr(t, err, oauthflow.ErrWrongMethod, "Poll of a manual flow")
	_, err = h.service.Complete(ctx, owner(), device.ID, oauthflow.CompleteInput{Code: fixtureCode})
	wantErr(t, err, oauthflow.ErrWrongMethod, "Complete of a device flow")
	_, err = h.service.Callback(ctx, oauthflow.CompleteInput{Code: fixtureCode, State: stateOf(t, manual)})
	wantErr(t, err, oauthflow.ErrWrongMethod, "Callback of a manual flow")
	if got, err := h.service.Get(ctx, owner(), manual.ID); err != nil || got.Status != oauthflow.StatusPending {
		t.Fatalf("a refused operation changed the flow: view=%+v err=%v", got, err)
	}
}
