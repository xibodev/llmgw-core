package oauthflow_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/xibodev/llmgw-core/oauthflow"
)

func TestCompleteSavesTheCredentialAndIsSingleUse(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	manual := h.start(t, oauthflow.MethodManual, oauthflow.WithParams(map[string]string{"connection_name": "personal"}))
	done, err := h.service.Complete(ctx, owner(), manual.ID, oauthflow.CompleteInput{Code: " " + fixtureCode + " "})
	if err != nil || done.Status != oauthflow.StatusComplete || done.CredentialKey != "user-1/fixture-provider/personal" || done.AuthorizationURL != "" {
		t.Fatalf("Complete: view=%+v err=%v", done, err)
	}
	assertNoSecrets(t, done)
	if record, err := h.credentials.Load(ctx, done.CredentialKey); err != nil || record.AccessToken != secretAccess {
		t.Fatalf("saved credential=%s err=%v", record, err)
	}
	_, err = h.service.Complete(ctx, owner(), manual.ID, oauthflow.CompleteInput{Code: fixtureCode})
	wantErr(t, err, oauthflow.ErrFlowNotFound, "replayed Complete")
	if got, err := h.service.Get(ctx, owner(), manual.ID); err != nil || got.Status != oauthflow.StatusComplete {
		t.Fatalf("Get after completion: view=%+v err=%v", got, err)
	}

	browser := h.start(t, oauthflow.MethodBrowser, oauthflow.WithRedirectURI("https://app.example.test/oauth/callback"))
	routed, err := h.service.Complete(ctx, owner(), browser.ID, oauthflow.CompleteInput{Code: fixtureCode, State: stateOf(t, browser)})
	if err != nil || routed.Status != oauthflow.StatusComplete {
		t.Fatalf("Complete of a browser flow with its state: view=%+v err=%v", routed, err)
	}
	if _, exchanges := h.driver.counts(); exchanges != 2 {
		t.Fatalf("exchanges=%d, want 2", exchanges)
	}
}

func TestCompleteConsumesEvenWhenTheExchangeFails(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	view := h.start(t, oauthflow.MethodManual)
	h.driver.exchangeErr = errors.New("fixture provider: invalid_grant")
	failed, err := h.service.Complete(ctx, owner(), view.ID, oauthflow.CompleteInput{Code: fixtureCode})
	if !errors.Is(err, h.driver.exchangeErr) || failed.Status != oauthflow.StatusFailed {
		t.Fatalf("failed exchange: view=%+v err=%v", failed, err)
	}
	h.driver.exchangeErr = nil
	_, err = h.service.Complete(ctx, owner(), view.ID, oauthflow.CompleteInput{Code: fixtureCode})
	wantErr(t, err, oauthflow.ErrFlowNotFound, "Complete after a failed exchange")
	if got, err := h.service.Get(ctx, owner(), view.ID); err != nil || got.Status != oauthflow.StatusFailed {
		t.Fatalf("Get after a failed exchange: view=%+v err=%v", got, err)
	}
	if _, exchanges := h.driver.counts(); exchanges != 1 {
		t.Fatalf("the code was exchanged %d times, want 1", exchanges)
	}
}

// A wrong pasted redirect URL spends nothing, as the gateway's
// TestConsumerManualFlowCreatesProviderOnlyAfterSuccessfulCompletion expects.
func TestCompleteLeavesTheFlowOnAMismatchedState(t *testing.T) {
	t.Parallel()
	for _, method := range []oauthflow.Method{oauthflow.MethodManual, oauthflow.MethodBrowser} {
		h := newHarness(t)
		ctx := context.Background()
		view := h.start(t, method, oauthflow.WithRedirectURI(fixtureRedirect))
		wrong := oauthflow.CompleteInput{Code: fixtureCode, State: secretState + "-other"}
		pending, err := h.service.Complete(ctx, owner(), view.ID, wrong)
		wantErr(t, err, oauthflow.ErrStateMismatch, string(method)+": mismatched state")
		if pending.Status != oauthflow.StatusPending || pending.ID != view.ID {
			t.Fatalf("%s: mismatched state view=%+v", method, pending)
		}
		assertNoSecrets(t, pending)
		if got, err := h.service.Get(ctx, owner(), view.ID); err != nil || got.Status != oauthflow.StatusPending {
			t.Fatalf("%s: a mismatched state spent the flow: view=%+v err=%v", method, got, err)
		}
		if _, exchanges := h.driver.counts(); exchanges != 0 {
			t.Fatalf("%s: a mismatched state reached the exchange", method)
		}
		done, err := h.service.Complete(ctx, owner(), view.ID, oauthflow.CompleteInput{Code: fixtureCode, State: stateOf(t, view)})
		if err != nil || done.Status != oauthflow.StatusComplete {
			t.Fatalf("%s: retry with the right state: view=%+v err=%v", method, done, err)
		}
		if _, exchanges := h.driver.counts(); exchanges != 1 {
			t.Fatalf("%s: exchanges=%d, want 1", method, exchanges)
		}
	}
}

func TestCompleteEndsTheFlowOnADenial(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	view := h.start(t, oauthflow.MethodManual)
	ended, err := h.service.Complete(ctx, owner(), view.ID, oauthflow.CompleteInput{Error: "access_denied"})
	wantErr(t, err, oauthflow.ErrAccessDenied, "provider denial")
	if ended.Status != oauthflow.StatusFailed {
		t.Fatalf("denial view=%+v", ended)
	}
	_, err = h.service.Complete(ctx, owner(), view.ID, oauthflow.CompleteInput{Code: fixtureCode, State: stateOf(t, view)})
	wantErr(t, err, oauthflow.ErrFlowNotFound, "retry after a denial")
	if _, exchanges := h.driver.counts(); exchanges != 0 {
		t.Fatal("a denied flow reached the exchange")
	}
}

func TestCompleteWithoutACodeLeavesTheFlow(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	view := h.start(t, oauthflow.MethodManual)
	_, err := h.service.Complete(ctx, owner(), view.ID, oauthflow.CompleteInput{Code: "  "})
	wantErr(t, err, oauthflow.ErrCodeRequired, "empty input")
	if done, err := h.service.Complete(ctx, owner(), view.ID, oauthflow.CompleteInput{Code: fixtureCode}); err != nil || done.Status != oauthflow.StatusComplete {
		t.Fatalf("Complete after an empty input: view=%+v err=%v", done, err)
	}
}

func TestConcurrentCompletesExchangeTheCodeOnce(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	view := h.start(t, oauthflow.MethodManual)
	var (
		group     sync.WaitGroup
		mu        sync.Mutex
		successes int
		others    []error
	)
	for range 16 {
		group.Add(1)
		go func() {
			defer group.Done()
			_, err := h.service.Complete(context.Background(), owner(), view.ID, oauthflow.CompleteInput{Code: fixtureCode})
			mu.Lock()
			defer mu.Unlock()
			if err == nil {
				successes++
			} else if !errors.Is(err, oauthflow.ErrFlowNotFound) {
				others = append(others, err)
			}
		}()
	}
	group.Wait()
	if _, exchanges := h.driver.counts(); successes != 1 || exchanges != 1 || len(others) != 0 {
		t.Fatalf("successes=%d exchanges=%d other errors=%v, want one of each and none", successes, exchanges, others)
	}
}
