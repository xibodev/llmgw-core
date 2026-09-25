package oauthflow_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/xibodev/llmgw-core/oauthflow"
)

func TestPollHonoursTheIntervalAndSlowDown(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	view := h.start(t, oauthflow.MethodDevice)

	early, err := h.service.Poll(ctx, owner(), view.ID)
	wantErr(t, err, oauthflow.ErrSlowDown, "poll before the first interval")
	if polls, _ := h.driver.counts(); polls != 0 || early.Status != oauthflow.StatusPending || early.Interval != 5 {
		t.Fatalf("early poll reached the provider %d times: %+v", polls, early)
	}
	h.clock.Advance(5 * time.Second)
	pending, err := h.service.Poll(ctx, owner(), view.ID)
	if err != nil || pending.Status != oauthflow.StatusPending || pending.Interval != 5 {
		t.Fatalf("pending poll: view=%+v err=%v", pending, err)
	}
	if flow := h.driver.seen[0]; flow.Secrets.DeviceCode == "" || flow.Secrets.DriverData["token"] != secretDriver {
		t.Fatal("the driver did not receive the flow's device code and driver data")
	}
	h.driver.queue(status(oauthflow.PollSlowDown))
	h.clock.Advance(5 * time.Second)
	slowed, err := h.service.Poll(ctx, owner(), view.ID)
	wantErr(t, err, oauthflow.ErrSlowDown, "provider slow_down")
	if slowed.Interval != 10 {
		t.Fatalf("slow_down left the interval at %d seconds", slowed.Interval)
	}
	h.clock.Advance(5 * time.Second)
	if _, err := h.service.Poll(ctx, owner(), view.ID); !errors.Is(err, oauthflow.ErrSlowDown) {
		t.Fatalf("a poll after the old interval was allowed: %v", err)
	}
	h.clock.Advance(5 * time.Second)
	if got, err := h.service.Poll(ctx, owner(), view.ID); err != nil || got.Interval != 10 {
		t.Fatalf("poll after the lengthened interval: view=%+v err=%v", got, err)
	}
	if polls, _ := h.driver.counts(); polls != 3 {
		t.Fatalf("provider polled %d times, want 3", polls)
	}
}

func TestPollApprovalSavesTheCredentialOnce(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	view := h.start(t, oauthflow.MethodDevice)
	h.driver.queue(approved())
	h.clock.Advance(5 * time.Second)
	done, err := h.service.Poll(ctx, owner(), view.ID)
	if err != nil || done.Status != oauthflow.StatusComplete || done.CredentialKey != "user-1/fixture-provider/default" {
		t.Fatalf("approved poll: view=%+v err=%v", done, err)
	}
	assertNoSecrets(t, done)
	record, err := h.credentials.Load(ctx, done.CredentialKey)
	if err != nil || record.AccessToken != secretAccess || record.RefreshToken != secretRefresh || record.AccountID != "fixture-account" {
		t.Fatalf("saved credential=%s err=%v", record, err)
	}
	h.clock.Advance(time.Minute)
	again, err := h.service.Poll(ctx, owner(), view.ID)
	if err != nil || again.Status != oauthflow.StatusComplete {
		t.Fatalf("poll after completion: view=%+v err=%v", again, err)
	}
	if got, err := h.service.Get(ctx, owner(), view.ID); err != nil || got.Status != oauthflow.StatusComplete || got.UserCode != "" {
		t.Fatalf("Get after completion: view=%+v err=%v", got, err)
	}
	if polls, _ := h.driver.counts(); polls != 1 {
		t.Fatalf("provider polled %d times after approval", polls)
	}
}

func TestPollDenialAndProviderExpiryEndTheFlow(t *testing.T) {
	t.Parallel()
	cases := map[oauthflow.PollStatus]struct {
		err    error
		status oauthflow.Status
	}{
		oauthflow.PollDenied:  {oauthflow.ErrAccessDenied, oauthflow.StatusFailed},
		oauthflow.PollExpired: {oauthflow.ErrFlowExpired, oauthflow.StatusExpired},
	}
	for result, want := range cases {
		h := newHarness(t)
		ctx := context.Background()
		view := h.start(t, oauthflow.MethodDevice)
		h.driver.queue(status(result))
		h.clock.Advance(5 * time.Second)
		ended, err := h.service.Poll(ctx, owner(), view.ID)
		wantErr(t, err, want.err, string(result))
		if ended.Status != want.status {
			t.Fatalf("%s: view=%+v", result, ended)
		}
		h.clock.Advance(time.Minute)
		if got, err := h.service.Poll(ctx, owner(), view.ID); err != nil || got.Status != want.status {
			t.Fatalf("%s: later poll view=%+v err=%v", result, got, err)
		}
		if polls, _ := h.driver.counts(); polls != 1 {
			t.Fatalf("%s: an ended flow was polled again", result)
		}
	}
}

func TestPollTransientErrorKeepsTheFlow(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	ctx := context.Background()
	view := h.start(t, oauthflow.MethodDevice)
	transient := errors.New("fixture transport failure")
	h.driver.queue(pollStep{err: transient}, approved())
	h.clock.Advance(5 * time.Second)
	failed, err := h.service.Poll(ctx, owner(), view.ID)
	if !errors.Is(err, transient) || failed.Status != oauthflow.StatusPending {
		t.Fatalf("transient failure: view=%+v err=%v", failed, err)
	}
	h.clock.Advance(5 * time.Second)
	if done, err := h.service.Poll(ctx, owner(), view.ID); err != nil || done.Status != oauthflow.StatusComplete {
		t.Fatalf("poll after a transient failure: view=%+v err=%v", done, err)
	}
}

func TestConcurrentPollsReachTheProviderOnce(t *testing.T) {
	t.Parallel()
	h := newHarness(t)
	view := h.start(t, oauthflow.MethodDevice)
	h.clock.Advance(5 * time.Second)
	var group sync.WaitGroup
	for range 16 {
		group.Add(1)
		go func() {
			defer group.Done()
			_, _ = h.service.Poll(context.Background(), owner(), view.ID)
		}()
	}
	group.Wait()
	if polls, _ := h.driver.counts(); polls != 1 {
		t.Fatalf("16 concurrent polls reached the provider %d times, want 1", polls)
	}
}
