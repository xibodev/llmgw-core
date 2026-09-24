package oauthflow

import (
	"context"
	"errors"
	"fmt"

	core "github.com/xibodev/llmgw-core"
)

// Poll asks the provider once whether the owner approved caller's device
// flow, and never more often than the flow's interval.
//
//   - Pending returns the pending view. A poll before the interval elapsed,
//     or a provider's slow_down, returns it with ErrSlowDown; slow_down
//     lengthens the interval for the rest of the flow.
//   - Approval consumes the flow, saves the credential and returns the
//     complete view.
//   - Denial or the provider's expiry consumes the flow and returns
//     ErrAccessDenied or ErrFlowExpired with the ended view.
//   - A driver error is transient: the flow stays pending.
//
// A flow that already ended returns its view and no error.
func (s *Service) Poll(ctx context.Context, caller core.Caller, id string) (View, error) {
	flow, err := s.store.Get(ctx, caller, id)
	if errors.Is(err, ErrFlowExpired) {
		return expiredView(id), err
	}
	if err != nil {
		return View{}, err
	}
	if flow.Method != MethodDevice {
		return View{}, ErrWrongMethod
	}
	if flow.Consumed() {
		return viewOf(flow), nil
	}
	driver, err := s.driver(flow.Instance, flow.Method)
	if err != nil {
		return View{}, err
	}
	now := s.now()
	if now.Before(flow.Progress.NextPollAt) {
		return viewOf(flow), ErrSlowDown
	}
	// Claim this poll by moving NextPollAt forward first: of concurrent
	// pollers in any process, only the one whose update lands asks the
	// provider.
	claim := flow
	claim.Progress.NextPollAt = now.Add(flow.Progress.Interval)
	claimed, err := s.store.Update(ctx, caller, claim)
	switch {
	case errors.Is(err, ErrConflict):
		return viewOf(flow), ErrSlowDown
	case errors.Is(err, ErrFlowExpired):
		return expiredView(id), err
	case err != nil:
		return View{}, err
	}
	result, err := driver.(DeviceDriver).Poll(ctx, claimed)
	if err != nil {
		return viewOf(claimed), fmt.Errorf("oauthflow: poll %s: %w", claimed.Instance, err)
	}
	switch result.Status {
	case PollPending:
		return viewOf(claimed), nil
	case PollSlowDown:
		slower := claimed
		slower.Progress.Interval += slowDownStep
		slower.Progress.NextPollAt = now.Add(slower.Progress.Interval)
		if updated, err := s.store.Update(ctx, caller, slower); err == nil {
			claimed = updated
		}
		return viewOf(claimed), ErrSlowDown
	case PollApproved:
		consumed, err := s.store.Consume(ctx, caller, id)
		if err != nil {
			// Another request ended the flow first; its record is discarded.
			return View{}, err
		}
		return s.save(ctx, consumed, result.Record)
	case PollDenied, PollExpired:
		consumed, err := s.store.Consume(ctx, caller, id)
		if err != nil {
			return View{}, err
		}
		if result.Status == PollExpired {
			return s.finish(ctx, consumed, OutcomeExpired, ""), ErrFlowExpired
		}
		return s.finish(ctx, consumed, OutcomeFailed, ""), ErrAccessDenied
	default:
		return viewOf(claimed), fmt.Errorf("oauthflow: poll %s: unknown status %q", claimed.Instance, result.Status)
	}
}
