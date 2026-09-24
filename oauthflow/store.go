package oauthflow

import (
	"context"

	core "github.com/xibodev/llmgw-core"
)

// FlowStore keeps flows where every process serving the product can reach
// them. Implementations must be safe for concurrent use; the oauthflowtest
// package verifies the guarantees below.
//
// Every operation that names a caller compares the whole core.Caller (ID,
// Kind and ProjectID) with the flow's owner and answers ErrFlowNotFound on a
// mismatch, before it looks at expiry, so another caller learns nothing.
// Returned flows are copies.
type FlowStore interface {
	// Create stores a new flow and assigns its revision. A flow whose id is
	// already stored fails with ErrFlowExists.
	Create(ctx context.Context, flow Flow) error
	// Get returns caller's flow: ErrFlowNotFound for an unknown flow or
	// another caller's, and ErrFlowExpired once it has expired. A consumed
	// flow is still returned, without secrets, until it expires, so its
	// owner can read the outcome.
	Get(ctx context.Context, caller core.Caller, id string) (Flow, error)
	// Consume atomically ends caller's pending flow and returns it with its
	// secrets. Of any number of concurrent calls exactly one succeeds;
	// every later call gets ErrFlowNotFound. An expired flow gives
	// ErrFlowExpired and can never be consumed. The stored flow keeps no
	// secrets afterwards.
	Consume(ctx context.Context, caller core.Caller, id string) (Flow, error)
	// Update stores flow.Progress, and nothing else, if the stored revision
	// still equals flow.Revision, and returns the flow with its new
	// revision; otherwise it fails with ErrConflict.
	//
	// Device polling needs it: a provider's slow_down lengthens the interval
	// for the rest of the flow, and a poller claims the next poll by moving
	// NextPollAt forward, so concurrent polls, from any process, never reach
	// the provider together. It also records the outcome of a consumed flow.
	Update(ctx context.Context, caller core.Caller, flow Flow) (Flow, error)
	// ResolveState returns the owner and id of the unconsumed flow whose
	// OAuth state is state, or ErrFlowNotFound, or ErrFlowExpired.
	//
	// A provider's redirect carries no session, so a browser callback can
	// name no caller. The state stands in for one: it is unguessable and only
	// ever appeared inside the owner's authorization URL. Implementations
	// should index a hash of the state rather than the state itself.
	ResolveState(ctx context.Context, state string) (owner core.Caller, id string, err error)
}
