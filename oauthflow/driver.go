package oauthflow

import (
	"context"
	"time"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
)

// Driver runs one provider's side of one method. A product registers
// drivers through Options.Drivers; each wraps a protocol package such as
// llm-provider-auth's browseroauth, codex or copilot.
//
// A device driver also implements DeviceDriver, and a browser or manual
// driver implements CodeDriver. Errors a driver returns reach the product,
// so they must never contain token material.
type Driver interface {
	// Start begins one authorization attempt with the provider.
	Start(ctx context.Context, request StartRequest) (Authorization, error)
}

// DeviceDriver polls a device authorization.
type DeviceDriver interface {
	Driver
	// Poll asks the provider once whether the owner approved flow. An error
	// is transient: the flow stays pending and may be polled again.
	Poll(ctx context.Context, flow Flow) (PollResult, error)
}

// CodeDriver exchanges the authorization code of a browser or manual flow,
// using the flow's verifier, redirect URI and driver data.
type CodeDriver interface {
	Driver
	Exchange(ctx context.Context, flow Flow, code string) (tokenstore.Record, error)
}

// StartRequest is what a driver needs to begin.
type StartRequest struct {
	Caller   core.Caller
	Instance string
	Method   Method
	// RedirectURI is the product's callback for this request, when it has
	// one; both products derive it from the request's origin.
	RedirectURI string
	// Params are the product's start inputs, such as an administrator's
	// OAuth client settings. A driver copies what it needs later into
	// Authorization.Secrets.DriverData.
	Params map[string]string
}

// Authorization is a started attempt: what the owner is shown, and the
// secrets the server keeps.
type Authorization struct {
	// AuthorizationURL is where a browser or manual flow sends the owner.
	AuthorizationURL string
	// UserCode, VerificationURI and VerificationURIComplete are what a
	// device flow shows the owner.
	UserCode                string
	VerificationURI         string
	VerificationURIComplete string
	// Interval is the provider's minimum polling interval.
	Interval time.Duration
	// ExpiresIn is how long the attempt stays valid; zero uses the
	// Service's default.
	ExpiresIn time.Duration
	// Secrets are kept on the server. The Service fills Params itself.
	Secrets Secrets
}

// PollStatus is a device provider's answer to one poll.
type PollStatus string

const (
	// PollPending means the owner has not approved yet.
	PollPending PollStatus = "pending"
	// PollSlowDown means the provider asked for a longer interval.
	PollSlowDown PollStatus = "slow_down"
	// PollApproved means Record holds the new credential.
	PollApproved PollStatus = "approved"
	// PollDenied means the owner or provider refused; the flow ends.
	PollDenied PollStatus = "denied"
	// PollExpired means the provider expired the device code; the flow ends.
	PollExpired PollStatus = "expired"
)

// PollResult is the outcome of one device poll.
type PollResult struct {
	Status PollStatus
	// Record is the credential when Status is PollApproved.
	Record tokenstore.Record
}
