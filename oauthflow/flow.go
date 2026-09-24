package oauthflow

import (
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"time"

	core "github.com/xibodev/llmgw-core"
)

var (
	// ErrFlowNotFound reports a flow that does not exist, was already
	// consumed, or belongs to another caller. The three are deliberately
	// indistinguishable, so a flow id never reveals that someone else's flow
	// exists.
	ErrFlowNotFound = errors.New("oauthflow: flow not found")
	// ErrFlowExpired reports a flow past its expiry. Only its owner sees it.
	ErrFlowExpired = errors.New("oauthflow: flow expired")
	// ErrFlowExists reports a Create whose id is already stored.
	ErrFlowExists = errors.New("oauthflow: flow id already exists")
	// ErrConflict reports an Update whose revision is no longer current.
	ErrConflict = errors.New("oauthflow: flow changed concurrently")
)

// Method is how the owner authorizes a flow.
type Method string

const (
	// MethodBrowser is an authorization-code grant with PKCE whose redirect
	// the product serves: the provider sends the browser back with the code.
	MethodBrowser Method = "browser"
	// MethodDevice is an RFC 8628 device authorization grant, completed by
	// polling while the owner enters a user code elsewhere.
	MethodDevice Method = "device"
	// MethodManual is an authorization-code grant with PKCE whose redirect
	// the product does not serve, so the owner pastes the code or the whole
	// redirect URL back.
	MethodManual Method = "manual"
)

// Outcome records how a consumed flow ended.
type Outcome string

const (
	// OutcomeComplete means the credential was saved.
	OutcomeComplete Outcome = "complete"
	// OutcomeFailed means the flow ended without a credential.
	OutcomeFailed Outcome = "failed"
)

// Flow is one authorization attempt.
//
// The store assigns Revision on every write and sets ConsumedAt when the flow
// is consumed. Secrets never leave the server; a consumed flow keeps none.
// String, GoString and LogValue omit them, so a logged flow leaks nothing.
type Flow struct {
	ID         string
	Caller     core.Caller
	Instance   string
	Method     Method
	CreatedAt  time.Time
	ExpiresAt  time.Time
	Revision   string
	ConsumedAt time.Time

	// What the owner is shown. The OAuth state appears only inside
	// AuthorizationURL, which only the owner receives.
	AuthorizationURL        string
	UserCode                string
	VerificationURI         string
	VerificationURIComplete string

	// Progress is the only part Update may change.
	Progress Progress

	Secrets Secrets
}

// Progress is what changes after Create: the device polling schedule while a
// flow is pending, and the outcome once it is consumed.
type Progress struct {
	// Interval is the minimum wait between device polls. A provider's
	// slow_down lengthens it.
	Interval time.Duration
	// NextPollAt is the earliest time the provider may be polled again.
	NextPollAt time.Time
	// Outcome is empty until a consumed flow is finished.
	Outcome Outcome
	// CredentialKey is the key the credential was saved under.
	CredentialKey string
}

// Secrets are the server-only values of a flow.
type Secrets struct {
	Verifier    string
	State       string
	DeviceCode  string
	RedirectURI string
	// DriverData is opaque provider-specific state the driver needs later.
	DriverData map[string]string
	// Params are the product's start inputs, which may include client
	// secrets.
	Params map[string]string
}

// Consumed reports whether the flow was consumed.
func (f Flow) Consumed() bool { return !f.ConsumedAt.IsZero() }

// Expired reports whether the flow is past its expiry at now.
func (f Flow) Expired(now time.Time) bool { return !now.Before(f.ExpiresAt) }

// Clone returns a deep copy, so stores and callers never share maps.
func (f Flow) Clone() Flow {
	f.Secrets.DriverData = maps.Clone(f.Secrets.DriverData)
	f.Secrets.Params = maps.Clone(f.Secrets.Params)
	return f
}

// String describes the flow without secrets.
func (f Flow) String() string {
	return fmt.Sprintf("oauthflow.Flow{ID:%q Caller:%+v Instance:%q Method:%q ExpiresAt:%s Consumed:%t Outcome:%q}",
		f.ID, f.Caller, f.Instance, f.Method, f.ExpiresAt.UTC().Format(time.RFC3339), f.Consumed(), f.Progress.Outcome)
}

// GoString keeps %#v from printing secrets.
func (f Flow) GoString() string { return f.String() }

// LogValue keeps structured logging from printing secrets.
func (f Flow) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("id", f.ID),
		slog.String("instance", f.Instance),
		slog.String("method", string(f.Method)),
		slog.Time("expires_at", f.ExpiresAt),
		slog.Bool("consumed", f.Consumed()),
	)
}

// NewID returns an unguessable flow id: 32 bytes from crypto/rand.
func NewID() (string, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("oauthflow: generate flow id: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(raw), nil
}
