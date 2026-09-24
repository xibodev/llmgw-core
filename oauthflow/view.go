package oauthflow

import "time"

// Status is where a flow stands, as its owner sees it.
type Status string

const (
	// StatusPending means the flow awaits the owner, or its completion is
	// in progress.
	StatusPending Status = "pending"
	// StatusComplete means the credential was saved.
	StatusComplete Status = "complete"
	// StatusFailed means the flow ended without a credential.
	StatusFailed Status = "failed"
	// StatusExpired means the flow expired before it completed.
	StatusExpired Status = "expired"
)

// View is the public projection of a flow, the only one Start, Get, Poll,
// Complete and Callback return. It never holds the PKCE verifier, the device
// code, the redirect URI, driver data, start parameters or any token. The
// OAuth state appears only inside AuthorizationURL, which a pending browser
// or manual flow shows its owner.
type View struct {
	ID       string `json:"id"`
	Instance string `json:"instance"`
	Method   Method `json:"method"`
	Status   Status `json:"status"`

	AuthorizationURL        string `json:"authorization_url,omitempty"`
	UserCode                string `json:"user_code,omitempty"`
	VerificationURI         string `json:"verification_uri,omitempty"`
	VerificationURIComplete string `json:"verification_uri_complete,omitempty"`
	// Interval is the device polling interval in seconds.
	Interval  int       `json:"interval,omitempty"`
	ExpiresAt time.Time `json:"expires_at"`

	// CredentialKey names the saved credential once the flow is complete,
	// so a product can show the connection it created. It is a storage key,
	// not a secret.
	CredentialKey string `json:"credential_key,omitempty"`
}

// viewOf projects flow. Presentation fields are shown only while the flow
// awaits its owner.
func viewOf(flow Flow) View {
	view := View{
		ID: flow.ID, Instance: flow.Instance, Method: flow.Method,
		Status: StatusPending, ExpiresAt: flow.ExpiresAt,
	}
	if flow.Consumed() {
		switch flow.Progress.Outcome {
		case OutcomeComplete:
			view.Status, view.CredentialKey = StatusComplete, flow.Progress.CredentialKey
		case OutcomeFailed:
			view.Status = StatusFailed
		case OutcomeExpired:
			view.Status = StatusExpired
		}
		return view
	}
	view.AuthorizationURL = flow.AuthorizationURL
	view.UserCode = flow.UserCode
	view.VerificationURI = flow.VerificationURI
	view.VerificationURIComplete = flow.VerificationURIComplete
	if flow.Method == MethodDevice {
		view.Interval = int(flow.Progress.Interval / time.Second)
	}
	return view
}

// expiredView is what an owner sees of a flow the store reports expired.
func expiredView(id string) View {
	return View{ID: id, Status: StatusExpired}
}
