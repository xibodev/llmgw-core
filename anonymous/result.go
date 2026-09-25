package anonymous

import (
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

// Statuses of a result.
const (
	// StatusManaged is Enroll's status for a provider the automation
	// manages, which a run goes on to check.
	StatusManaged = "managed"
	// StatusPassed is a check in which a probe verified inference, and
	// StatusFailed any other check.
	StatusPassed = "passed"
	StatusFailed = "failed"
)

// OperationVerify is the operation of every check.
const OperationVerify = "verify"

// Failure codes of a check, the gateway's.
const (
	FailureEvidenceUnavailable    = "evidence_unavailable"
	FailureAuthenticationRejected = "authentication_rejected"
	FailureCatalogFailed          = "catalog_failed"
	FailureVerificationFailed     = "verification_failed"
	FailureModelUnavailable       = "model_unavailable"
)

// Authentication states of a check: whether the provider accepted
// anonymous access.
const (
	AuthenticationAccepted = "accepted"
	AuthenticationRejected = "rejected"
	AuthenticationUnknown  = "unknown"
)

// Result is one provider's outcome in a run, with the fields of the
// gateway's automation report. A provider the automation leaves alone has
// only its ids and the status Enroll gave it.
type Result struct {
	ProviderID string `json:"provider_id"`
	RegistryID string `json:"registry_id,omitempty"`
	// Status is StatusPassed or StatusFailed for a check.
	Status    string `json:"status"`
	Operation string `json:"operation,omitempty"`
	Success   bool   `json:"success"`
	// FailureCode says why a check failed, as one of the Failure codes.
	FailureCode string `json:"failure_code,omitempty"`
	// Details says why no probe ran: the connect error, whose message
	// names the failed operation and never quotes the upstream, or that no
	// model was available.
	Details string `json:"details,omitempty"`
	// VerificationError is set when every probe failed.
	VerificationError   string                     `json:"verification_error,omitempty"`
	AuthenticationState string                     `json:"authentication_state,omitempty"`
	CatalogEvidence     core.CatalogEvidenceStatus `json:"catalog_evidence,omitempty"`
	CompletionEvidence  core.CompletionProbeStatus `json:"completion_evidence,omitempty"`
	// Targets are the published targets and Published counts them. Probed
	// counts the probes, and Verified and Failed their outcomes.
	Targets   []core.Target `json:"targets,omitempty"`
	Published int           `json:"published,omitempty"`
	Probed    int           `json:"probed,omitempty"`
	Verified  int           `json:"verified,omitempty"`
	Failed    int           `json:"failed,omitempty"`
	// Retryable and RetryAfter are the provider health's once probes ran.
	Retryable  bool          `json:"retryable,omitempty"`
	RetryAfter time.Duration `json:"retry_after,omitempty"`
	// Connect is the connect outcome the result derives from.
	Connect core.ProviderConnectResult `json:"-"`
	// RecordErr is why the result was not kept: Record's error, or the
	// context's when the run stopped during the check.
	RecordErr error `json:"-"`
}

// resultOf derives a check's result from its connect outcome, as the
// gateway's automation does.
func resultOf(profile providers.AnonymousProviderProfile, providerID string, connected core.ProviderConnectResult, err error) Result {
	result := Result{
		ProviderID: providerID, RegistryID: profile.RegistryID, Operation: OperationVerify, Status: StatusFailed,
		AuthenticationState: authenticationState(connected.Health), CatalogEvidence: connected.Catalog.Status,
		CompletionEvidence: core.CompletionNotProbed, Connect: connected,
	}
	if err != nil {
		result.FailureCode, result.Details = healthFailureCode(connected.Health, true), err.Error()
		return result
	}
	if len(connected.Probes) == 0 {
		result.FailureCode = FailureModelUnavailable
		result.Details = "No reviewed anonymous model was available for verification."
		return result
	}
	result.Targets, result.Published, result.Probed = connected.Targets, len(connected.Targets), len(connected.Probes)
	for _, probe := range connected.Probes {
		if probe.Status == core.CompletionVerified {
			result.Verified++
		} else {
			result.Failed++
		}
	}
	result.CompletionEvidence = core.CompletionFailed
	if result.Failed == 0 {
		result.CompletionEvidence = core.CompletionVerified
	}
	if result.Verified > 0 {
		result.Success, result.Status, result.AuthenticationState = true, StatusPassed, AuthenticationAccepted
	} else {
		result.FailureCode = healthFailureCode(connected.Health, false)
		result.VerificationError = "Provider inference verification failed."
	}
	result.Retryable, result.RetryAfter = connected.Health.Retryable, connected.Health.RetryAfter
	return result
}
