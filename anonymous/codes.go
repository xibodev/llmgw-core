package anonymous

import (
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// authenticationState is the gateway's reading of whether the provider
// accepted anonymous access: rejected on an authentication or permission
// failure, accepted once it answered, and unknown otherwise.
func authenticationState(health core.ProviderHealthEvidence) string {
	if health.ErrorClass == core.ProviderErrorAuth || health.ErrorClass == core.ProviderErrorForbidden {
		return AuthenticationRejected
	}
	if health.Status == core.ProviderHealthHealthy || health.Status == core.ProviderHealthDegraded {
		return AuthenticationAccepted
	}
	return AuthenticationUnknown
}

// healthFailureCode is the gateway's failure code of a check whose health
// is health: a rejection, or the step that failed.
func healthFailureCode(health core.ProviderHealthEvidence, catalog bool) string {
	if health.ErrorClass == core.ProviderErrorAuth || health.ErrorClass == core.ProviderErrorForbidden {
		return FailureAuthenticationRejected
	}
	if catalog {
		return FailureCatalogFailed
	}
	return FailureVerificationFailed
}

// ProbeFailureCode is the failure code the gateway records for a probe:
// none for a verified probe, else the probe's own code, or
// FailureVerificationFailed when it has none.
func ProbeFailureCode(probe core.CompletionProbeEvidence) string {
	if probe.Status == core.CompletionVerified {
		return ""
	}
	code := strings.TrimSpace(probe.FailureCode)
	if code == "" || code == string(core.ProviderErrorNone) {
		return FailureVerificationFailed
	}
	return code
}
