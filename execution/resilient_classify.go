package execution

import (
	"net/http"

	core "github.com/xibodev/llmgw-core"
)

// TransientClassification reads err as the gateway's resilience wrapper
// does, for a Policy that repeats and breaks circuits as the gateway did.
// It is opt-in: core's own reading, which the executors and the default
// Observe use, stays as it is.
//
// It is err's own classification, with the gateway's reading of an
// upstream status. Of the statuses that refuse a request, only 408, 429,
// 500, 502, 503 and 504 are transient, so only they repeat and count
// against the circuit; any other ends the streak. Core's own reading
// repeats every 5xx, and counts neither 408 nor 429.
//
// Every failure that repeats counts against the circuit, a transport
// failure included. A failure without a status, an answer the upstream
// sent that could not be used, which carries a 2xx, and an operation its
// caller gave up on keep their own reading. Failover and RetryAfter stay
// err's own.
func TransientClassification(err error) core.ProviderErrorClassification {
	classification := core.ClassifyError(err)
	status := classification.StatusCode
	if status != 0 && (status < 200 || status > 299) && !canceled(err) {
		transient := transientStatus(status)
		classification.Retryable, classification.CircuitFailure = transient, transient
	}
	classification.CircuitFailure = classification.CircuitFailure || classification.Retryable
	return classification
}

// transientStatus reports an upstream status the gateway treats as
// transient: 408, 429, 500, 502, 503 or 504.
func transientStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusInternalServerError,
		http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	}
	return false
}
