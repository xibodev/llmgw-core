package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// The gateway's bounds on a complete upstream response: an inference
// answer, and a catalog document.
const (
	inferenceMaxResponseBytes = 64 << 20
	catalogMaxResponseBytes   = 8 << 20
)

// readInvocationResponseBody reads a response body within
// inferenceMaxResponseBytes, as the gateway reads every inference answer,
// and leaves closing it to the caller. A successful answer over the bound
// is unusable, and one that breaks off is a transport failure. A refused
// answer that cannot be read whole has no body instead, so a credential
// cut short before the suffix that identifies it never reaches a
// diagnostic.
func readInvocationResponseBody(ctx context.Context, response *http.Response, label string) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(response.Body, inferenceMaxResponseBytes+1))
	refused := response.StatusCode >= http.StatusBadRequest
	switch {
	case len(raw) > inferenceMaxResponseBytes && refused, err != nil && refused:
		return nil, nil
	case len(raw) > inferenceMaxResponseBytes:
		return nil, unusableResponse("the "+label+" response exceeds the size limit", nil)
	case err != nil:
		return nil, transportFailure(ctx, "the "+label+" response could not be read", err)
	}
	return raw, nil
}

// httpStatusFailure reports an answer the upstream refused, classified by
// statusClassification with the upstream's Retry-After, whatever the
// status. Message names the status and the identifiers the error body
// carries, never its text, so it is safe to show a client. The cause is
// an *InvocationError whose Msg is the gateway's own diagnostic, "<label>:
// upstream returned <status>: <upstream message>", redacted and bounded,
// for a product to log.
func httpStatusFailure(label string, response *http.Response, raw []byte, now time.Time) *core.ProviderError {
	status := response.StatusCode
	message := fmt.Sprintf("%s returned HTTP %d", label, status)
	if diagnostic := codexErrorDiagnostic(raw); diagnostic != "" {
		message += " (" + diagnostic + ")"
	}
	classification := statusClassification(status, retryAfterDelay(response.Header.Get("Retry-After"), now))
	detail := fmt.Sprintf("%s: upstream returned %d: %s", label, status, extractError(raw))
	return &core.ProviderError{
		Message:        message,
		Class:          core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: status}).ErrorClass,
		Classification: classification,
		Cause: &InvocationError{
			Msg: sanitizeDiagnosticTextLimit(detail, diagnosticErrorLimit), Status: status,
			Retryable: classification.Retryable, FailoverEligible: classification.FailoverEligible,
			CircuitFailure: classification.CircuitFailure, RetryAfter: classification.RetryAfter,
		},
	}
}

// statusClassification is how the gateway treats an upstream status. 408,
// 429, 500, 502, 503 and 504 are transient: they permit a retry and
// failover and count against the provider. Any other status, 401 and 403
// included, ends the request. Core's own reading of a status counts every
// 5xx as transient; the gateway's set is kept so a ported vertical routes
// as the gateway did.
func statusClassification(status int, retryAfter time.Duration) core.ProviderErrorClassification {
	transient := false
	switch status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusInternalServerError,
		http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		transient = true
	}
	return core.ProviderErrorClassification{
		StatusCode: status, Retryable: transient, FailoverEligible: transient, CircuitFailure: transient,
		RetryAfter: retryAfter,
	}
}

// extractError pulls a human error message out of a JSON error body,
// falling back to the raw text. It is the gateway's, verbatim; its result
// is upstream text, so it is sanitized before it goes anywhere.
func extractError(body []byte) string {
	var obj map[string]any
	if json.Unmarshal(body, &obj) == nil {
		if err, ok := obj["error"].(map[string]any); ok {
			if msg, ok := err["message"].(string); ok && msg != "" {
				return msg
			}
		}
		if s, ok := obj["error"].(string); ok {
			return s
		}
		return string(body)
	}
	t := strings.TrimSpace(string(body))
	if t == "" {
		return "no body"
	}
	return t
}

// retryAfterDelay reads a Retry-After value, delta-seconds or an
// HTTP-date, as a delay from now. Anything else means no delay: a
// malformed value, a negative count, a date already past, or a count a
// Duration cannot hold.
func retryAfterDelay(value string, now time.Time) time.Duration {
	value = strings.TrimSpace(value)
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil {
		if seconds <= 0 || seconds > int64(math.MaxInt64/time.Second) {
			return 0
		}
		return time.Duration(seconds) * time.Second
	}
	if at, err := http.ParseTime(value); err == nil && at.After(now) {
		return at.Sub(now)
	}
	return 0
}

// transportFailure reports a request that got no complete answer. It may
// repeat, it permits failover and it counts against the provider: a
// client timeout while the caller still waits is a slow upstream, which
// another attempt may get past. Once the caller gave up it permits nothing
// and counts against nothing.
func transportFailure(ctx context.Context, message string, err error) *core.ProviderError {
	failure := &core.ProviderError{
		Message: message, Class: core.ProviderErrorTransport, Cause: err,
		Classification: core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true},
	}
	if ctx.Err() != nil {
		failure.Classification = core.ProviderErrorClassification{}
	}
	return failure
}

// unusableResponse reports an answer the upstream sent that cannot be
// used, such as one over its size limit or not the JSON it should be.
// Another target may still serve the request, and it counts against the
// provider, but repeating it would only get the same answer.
func unusableResponse(message string, cause error) *core.ProviderError {
	return &core.ProviderError{
		Message: message, Class: core.ProviderErrorUpstream, Cause: cause,
		Classification: core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true},
	}
}
