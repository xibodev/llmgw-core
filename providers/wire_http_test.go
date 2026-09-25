package providers

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	core "github.com/xibodev/llmgw-core"
)

type partialErrorReader struct {
	delivered bool
}

func (reader *partialErrorReader) Read(buffer []byte) (int, error) {
	if reader.delivered {
		return 0, io.ErrUnexpectedEOF
	}
	reader.delivered = true
	return copy(buffer, []byte(`{"partial":`)), nil
}

func (reader *partialErrorReader) Close() error { return nil }

type zeroReader struct{}

func (zeroReader) Read(buffer []byte) (int, error) {
	clear(buffer)
	return len(buffer), nil
}

// Ported from the gateway's httpstream_test.go.
func TestInvocationResponseBodyReadFailureClassification(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name      string
		status    int
		wantError bool
	}{
		{name: "successful response", status: http.StatusOK, wantError: true},
		{name: "error response", status: http.StatusBadRequest},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			response := &http.Response{StatusCode: testCase.status, Body: &partialErrorReader{}}
			raw, err := readInvocationResponseBody(context.Background(), response, "Fixture")
			if testCase.wantError {
				var failure *core.ProviderError
				if !InvocationRetryable(err) || !errors.As(err, &failure) || failure.Class != core.ProviderErrorTransport ||
					!errors.Is(err, io.ErrUnexpectedEOF) {
					t.Fatalf("error=%v retryable=%v", err, InvocationRetryable(err))
				}
				return
			}
			if err != nil || len(raw) != 0 {
				t.Fatalf("raw=%q error=%v", raw, err)
			}
		})
	}
}

// Ported from the gateway's httpstream_test.go. Not parallel: each case
// reads the whole bound.
func TestInvocationResponseBodySizeLimit(t *testing.T) {
	for _, testCase := range []struct {
		name      string
		status    int
		wantError bool
	}{
		{name: "successful response", status: http.StatusOK, wantError: true},
		{name: "error response", status: http.StatusBadRequest},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			body := io.NopCloser(io.LimitReader(zeroReader{}, inferenceMaxResponseBytes+1))
			raw, err := readInvocationResponseBody(context.Background(), &http.Response{StatusCode: testCase.status, Body: body}, "Fixture")
			if testCase.wantError {
				if err == nil || !InvocationCircuitFailure(err) || InvocationRetryable(err) ||
					err.Error() != "the Fixture response exceeds the size limit" {
					t.Fatalf("error=%v circuit=%v retry=%v", err, InvocationCircuitFailure(err), InvocationRetryable(err))
				}
				return
			}
			if err != nil || len(raw) != 0 {
				t.Fatalf("raw=%d error=%v", len(raw), err)
			}
		})
	}
	body := io.NopCloser(io.LimitReader(zeroReader{}, inferenceMaxResponseBytes))
	if raw, err := readInvocationResponseBody(context.Background(), &http.Response{StatusCode: http.StatusOK, Body: body}, "Fixture"); err != nil || len(raw) != inferenceMaxResponseBytes {
		t.Fatalf("a response at the bound: %d bytes, err=%v", len(raw), err)
	}
}

func TestInvocationResponseBodyOfACallerWhoGaveUpPermitsNothing(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := readInvocationResponseBody(ctx, &http.Response{StatusCode: http.StatusOK, Body: &partialErrorReader{}}, "Fixture")
	if err == nil || core.ClassifyError(err) != (core.ProviderErrorClassification{}) {
		t.Fatalf("error=%v classification=%+v, want nothing permitted", err, core.ClassifyError(err))
	}
}

// Ported from the gateway's error_classification_test.go.
func TestRetryAfterDelay(t *testing.T) {
	t.Parallel()
	now := time.Unix(1_900_000_000, 0).UTC()
	for value, want := range map[string]time.Duration{
		"":           0,
		"7":          7 * time.Second,
		" 120 ":      2 * time.Minute,
		"0":          0,
		"-5":         0,
		"1.5":        0,
		"soon":       0,
		"9223372036": 9223372036 * time.Second,
		"9223372037": 0,
		now.Add(90 * time.Second).Format(http.TimeFormat): 90 * time.Second,
		now.Add(time.Hour).Format(time.RFC850):            time.Hour,
		now.Format(http.TimeFormat):                       0,
		now.Add(-time.Minute).Format(http.TimeFormat):     0,
	} {
		if got := retryAfterDelay(value, now); got != want {
			t.Errorf("retryAfterDelay(%q) = %v, want %v", value, got, want)
		}
	}
}

func TestStatusClassificationIsTheGatewaySet(t *testing.T) {
	t.Parallel()
	for _, status := range []int{400, 401, 403, 404, 408, 409, 422, 429, 500, 501, 502, 503, 504, 505, 529} {
		classification := statusClassification(status, time.Second)
		transient := status == 408 || status == 429 || status == 500 || status == 502 || status == 503 || status == 504
		want := core.ProviderErrorClassification{
			StatusCode: status, Retryable: transient, FailoverEligible: transient, CircuitFailure: transient, RetryAfter: time.Second,
		}
		if classification != want {
			t.Errorf("status %d = %+v, want %+v", status, classification, want)
		}
	}
}

func TestExtractError(t *testing.T) {
	t.Parallel()
	for body, want := range map[string]string{
		`{"error":{"message":"model overloaded","type":"server_error"}}`: "model overloaded",
		`{"error":{"message":"","type":"server_error"}}`:                 `{"error":{"message":"","type":"server_error"}}`,
		`{"error":"quota exhausted"}`:                                    "quota exhausted",
		`{"detail":"not found"}`:                                         `{"detail":"not found"}`,
		"  upstream unavailable \n":                                      "upstream unavailable",
		"":                                                               "no body",
	} {
		if got := extractError([]byte(body)); got != want {
			t.Errorf("extractError(%q) = %q, want %q", body, got, want)
		}
	}
}

// Ported from the gateway's TestHTTPInvocationErrorExtractsSanitizesAndKeepsStatus.
// The upstream's text stays out of the message and reaches the cause only
// redacted.
func TestHTTPStatusFailureExtractsSanitizesAndKeepsStatus(t *testing.T) {
	t.Parallel()
	token := syntheticGatewayToken()
	response := &http.Response{StatusCode: http.StatusServiceUnavailable, Header: http.Header{"Retry-After": {"7"}}}
	err := httpStatusFailure("embeddings", response, []byte(
		`{"error":{"message":"token `+token+` belongs to owner@example.test","code":"overloaded"}}`,
	), time.Now())
	if classification := core.ClassifyError(err); classification.StatusCode != http.StatusServiceUnavailable ||
		!classification.Retryable || classification.RetryAfter != 7*time.Second || err.Class != core.ProviderErrorUpstream {
		t.Fatalf("classification=%+v class=%s", classification, err.Class)
	}
	if err.Error() != "embeddings returned HTTP 503 (code=overloaded)" {
		t.Fatalf("message = %q", err.Error())
	}
	var cause *InvocationError
	if !errors.As(err, &cause) || cause.Status != http.StatusServiceUnavailable || cause.RetryAfter != 7*time.Second {
		t.Fatalf("cause = %#v", cause)
	}
	if strings.Contains(cause.Msg, token) || strings.Contains(cause.Msg, "owner@example.test") {
		t.Fatalf("the diagnostic exposed sensitive data: %q", cause.Msg)
	}
	if !strings.Contains(cause.Msg, "embeddings: upstream returned 503") || !strings.Contains(cause.Msg, redactedDiagnostic) {
		t.Fatalf("the diagnostic lost safe context: %q", cause.Msg)
	}
}

// Ported from the gateway's TestDiagnosticErrorTextIsSanitizedAndBounded.
func TestHTTPStatusFailureDiagnosticIsSanitizedAndBounded(t *testing.T) {
	t.Parallel()
	raw := strings.Repeat("safe ", 500) + "owner@example.test " + syntheticGatewayToken()
	err := httpStatusFailure("Fixture", &http.Response{StatusCode: http.StatusBadGateway}, []byte(raw), time.Now())
	var cause *InvocationError
	if !errors.As(err, &cause) || utf8.RuneCountInString(cause.Msg) > diagnosticErrorLimit {
		t.Fatalf("diagnostic has %d characters", utf8.RuneCountInString(cause.Msg))
	}
	for _, text := range []string{err.Error(), cause.Msg} {
		if strings.Contains(text, "owner@example.test") || strings.Contains(text, "llmgw_") {
			t.Fatalf("error exposed diagnostic data: %q", text)
		}
	}
}

// Ported from the gateway's TestProviderNon2xxErrorsAreSanitizedAndKeepStatus
// and TestProviderErrorSanitizesCredentialAcrossExtractionBoundary, through
// the kit a vertical calls: read the refused answer, then report it.
func TestRefusedAnswersAreSanitizedAndKeepTheirStatus(t *testing.T) {
	t.Parallel()
	gatewayToken := syntheticGatewayToken()
	gskToken := "gsk_" + strings.Repeat("z", 24)
	email := "provider-owner@example.test"
	prefix := `{"detail":"`
	boundary := prefix + strings.Repeat(" ", 500-len(prefix)) + gatewayToken + strings.Repeat(" safe", 600) + `"}`
	if start := strings.Index(boundary, gatewayToken); start >= 512 || start+len(gatewayToken) <= 512 {
		t.Fatalf("fixture secret range %d..%d does not cross byte 512", start, start+len(gatewayToken))
	}
	for name, check := range map[string]struct {
		status int
		body   string
	}{
		"error message": {http.StatusTooManyRequests, fmt.Sprintf(`{"error":{"status":"FAILED_PRECONDITION","message":"account %s used %s with Bearer %s"}}`, email, gatewayToken, gskToken)},
		"long detail":   {http.StatusServiceUnavailable, boundary},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(check.status)
				_, _ = io.WriteString(w, check.body)
			}))
			defer server.Close()
			response, err := server.Client().Get(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			raw, err := readInvocationResponseBody(context.Background(), response, "Fixture")
			if err != nil {
				t.Fatal(err)
			}
			failure := httpStatusFailure("Fixture", response, raw, time.Now())
			var cause *InvocationError
			if core.ClassifyError(failure).StatusCode != check.status || !errors.As(failure, &cause) {
				t.Fatalf("classification = %+v", core.ClassifyError(failure))
			}
			for _, text := range []string{failure.Error(), cause.Msg} {
				for _, sensitive := range []string{gatewayToken, gatewayToken[:12], gskToken, email} {
					if strings.Contains(text, sensitive) {
						t.Fatalf("error exposed %q: %q", sensitive, text)
					}
				}
			}
			if utf8.RuneCountInString(cause.Msg) > diagnosticErrorLimit || !strings.Contains(cause.Msg, redactedDiagnostic) {
				t.Fatalf("diagnostic has %d characters or no redaction marker: %q", utf8.RuneCountInString(cause.Msg), cause.Msg)
			}
		})
	}
}

func TestTransportAndUnusableFailures(t *testing.T) {
	t.Parallel()
	cause := errors.New("connection reset")
	failure := transportFailure(context.Background(), "Fixture could not be reached", cause)
	if failure.Class != core.ProviderErrorTransport || !errors.Is(failure, cause) || failure.Error() != "Fixture could not be reached" ||
		failure.Classification != (core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}) {
		t.Fatalf("transport failure = %+v", failure)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if gaveUp := transportFailure(ctx, "Fixture could not be reached", ctx.Err()); gaveUp.Classification != (core.ProviderErrorClassification{}) {
		t.Fatalf("a caller who gave up permits %+v", gaveUp.Classification)
	}
	unusable := unusableResponse("the Fixture response is not JSON", cause)
	if unusable.Class != core.ProviderErrorUpstream || !errors.Is(unusable, cause) ||
		unusable.Classification != (core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}) {
		t.Fatalf("unusable response = %+v", unusable)
	}
}
