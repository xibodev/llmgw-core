package providers

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// edgeTTSSkewedSignature is the gateway's signature for the fixture token
// an hour after the fixture clock.
const edgeTTSSkewedSignature = "208A80152F17964E9A4E206F7970062FD5E30F8BCB480D7743DD488A0E23C2BD"

func edgeTTSQuery(t *testing.T, dialed edgeTTSDialed) url.Values {
	t.Helper()
	parsed, err := url.Parse(dialed.url)
	if err != nil {
		t.Fatal(err)
	}
	return parsed.Query()
}

// Ported from the gateway's TestEdgeTTSDialRetriesForbiddenOnceAndClosesResponses.
// The skew is the instance's: it signs its next request with it, and
// another instance does not.
func TestEdgeTTSLearnsTheServiceClockFromARefusedHandshake(t *testing.T) {
	t.Parallel()
	speak := []edgeTTSFrame{edgeTTSAudioFrame("MP3"), edgeTTSTurnEnd()}
	refusal := edgeTTSAnswer{status: http.StatusForbidden, header: http.Header{"Date": {edgeTTSFixtureNow.Add(time.Hour).Format(time.RFC1123)}}}
	service := &edgeTTSService{answers: []edgeTTSAnswer{refusal, {frames: speak}}}
	provider := newFixtureEdgeTTS(t, service)
	credential := &core.Credential{APIKey: edgeTTSFixtureToken}
	for range 2 {
		if audio, err := provider.Synthesize(t.Context(), credential, "", "hello", ""); err != nil || string(audio) != "MP3" {
			t.Fatalf("audio = %q, err = %v", audio, err)
		}
	}
	dials, _, bodies := service.recorded()
	if len(dials) != 3 || bodies[0].count() != 1 || bodies[1].count() != 1 {
		t.Fatalf("dials = %d, body closes = %d, %d", len(dials), bodies[0].count(), bodies[1].count())
	}
	first, retry, next := edgeTTSQuery(t, dials[0]), edgeTTSQuery(t, dials[1]), edgeTTSQuery(t, dials[2])
	if first.Get("Sec-MS-GEC") != edgeTTSFixtureSignature || retry.Get("Sec-MS-GEC") != edgeTTSSkewedSignature || next.Get("Sec-MS-GEC") != edgeTTSSkewedSignature {
		t.Fatalf("signatures = %q, %q, %q", first.Get("Sec-MS-GEC"), retry.Get("Sec-MS-GEC"), next.Get("Sec-MS-GEC"))
	}
	if first.Get("ConnectionId") == retry.Get("ConnectionId") || first.Get("Ocp-Apim-Subscription-Key") != retry.Get("Ocp-Apim-Subscription-Key") ||
		first.Get("Sec-MS-GEC-Version") != retry.Get("Sec-MS-GEC-Version") {
		t.Fatalf("retry query = %v after %v", retry, first)
	}

	other := &edgeTTSService{answers: []edgeTTSAnswer{{frames: speak}}}
	if _, err := newFixtureEdgeTTS(t, other).Synthesize(t.Context(), credential, "", "hello", ""); err != nil {
		t.Fatal(err)
	}
	if dials, _, _ := other.recorded(); edgeTTSQuery(t, dials[0]).Get("Sec-MS-GEC") != edgeTTSFixtureSignature {
		t.Fatalf("another instance signed with %q", edgeTTSQuery(t, dials[0]).Get("Sec-MS-GEC"))
	}

	// A refusal whose Date does not parse teaches nothing, and a second
	// refusal ends the request.
	service = &edgeTTSService{answers: []edgeTTSAnswer{{status: http.StatusForbidden, header: http.Header{"Date": {"yesterday"}}}}}
	provider = newFixtureEdgeTTS(t, service)
	_, err := provider.Synthesize(t.Context(), credential, "", "hello", "")
	var failure *core.ProviderError
	dials, _, bodies = service.recorded()
	if !errors.As(err, &failure) || failure.Classification != (core.ProviderErrorClassification{StatusCode: http.StatusForbidden}) ||
		len(dials) != 2 || bodies[0].count() != 1 || bodies[1].count() != 1 || edgeTTSQuery(t, dials[1]).Get("Sec-MS-GEC") != edgeTTSFixtureSignature {
		t.Fatalf("err = %#v, dials = %d", err, len(dials))
	}
}

// Ported from the gateway's TestEdgeTTSDialHTTPFailuresAreClosedAndSafe and
// TestEdgeTTSDialTransportFailureIsGeneric.
func TestEdgeTTSDialFailuresAreClosedAndSafe(t *testing.T) {
	t.Parallel()
	const subscriptionKey = "llmgw_edge_subscription_secret_123456"
	credential := &core.Credential{APIKey: subscriptionKey}
	base := func(config *EdgeTTSConfig) { config.BaseURL = "https://speech.example.invalid/tts" }
	for _, tc := range []struct {
		status int
		want   core.ProviderErrorClassification
	}{
		{http.StatusUnauthorized, core.ProviderErrorClassification{StatusCode: 401, RetryAfter: 3 * time.Second}},
		{http.StatusTooManyRequests, core.ProviderErrorClassification{StatusCode: 429, Retryable: true, FailoverEligible: true, CircuitFailure: true, RetryAfter: 3 * time.Second}},
	} {
		service := &edgeTTSService{answers: []edgeTTSAnswer{{status: tc.status, header: http.Header{"Retry-After": {"3"}}}}}
		_, err := newFixtureEdgeTTS(t, service, base).Synthesize(t.Context(), credential, "", "hello", "")
		var failure *core.ProviderError
		dials, _, bodies := service.recorded()
		if !errors.As(err, &failure) || failure.Classification != tc.want || len(dials) != 1 || bodies[0].count() != 1 {
			t.Fatalf("HTTP %d: err = %#v, dials = %d", tc.status, err, len(dials))
		}
		query := edgeTTSQuery(t, dials[0])
		for _, secret := range []string{subscriptionKey, query.Get("Sec-MS-GEC"), query.Get("ConnectionId"), "speech.example.invalid", "Ocp-Apim-Subscription-Key", "Sec-MS-GEC", "ConnectionId"} {
			if strings.Contains(err.Error()+errors.Unwrap(err).Error(), secret) {
				t.Fatalf("HTTP %d: the error leaked %q: %q", tc.status, secret, err)
			}
		}
	}

	service := &edgeTTSService{answers: []edgeTTSAnswer{{err: &url.Error{Op: "dial", URL: "wss://speech.example.invalid/tts?Ocp-Apim-Subscription-Key=" + subscriptionKey, Err: context.DeadlineExceeded}}}}
	_, err := newFixtureEdgeTTS(t, service, base).Synthesize(t.Context(), credential, "", "hello", "")
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Message != "Edge TTS could not open its websocket" || failure.Class != core.ProviderErrorTransport ||
		failure.Classification != (core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}) ||
		!errors.Is(err, context.DeadlineExceeded) || strings.Contains(failure.Cause.Error(), subscriptionKey) {
		t.Fatalf("transport failure = %#v", err)
	}
}
