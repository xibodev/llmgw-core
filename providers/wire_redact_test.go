package providers

import (
	"bytes"
	"encoding/base64"
	"strings"
	"testing"
	"unicode/utf8"
)

func syntheticGatewayToken() string {
	return "llmgw_" + base64.RawURLEncoding.EncodeToString(bytes.Repeat([]byte{0xab}, 24))
}

// Ported from the gateway's error_redaction_test.go.
func TestSanitizeDiagnosticText(t *testing.T) {
	t.Parallel()
	gatewayToken := syntheticGatewayToken()
	pem := "-----BEGIN PRIVATE KEY-----\nZmFrZSBrZXkgbWF0ZXJpYWw=\n-----END PRIVATE KEY-----"
	cases := []struct {
		name   string
		secret string
		text   string
	}{
		{name: "email", secret: "owner@example.test", text: "account owner@example.test failed"},
		{name: "GitHub token", secret: "gho_" + strings.Repeat("a", 16), text: "token gho_" + strings.Repeat("a", 16)},
		{name: "OpenAI token", secret: "sk-" + strings.Repeat("b", 24), text: "credential sk-" + strings.Repeat("b", 24)},
		{name: "gateway token", secret: gatewayToken, text: "gateway rejected " + gatewayToken},
		{name: "gsk token", secret: "gsk_" + strings.Repeat("c", 12), text: "provider rejected gsk_" + strings.Repeat("c", 12)},
		{name: "Bearer value", secret: "small!bearer/value", text: "Authorization: Bearer small!bearer/value"},
		{name: "query assignment", secret: "short-value", text: "https://example.test/callback?api_key=short-value&mode=test"},
		{name: "plain key query assignment", secret: "small-key", text: "https://example.test/callback?key=small-key&mode=test"},
		{name: "quoted key assignment", secret: "small-json-key", text: `{"key": "small-json-key"}`},
		{name: "key value assignment", secret: "small-pass", text: `password: "small-pass"`},
		{name: "long hex", secret: strings.Repeat("deadbeef", 5), text: "digest=" + strings.Repeat("deadbeef", 5)},
		{name: "long alphanumeric", secret: "AbCdEf0123456789GhIjKlMn", text: "opaque AbCdEf0123456789GhIjKlMn"},
		{name: "PEM block", secret: pem, text: "parse failed:\n" + pem},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			got := sanitizeDiagnosticText(testCase.text)
			if strings.Contains(got, testCase.secret) {
				t.Fatalf("sensitive value survived sanitization: %q", got)
			}
			if !strings.Contains(got, redactedDiagnostic) {
				t.Fatalf("sanitized text has no redaction marker: %q", got)
			}
			if twice := sanitizeDiagnosticText(got); twice != got {
				t.Fatalf("sanitization is not idempotent: once=%q twice=%q", got, twice)
			}
		})
	}

	ordinary := "request_id model_name abcdefghijklmnopqrstuvwx_identifier ordinary_snake_case service account key: missing fields"
	if got := sanitizeDiagnosticText(ordinary); got != ordinary {
		t.Fatalf("ordinary underscore identifiers changed: %q", got)
	}
}

// Ported from the gateway's error_redaction_test.go.
func TestSanitizeDiagnosticTextLimitSanitizesBeforeTruncating(t *testing.T) {
	t.Parallel()
	secret := syntheticGatewayToken()
	text := strings.Repeat("safe ", 35) + secret
	got := sanitizeDiagnosticTextLimit(text, 200)
	if utf8.RuneCountInString(got) > 200 {
		t.Fatalf("bounded text has %d characters", utf8.RuneCountInString(got))
	}
	if strings.Contains(got, secret[:24]) {
		t.Fatalf("truncated credential prefix survived: %q", got)
	}
	if !strings.Contains(got, redactedDiagnostic) {
		t.Fatalf("credential was not sanitized before truncation: %q", got)
	}
}

// Ported from the gateway diagnostics package's tests: the limit counts
// runes, never splits one, and keeps nothing at zero or below.
func TestSanitizeDiagnosticTextLimitCountsRunes(t *testing.T) {
	t.Parallel()
	secret := "llmgw_" + strings.Repeat("a", 32)
	got := sanitizeDiagnosticTextLimit(strings.Repeat("界", 190)+secret, 200)
	if utf8.RuneCountInString(got) > 200 || !utf8.ValidString(got) {
		t.Fatalf("bounded text has %d runes or is not UTF-8: %q", utf8.RuneCountInString(got), got)
	}
	if strings.Contains(got, secret[:24]) || !strings.Contains(got, redactedDiagnostic) {
		t.Fatalf("credential was not sanitized before limiting: %q", got)
	}
	for _, limit := range []int{0, -1} {
		if got := sanitizeDiagnosticTextLimit("safe", limit); got != "" {
			t.Fatalf("limit %d kept %q", limit, got)
		}
	}
	if got := sanitizeDiagnosticTextLimit("safe", 12); got != "safe" {
		t.Fatalf("text within the limit changed: %q", got)
	}
}
