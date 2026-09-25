package providers

import "regexp"

// The gateway's redaction of diagnostic text, ported from its diagnostics
// package: credential-shaped values and personal data become
// redactedDiagnostic. Underscores are absent from the generic long-value
// rule, so ordinary identifiers survive.
const (
	redactedDiagnostic   = "[redacted]"
	diagnosticErrorLimit = 2048
	sensitiveKeyPattern  = `(?:access[_-]?token|refresh[_-]?token|id[_-]?token|api[_-]?key|x[_-]?api[_-]?key|x[_-]?goog[_-]?api[_-]?key|client[_-]?secret|private[_-]?key|proxy[_-]?authorization|authorization|password|passwd|credential|cookie|session[_-]?id|session|secret|token|auth)`
)

var (
	pemBlockRE            = regexp.MustCompile(`(?is)-----BEGIN [A-Z0-9][A-Z0-9 -]{0,63}-----.*?-----END [A-Z0-9][A-Z0-9 -]{0,63}-----`)
	sensitiveAssignmentRE = regexp.MustCompile(
		`(?i)(\b` + sensitiveKeyPattern + `\b["']?\s*[:=]\s*)` +
			`(?:"[^"\r\n]*"|'[^'\r\n]*'|Bearer[ \t]+(?:\[redacted\]|[^\s,;&}\]\r\n"']+)|\[redacted\]|[^\s,;&}\]\r\n]+)`,
	)
	keyAssignmentRE = regexp.MustCompile(
		`(?i)((?:\bkey\s*=|["']key["']\s*[:=])\s*)` +
			`(?:"[^"\r\n]*"|'[^'\r\n]*'|\[redacted\]|[^\s,;&}\]\r\n]+)`,
	)
	bearerRE       = regexp.MustCompile(`(?i)\bBearer[ \t]+(?:\[redacted\]|[^\s,;&}\]\r\n"']+)`)
	emailRE        = regexp.MustCompile(`(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b`)
	gatewayTokenRE = regexp.MustCompile(`\bllmgw_[A-Za-z0-9_-]{32}([^A-Za-z0-9_-]|$)`)
	tokenRE        = regexp.MustCompile(`(?i)\b(?:gh[oupsr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,}|sk-[A-Za-z0-9_-]{10,}|gsk_[A-Za-z0-9_-]{10,})\b`)
	longValueRE    = regexp.MustCompile(`\b[A-Za-z0-9]{20,}\b`)
)

// sanitizeDiagnosticText removes credential-shaped and personally
// identifying values from text that may be logged or shown. Applying it
// again changes nothing.
func sanitizeDiagnosticText(text string) string {
	text = pemBlockRE.ReplaceAllString(text, redactedDiagnostic)
	text = sensitiveAssignmentRE.ReplaceAllString(text, `${1}`+redactedDiagnostic)
	text = keyAssignmentRE.ReplaceAllString(text, `${1}`+redactedDiagnostic)
	text = bearerRE.ReplaceAllString(text, "Bearer "+redactedDiagnostic)
	text = emailRE.ReplaceAllString(text, redactedDiagnostic)
	text = gatewayTokenRE.ReplaceAllString(text, redactedDiagnostic+`${1}`)
	text = tokenRE.ReplaceAllString(text, redactedDiagnostic)
	return longValueRE.ReplaceAllString(text, redactedDiagnostic)
}

// sanitizeDiagnosticTextLimit sanitizes text, then keeps at most maxChars
// runes of it. Sanitizing first keeps a credential the limit cuts short
// from escaping the patterns that recognize it whole.
func sanitizeDiagnosticTextLimit(text string, maxChars int) string {
	text = sanitizeDiagnosticText(text)
	if maxChars <= 0 {
		return ""
	}
	if runes := []rune(text); len(runes) > maxChars {
		return string(runes[:maxChars])
	}
	return text
}
