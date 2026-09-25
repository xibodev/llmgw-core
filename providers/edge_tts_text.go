package providers

import (
	"strings"
	"unicode/utf8"
)

// edgeTTSSanitize replaces the control characters the service rejects
// with spaces, as the gateway does; tab, line feed and carriage return
// stay.
func edgeTTSSanitize(text string) string {
	var builder strings.Builder
	for _, r := range text {
		code := int(r)
		if (code >= 0 && code <= 8) || code == 11 || code == 12 || (code >= 14 && code <= 31) {
			builder.WriteRune(' ')
		} else {
			builder.WriteRune(r)
		}
	}
	return builder.String()
}

// edgeTTSEscapeXML escapes text for SSML as the gateway does.
func edgeTTSEscapeXML(text string) string {
	return strings.NewReplacer(
		"&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&apos;",
	).Replace(text)
}

// edgeTTSSplit breaks escaped text into chunks of at most limit bytes as
// the gateway does: at the last whitespace within the limit, or else at the
// limit but before an entity it would cut, and dropping the whitespace a
// chunk starts with. A limit shorter than an entity, which the gateway
// never uses, can still cut one.
//
// Unlike the gateway, a chunk cut at the limit ends where a character does.
// The gateway can cut a character in two when no whitespace falls within
// the limit, as in long Chinese text, and a text message that is not UTF-8
// fails the connection, so only frames that would fail differ.
func edgeTTSSplit(text string, limit int) []string {
	if len(text) <= limit {
		return []string{text}
	}
	var chunks []string
	remaining := text
	for len(remaining) > limit {
		cut := strings.LastIndexAny(remaining[:limit], " \t\n")
		if cut <= 0 {
			cut = limit
			for cut > 0 && !utf8.RuneStart(remaining[cut]) {
				cut--
			}
			if ampersand := strings.LastIndex(remaining[:cut], "&"); ampersand >= 0 && !strings.Contains(remaining[ampersand:cut], ";") {
				cut = ampersand
			}
			if cut == 0 {
				cut = limit
			}
		}
		chunks = append(chunks, remaining[:cut])
		remaining = strings.TrimLeft(remaining[cut:], " \t\n")
	}
	if remaining != "" {
		chunks = append(chunks, remaining)
	}
	return chunks
}
