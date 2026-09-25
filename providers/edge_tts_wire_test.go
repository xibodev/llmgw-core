package providers

import (
	"crypto/sha256"
	"encoding/hex"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// The expected values are what the gateway's functions return for the same
// inputs.
func TestEdgeTTSSignatureIsTheGateways(t *testing.T) {
	t.Parallel()
	at := func(clock time.Time) float64 { return float64(clock.Unix()) }
	for _, tc := range []struct {
		clock       time.Time
		token, want string
	}{
		{edgeTTSFixtureNow, edgeTTSFixtureToken, edgeTTSFixtureSignature},
		{edgeTTSFixtureNow.Add(time.Hour), edgeTTSFixtureToken, edgeTTSSkewedSignature},
		{edgeTTSFixtureNow, edgeTTSDefaultToken, "FCAF9AFC958E0FFD247DAA1372D435595547949C6FFF8721B298DF5C94147A35"},
		// The signature holds for the five minutes the clock rounds to.
		{time.Date(2026, time.March, 4, 5, 5, 0, 0, time.UTC), edgeTTSFixtureToken, edgeTTSFixtureSignature},
		{time.Date(2026, time.March, 4, 5, 9, 59, 0, time.UTC), edgeTTSFixtureToken, edgeTTSFixtureSignature},
	} {
		if got := edgeTTSSignature(at(tc.clock), tc.token); got != tc.want {
			t.Fatalf("signature at %v = %s, want %s", tc.clock, got, tc.want)
		}
	}
	// Integer arithmetic derives the same signature.
	next := time.Date(2026, time.March, 4, 5, 10, 0, 0, time.UTC)
	ticks := (next.Unix() + edgeTTSWindowsEpoch) / 300 * 300 * 10_000_000
	digest := sha256.Sum256([]byte(strconv.FormatInt(ticks, 10) + edgeTTSFixtureToken))
	if got, want := edgeTTSSignature(at(next), edgeTTSFixtureToken), strings.ToUpper(hex.EncodeToString(digest[:])); got != want || got == edgeTTSFixtureSignature {
		t.Fatalf("signature at %v = %s, want %s", next, got, want)
	}
	provider := newFixtureEdgeTTS(t, &edgeTTSService{})
	const voices = "https://api.msedgeservices.com/tts/cognitiveservices/voices/list?Ocp-Apim-Subscription-Key=fixture-token" +
		"&Sec-MS-GEC=" + edgeTTSFixtureSignature + "&Sec-MS-GEC-Version=1-140.0.3485.14"
	if got := provider.voicesURL(edgeTTSFixtureToken); got != voices {
		t.Fatalf("voices URL = %s", got)
	}
}

func TestEdgeTTSMapsSpeedAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	for speed, want := range map[float64]string{
		0: "+0%", -1: "+0%", 0.25: "-75%", 0.5: "-50%", 0.994: "-1%", 0.995: "-1%",
		1: "+0%", 1.005: "+0%", 1.2: "+20%", 1.25: "+25%", 2: "+100%", 4: "+300%", 1e16: "+1000000000000000000%",
	} {
		if got, ok := edgeTTSRate(speed); !ok || got != want {
			t.Fatalf("rate(%v) = %q, %v, want %q", speed, got, ok, want)
		}
	}
	// The gateway renders these as an int's overflow, which differs by
	// platform: "-9223372036854775808%" on amd64.
	for _, speed := range []float64{1e17, 1e300} {
		if got, ok := edgeTTSRate(speed); ok {
			t.Fatalf("rate(%v) = %q, want none", speed, got)
		}
	}
}

func TestEdgeTTSCleansTextAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	if got, want := edgeTTSEscapeXML(edgeTTSSanitize("a\x00b\x08c\td\ne\x0bf\x0cg\rh\x1fi\x7fj'k\"l")), "a b c\td\ne f g\rh i\x7fj&apos;k&quot;l"; got != want {
		t.Fatalf("cleaned = %q, want %q", got, want)
	}
	digest := func(chunks []string) string {
		sum := sha256.Sum256([]byte(strings.Join(chunks, "|")))
		return hex.EncodeToString(sum[:8])
	}
	for _, tc := range []struct {
		text    string
		limit   int
		lengths []int
		digest  string
	}{
		{strings.Repeat("hello world ", 40) + "&amp;" + strings.Repeat(" tail text", 30), 128, []int{125, 125, 125, 127, 124, 124, 29}, "4cbeb96c16e3ccba"},
		{strings.Repeat("a", 120) + "&amp;" + strings.Repeat("b", 20), 128, []int{128, 17}, "01c4cd1018829a46"},
		{"&amp;" + strings.Repeat("c", 200), 128, []int{128, 77}, "05b71871a688f44c"},
		{strings.Repeat("a", 125) + "&amp;" + strings.Repeat("b", 10), 128, []int{125, 15}, "ccf10a799a7951fe"},
		{"&amp;x", 3, []int{3, 3}, "d15d4f71282b919f"},
		{edgeTTSEscapeXML(strings.Repeat("Hello <world> & friends. ", 400)), edgeTTSMaxMessageSize, []int{4094, 4094, 4094, 1715}, "c500d5e6504cbaf3"},
	} {
		chunks := edgeTTSSplit(tc.text, tc.limit)
		lengths := make([]int, len(chunks))
		for index, chunk := range chunks {
			lengths[index] = len(chunk)
		}
		if !slices.Equal(lengths, tc.lengths) || digest(chunks) != tc.digest {
			t.Fatalf("split(%d) = %v %s, want %v %s", tc.limit, lengths, digest(chunks), tc.lengths, tc.digest)
		}
	}
	// Where no whitespace falls within the limit, a chunk ends where a
	// character does, which the gateway's need not.
	chunks := edgeTTSSplit(strings.Repeat("语音", 20), 16)
	if strings.Join(chunks, "") != strings.Repeat("语音", 20) || len(chunks) != 8 {
		t.Fatalf("chunks = %q", chunks)
	}
	for _, chunk := range chunks {
		if !utf8.ValidString(chunk) || len(chunk) > 16 {
			t.Fatalf("chunk %q is not whole characters within the limit", chunk)
		}
	}
}
