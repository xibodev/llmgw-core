package providers

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// The operational constants of Edge's read-aloud feature, as the gateway
// sends them. The access token is the public one the feature itself sends,
// published across the edge-tts ecosystem; a credential overrides it.
const (
	edgeTTSDefaultBase    = "api.msedgeservices.com/tts/cognitiveservices"
	edgeTTSDefaultToken   = "6A5AA1D4EAFF4E9FB37E23D68491D6F4"
	edgeTTSDefaultVoice   = "en-US-EmmaMultilingualNeural"
	edgeTTSChromiumFull   = "140.0.3485.14"
	edgeTTSChromiumMajor  = "140"
	edgeTTSWindowsEpoch   = 11644473600
	edgeTTSMaxMessageSize = 4096
	edgeTTSSubprotocol    = "synthesize"
	edgeTTSOrigin         = "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold"
	edgeTTSUserAgent      = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" +
		" (KHTML, like Gecko) Chrome/" + edgeTTSChromiumMajor + ".0.0.0 Safari/537.36" +
		" Edg/" + edgeTTSChromiumMajor + ".0.0.0"
	edgeTTSTimestampLayout = "Mon Jan 02 2006 15:04:05 GMT+0000 (Coordinated Universal Time)"
)

// The gateway interpolates the voice and the rate into SSML attributes and
// the token into URLs, all unescaped, so each is checked instead: a voice
// is letters, digits and the punctuation of a voice's long name, a rate a
// signed percentage, as the gateway computes one, and a token URL-safe.
// What passes is sent as the gateway sends it.
var (
	edgeTTSVoicePattern = regexp.MustCompile(`^[\p{L}\p{N}][\p{L}\p{N} ._,()-]*$`)
	edgeTTSRatePattern  = regexp.MustCompile(`^[+-][0-9]+%$`)
	edgeTTSTokenPattern = regexp.MustCompile(`^[A-Za-z0-9._~-]+$`)
)

// edgeTTSSignature derives the Sec-MS-GEC request signature as the gateway
// does: the SHA-256 of the service's time as Windows file time ticks,
// rounded down to five minutes, followed by the access token, in upper-case
// hex. The arithmetic is the gateway's, in float64, which holds every such
// tick count exactly.
func edgeTTSSignature(serviceUnix float64, token string) string {
	ticks := serviceUnix + edgeTTSWindowsEpoch
	ticks = math.Floor(ticks/300) * 300
	ticks *= 1e7
	digest := sha256.Sum256([]byte(fmt.Sprintf("%.0f%s", ticks, token)))
	return strings.ToUpper(hex.EncodeToString(digest[:]))
}

func (p *EdgeTTS) websocketURL(token string) string {
	scheme := "wss://"
	if p.insecure {
		scheme = "ws://"
	}
	return scheme + p.base + "/websocket/v1?Ocp-Apim-Subscription-Key=" + token +
		"&Sec-MS-GEC=" + edgeTTSSignature(p.serviceNow(), token) +
		"&Sec-MS-GEC-Version=1-" + edgeTTSChromiumFull +
		"&ConnectionId=" + p.newID()
}

// voicesURL is the voice list's, signed as a synthesis is: without the
// signature the service refuses it with 403.
func (p *EdgeTTS) voicesURL(token string) string {
	scheme := "https://"
	if p.insecure {
		scheme = "http://"
	}
	return scheme + p.base + "/voices/list?Ocp-Apim-Subscription-Key=" + token +
		"&Sec-MS-GEC=" + edgeTTSSignature(p.serviceNow(), token) +
		"&Sec-MS-GEC-Version=1-" + edgeTTSChromiumFull
}

// edgeTTSSpeechConfig is the first message of a synthesis, as the gateway
// sends it: word boundaries on, sentence boundaries off, and MP3.
func edgeTTSSpeechConfig(now time.Time) []byte {
	return []byte("X-Timestamp:" + now.UTC().Format(edgeTTSTimestampLayout) + "\r\n" +
		"Content-Type:application/json; charset=utf-8\r\n" +
		"Path:speech.config\r\n\r\n" +
		`{"context":{"synthesis":{"audio":{"metadataoptions":{` +
		`"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"true"},` +
		`"outputFormat":"` + EdgeTTSOutputFormat + `"}}}}`)
}

// edgeTTSSSML is the message that asks for one chunk's speech, as the
// gateway sends it. Its X-Timestamp ends in the Z the service expects.
func edgeTTSSSML(requestID string, now time.Time, voice, rate, escapedText string) []byte {
	ssml := "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>" +
		"<voice name='" + voice + "'><prosody pitch='+0Hz' rate='" + rate + "' volume='+0%'>" +
		escapedText + "</prosody></voice></speak>"
	return []byte("X-RequestId:" + requestID + "\r\n" +
		"Content-Type:application/ssml+xml\r\n" +
		"X-Timestamp:" + now.UTC().Format(edgeTTSTimestampLayout) + "Z\r\n" +
		"Path:ssml\r\n\r\n" + ssml)
}

// edgeTTSHeaders reads the header lines that open a service message.
func edgeTTSHeaders(data []byte) map[string]string {
	headers := map[string]string{}
	text := string(data)
	if separator := strings.Index(text, "\r\n\r\n"); separator >= 0 {
		text = text[:separator]
	}
	for _, line := range strings.Split(text, "\r\n") {
		if key, value, found := strings.Cut(line, ":"); found {
			headers[strings.TrimSpace(key)] = strings.TrimSpace(value)
		}
	}
	return headers
}

// edgeTTSAudio returns the audio a binary message carries after its
// two-byte big-endian header length and the headers, when they name the
// audio path.
func edgeTTSAudio(data []byte) ([]byte, bool) {
	if len(data) < 2 {
		return nil, false
	}
	headerLength := int(binary.BigEndian.Uint16(data[:2]))
	if headerLength+2 > len(data) || edgeTTSHeaders(data[2 : 2+headerLength])["Path"] != "audio" {
		return nil, false
	}
	return data[2+headerLength:], true
}

// edgeTTSRate maps an OpenAI speech speed, where 1 is normal, to the
// prosody rate the service takes, as the gateway does: 1.2 is "+20%", 0.5
// "-50%", and zero or less normal. A speed whose percentage an int64 cannot
// hold, which the gateway renders differently on each platform, has none.
func edgeTTSRate(speed float64) (string, bool) {
	if speed <= 0 {
		return "+0%", true
	}
	percent := math.Round((speed - 1) * 100)
	if percent >= math.MaxInt64 || math.IsNaN(percent) {
		return "", false
	}
	if percent >= 0 {
		return "+" + strconv.FormatInt(int64(percent), 10) + "%", true
	}
	return strconv.FormatInt(int64(percent), 10) + "%", true
}
