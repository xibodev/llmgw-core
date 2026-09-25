package providers

import (
	"context"
	"errors"
	"net/http"
	"time"
)

// The websocket message types Edge TTS reads: RFC 6455's opcodes for them,
// which gorilla/websocket's TextMessage and BinaryMessage are too.
const (
	WebSocketTextMessage   = 1
	WebSocketBinaryMessage = 2
)

// WebSocketDialer opens a websocket connection to url, sending header with
// the handshake and offering subprotocols, as gorilla/websocket's
// Dialer.DialContext does with its Subprotocols. Core has no websocket
// client, so a product that serves Edge TTS supplies one.
//
// The dial ends when ctx does; the connection it returns does not depend
// on ctx. A handshake the server refuses returns the response it got, with
// its status and headers, beside the error, so Edge TTS can read the
// server's Date. Edge TTS closes the body of every response it is given.
type WebSocketDialer func(ctx context.Context, url string, header http.Header, subprotocols []string) (WebSocketConn, *http.Response, error)

// WebSocketConn is an open websocket connection: as much of one as Edge TTS
// uses. Edge TTS writes and reads on one goroutine, and closes the
// connection from another once the exchange's context ends, which is how a
// read that waits past the deadline ends. So Close must be safe to call
// concurrently with the other methods, and more than once.
type WebSocketConn interface {
	// WriteText sends data as one text message.
	WriteText(ctx context.Context, data []byte) error
	// Read returns the next data message and its type,
	// WebSocketTextMessage or WebSocketBinaryMessage, answering control
	// messages itself. ctx carries the exchange's deadline, which an
	// adapter may apply to the connection.
	Read(ctx context.Context) (messageType int, data []byte, err error)
	// Close closes the connection.
	Close() error
}

// edgeTTSRedactedError keeps an error for errors.Is and errors.As while
// its text, which may quote a signed URL and so the access token, is
// redacted.
type edgeTTSRedactedError struct{ err error }

func (e *edgeTTSRedactedError) Error() string {
	return sanitizeDiagnosticTextLimit(e.err.Error(), diagnosticErrorLimit)
}

func (e *edgeTTSRedactedError) Unwrap() error { return e.err }

// open dials the synthesis websocket as the gateway does. A handshake
// refused with 403 usually means the signature's clock drifted from the
// service's, so the skew is learned from the refusal's Date and the dial
// repeated once, with a new signature and connection ID.
func (p *EdgeTTS) open(ctx context.Context, token string) (WebSocketConn, error) {
	conn, response, err := p.dialOnce(ctx, token)
	if err != nil && response != nil && response.StatusCode == http.StatusForbidden {
		p.learn(response.Header.Get("Date"))
		conn, response, err = p.dialOnce(ctx, token)
	}
	switch {
	case err == nil:
		return conn, nil
	case response != nil && response.StatusCode != 0:
		return nil, httpStatusFailure("Edge TTS", response, nil, p.now())
	}
	return nil, transportFailure(ctx, "Edge TTS could not open its websocket", &edgeTTSRedactedError{err})
}

// dialOnce makes one handshake within the timeout, with the headers of
// Edge's read-aloud feature, and closes the body of the response it gets.
func (p *EdgeTTS) dialOnce(ctx context.Context, token string) (WebSocketConn, *http.Response, error) {
	dialCtx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	header := http.Header{}
	header.Set("User-Agent", edgeTTSUserAgent)
	header.Set("Origin", edgeTTSOrigin)
	header.Set("Pragma", "no-cache")
	header.Set("Cache-Control", "no-cache")
	conn, response, err := p.dial(dialCtx, p.websocketURL(token), header, []string{edgeTTSSubprotocol})
	if response != nil && response.Body != nil {
		_ = response.Body.Close()
	}
	switch {
	case err != nil && conn != nil:
		_ = conn.Close()
		conn = nil
	case err == nil && conn == nil:
		err = errors.New("the websocket dialer returned no connection")
	}
	return conn, response, err
}

// learn takes the skew from date, the RFC 1123 Date of the service's
// answer, as the gateway does: the service's clock is ahead of the local
// one by the difference. A date that does not parse teaches nothing.
func (p *EdgeTTS) learn(date string) {
	server, err := time.Parse(time.RFC1123, date)
	if err != nil {
		return
	}
	skew := float64(server.UTC().Unix()) - float64(p.now().UTC().Unix())
	p.mu.Lock()
	p.skew = skew
	p.mu.Unlock()
}

// serviceNow is the service's time in Unix seconds as this instance
// estimates it: the local clock and the skew it learned.
func (p *EdgeTTS) serviceNow() float64 {
	local := float64(p.now().UTC().Unix())
	p.mu.Lock()
	defer p.mu.Unlock()
	return local + p.skew
}
