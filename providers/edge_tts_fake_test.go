package providers

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net/http"
	"slices"
	"sync"
)

// edgeTTSFrame is one message a scripted connection sends.
type edgeTTSFrame struct {
	kind int
	data []byte
}

// edgeTTSBinaryFrame is a binary message as the service sends one: a
// two-byte big-endian header length, the headers, and the payload.
func edgeTTSBinaryFrame(headers, payload string) edgeTTSFrame {
	data := make([]byte, 2+len(headers)+len(payload))
	binary.BigEndian.PutUint16(data, uint16(len(headers)))
	copy(data[2:], headers)
	copy(data[2+len(headers):], payload)
	return edgeTTSFrame{kind: WebSocketBinaryMessage, data: data}
}

func edgeTTSAudioFrame(audio string) edgeTTSFrame {
	return edgeTTSBinaryFrame("X-RequestId:fixture\r\nContent-Type:audio/mpeg\r\nPath:audio\r\n", audio)
}

func edgeTTSTextFrame(text string) edgeTTSFrame {
	return edgeTTSFrame{kind: WebSocketTextMessage, data: []byte(text)}
}

func edgeTTSTurnEnd() edgeTTSFrame {
	return edgeTTSTextFrame("X-RequestId:fixture\r\nPath:turn.end\r\n\r\n{}")
}

// edgeTTSAnswer is how the scripted service answers one dial: a refused
// handshake, a transport failure, or a connection that sends frames once
// it has read the speech configuration and the SSML, and then waits until
// it is closed, calling stalled, when set, as it starts to wait.
type edgeTTSAnswer struct {
	status  int
	header  http.Header
	err     error
	frames  []edgeTTSFrame
	stalled func()
}

// edgeTTSDialed is one dial the scripted service received.
type edgeTTSDialed struct {
	url          string
	header       http.Header
	subprotocols []string
}

// edgeTTSCloseCounter is a response body that counts its closes.
type edgeTTSCloseCounter struct {
	mu     sync.Mutex
	closes int
}

func (*edgeTTSCloseCounter) Read([]byte) (int, error) { return 0, io.EOF }
func (b *edgeTTSCloseCounter) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closes++
	return nil
}

func (b *edgeTTSCloseCounter) count() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.closes
}

// edgeTTSService is a scripted Edge TTS service behind a WebSocketDialer.
// Each dial takes the next answer; the last one repeats.
type edgeTTSService struct {
	mu      sync.Mutex
	answers []edgeTTSAnswer
	dials   []edgeTTSDialed
	bodies  []*edgeTTSCloseCounter
	conns   []*edgeTTSConn
}

func (s *edgeTTSService) dial(_ context.Context, url string, header http.Header, subprotocols []string) (WebSocketConn, *http.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.dials = append(s.dials, edgeTTSDialed{url: url, header: header.Clone(), subprotocols: slices.Clone(subprotocols)})
	answer := s.answers[min(len(s.dials), len(s.answers))-1]
	body := &edgeTTSCloseCounter{}
	s.bodies = append(s.bodies, body)
	switch {
	case answer.status != 0:
		// A library's handshake error may quote the signed URL.
		return nil, &http.Response{StatusCode: answer.status, Header: answer.header, Body: body}, errors.New("websocket: bad handshake for " + url)
	case answer.err != nil:
		return nil, nil, answer.err
	}
	conn := &edgeTTSConn{frames: slices.Clone(answer.frames), stalled: answer.stalled, closed: make(chan struct{})}
	s.conns = append(s.conns, conn)
	return conn, &http.Response{StatusCode: http.StatusSwitchingProtocols, Header: http.Header{}, Body: body}, nil
}

func (s *edgeTTSService) recorded() ([]edgeTTSDialed, []*edgeTTSConn, []*edgeTTSCloseCounter) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.dials), slices.Clone(s.conns), slices.Clone(s.bodies)
}

// edgeTTSConn is a scripted connection. It fails a read that comes before
// the request is written, and once its frames run out it waits until it
// is closed, as a connection does whose service never ends its turn.
type edgeTTSConn struct {
	mu      sync.Mutex
	written [][]byte
	frames  []edgeTTSFrame
	stalled func()
	closes  int
	closed  chan struct{}
}

func (c *edgeTTSConn) WriteText(_ context.Context, data []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closes > 0 {
		return errors.New("write on a closed connection")
	}
	c.written = append(c.written, slices.Clone(data))
	return nil
}

func (c *edgeTTSConn) Read(context.Context) (int, []byte, error) {
	c.mu.Lock()
	switch {
	case len(c.written) < 2:
		c.mu.Unlock()
		return 0, nil, errors.New("read before the request was written")
	case len(c.frames) > 0:
		frame := c.frames[0]
		c.frames = c.frames[1:]
		c.mu.Unlock()
		return frame.kind, frame.data, nil
	}
	stalled := c.stalled
	c.mu.Unlock()
	if stalled != nil {
		stalled()
	}
	<-c.closed
	return 0, nil, errors.New("read on a closed connection")
}

func (c *edgeTTSConn) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closes++; c.closes == 1 {
		close(c.closed)
	}
	return nil
}

func (c *edgeTTSConn) state() ([][]byte, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.written), c.closes
}
