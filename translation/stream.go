package translation

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"sync"

	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
)

// messagesStream renders a Chat Completions stream as an Anthropic Messages
// stream. llm-translate's converter pushes events, so it runs in a goroutine
// that feeds a pull-based core.StreamIter.
type messagesStream struct {
	upstream core.StreamIter
	frames   chan []byte
	stop     chan struct{}
	stopOnce sync.Once
	finished chan struct{}

	mu     sync.Mutex
	losses []core.Loss
	err    error
}

func newMessagesStream(upstream core.StreamIter, model string, requestLosses []core.Loss) *messagesStream {
	s := &messagesStream{
		upstream: upstream,
		frames:   make(chan []byte),
		stop:     make(chan struct{}),
		finished: make(chan struct{}),
		losses:   append([]core.Loss(nil), requestLosses...),
	}
	go s.run(model)
	return s
}

func (s *messagesStream) run(model string) {
	defer close(s.finished)
	defer close(s.frames)
	report := translate.OpenAIStreamToAnthropicSSEWithReport(s.nextChunk, model, s.emit)
	s.mu.Lock()
	s.losses = append(s.losses, report.Losses...)
	s.mu.Unlock()
}

// nextChunk hands the converter the next Chat chunk payload, without its SSE
// framing. The stream ends at [DONE], at the upstream's end, on an upstream
// error, or when the consumer closes the stream.
func (s *messagesStream) nextChunk() (string, bool) {
	for {
		select {
		case <-s.stop:
			return "", false
		default:
		}
		frame, err := s.upstream.Next()
		if err != nil {
			if !errors.Is(err, io.EOF) {
				s.mu.Lock()
				s.err = err
				s.mu.Unlock()
			}
			return "", false
		}
		payload, ok := ssePayload(frame)
		if !ok {
			continue
		}
		if payload == "[DONE]" {
			return "", false
		}
		return payload, true
	}
}

// emit forwards one converted event. After an upstream failure the
// converter still emits its closing events; they are dropped, so a failed
// stream never looks complete.
func (s *messagesStream) emit(event string) {
	s.mu.Lock()
	failed := s.err != nil
	s.mu.Unlock()
	if failed {
		return
	}
	select {
	case s.frames <- []byte(event):
	case <-s.stop:
	}
}

// Next returns the next Messages event, the upstream's error once the events
// before it are delivered, or io.EOF.
func (s *messagesStream) Next() ([]byte, error) {
	if frame, ok := <-s.frames; ok {
		return frame, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return nil, s.err
	}
	return nil, io.EOF
}

// Close stops the conversion, closes the upstream stream, and waits for the
// converter, so no goroutine outlives the stream. The upstream's Close must
// unblock a pending Next, as closing an HTTP response body does.
func (s *messagesStream) Close() error {
	s.stopOnce.Do(func() { close(s.stop) })
	err := s.upstream.Close()
	for range s.frames {
	}
	<-s.finished
	return err
}

// Losses implements core.LossReporter.
func (s *messagesStream) Losses() []core.Loss {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]core.Loss(nil), s.losses...)
}

// ssePayload returns the data of one SSE record, joining multi-line data. A
// record without data, such as a comment or keepalive, has none.
func ssePayload(frame []byte) (string, bool) {
	var data []string
	for _, line := range strings.Split(string(bytes.TrimRight(frame, "\r\n")), "\n") {
		line = strings.TrimRight(line, "\r")
		if value, ok := strings.CutPrefix(line, "data:"); ok {
			data = append(data, strings.TrimPrefix(value, " "))
		}
	}
	if len(data) == 0 {
		return "", false
	}
	return strings.Join(data, "\n"), true
}
