package translation

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"

	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
)

// chatStream renders an Anthropic Messages stream as a Chat Completions
// stream: llm-translate's converter turns each Messages event into Chat
// chunks, and the stream ends with one data: [DONE]. The converter pulls
// lines and pushes chunks, so it runs in a goroutine that feeds a pull-based
// core.StreamIter, as messagesStream does in the other direction.
//
// Chat chunks carry no usage unless the request asks for it with
// stream_options.include_usage. When it does, the stream sends the usage the
// Messages stream reported in a last chunk without choices, as Chat does,
// and its usage loss is no longer reported.
//
// A Messages error event, an upstream error, or an upstream that ends before
// message_stop fails the stream once the chunks before it are delivered. A
// failed stream sends no finish reason and no [DONE], so it never looks
// complete.
type chatStream struct {
	upstream     core.StreamIter
	includeUsage bool
	frames       chan []byte
	stop         chan struct{}
	stopOnce     sync.Once
	finished     chan struct{}

	// The converter goroutine alone reads and writes these.
	id        string
	usage     messagesUsage
	usageSeen bool
	complete  bool

	// mu guards what the consumer reads too.
	mu     sync.Mutex
	losses []core.Loss
	err    error
}

func newChatStream(upstream core.StreamIter, model string, includeUsage bool, requestLosses []core.Loss) *chatStream {
	s := &chatStream{
		upstream:     upstream,
		includeUsage: includeUsage,
		frames:       make(chan []byte),
		stop:         make(chan struct{}),
		finished:     make(chan struct{}),
		losses:       append([]core.Loss(nil), requestLosses...),
	}
	go s.run(model)
	return s
}

func (s *chatStream) run(model string) {
	defer close(s.finished)
	defer close(s.frames)
	report := translate.AnthropicSSEToOpenAIChunksWithReport(s.nextLine(), model, s.emit)
	losses := report.Losses
	if !s.failed() {
		if s.includeUsage && s.usageSeen {
			s.emit(usageChunk(s.id, model, s.usage))
			losses = withoutUsageLoss(losses)
		}
		s.send([]byte("data: [DONE]\n\n"))
	}
	s.mu.Lock()
	s.losses = append(s.losses, losses...)
	s.mu.Unlock()
}

// nextLine hands the converter the upstream's lines one at a time. It stops
// at the upstream's end, on an upstream error or a Messages error event, or
// when the consumer closes the stream.
func (s *chatStream) nextLine() func() (string, bool) {
	var pending []string
	return func() (string, bool) {
		for len(pending) == 0 {
			if s.closing() {
				return "", false
			}
			frame, err := s.upstream.Next()
			if err != nil {
				switch {
				case s.closing():
					// The consumer closed the stream: what the upstream
					// reports now is the close, not a failure.
				case !errors.Is(err, io.EOF):
					s.fail(err)
				case !s.complete:
					s.fail(cutShort())
				}
				return "", false
			}
			if payload, ok := ssePayload(frame); ok {
				if failure := s.observe(payload); failure != nil {
					s.fail(failure)
					return "", false
				}
			}
			pending = strings.Split(strings.ReplaceAll(string(frame), "\r\n", "\n"), "\n")
		}
		line := pending[0]
		pending = pending[1:]
		return line, true
	}
}

// observe reads the usage and the end a Messages event reports, and returns
// the error a Messages error event reports.
func (s *chatStream) observe(payload string) error {
	var event struct {
		Type    string `json:"type"`
		Message struct {
			Usage map[string]any `json:"usage"`
		} `json:"message"`
		Usage map[string]any `json:"usage"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(payload), &event) != nil {
		return nil
	}
	switch event.Type {
	case "message_start":
		if event.Message.Usage != nil {
			s.usage.update(event.Message.Usage)
			s.usageSeen = true
		}
	case "message_delta":
		if event.Usage != nil {
			s.usage.update(event.Usage)
			s.usageSeen = true
		}
	case "message_stop":
		s.complete = true
	case "error":
		return messagesStreamError(event.Error.Type, event.Error.Message)
	}
	return nil
}

// messagesErrorStatus is the HTTP status the Messages API documents for an
// error type, so an error event classifies as the same error answered with
// its status would. An unknown type has none.
func messagesErrorStatus(kind string) int {
	switch kind {
	case "invalid_request_error":
		return 400
	case "authentication_error":
		return 401
	case "permission_error":
		return 403
	case "not_found_error":
		return 404
	case "request_too_large":
		return 413
	case "rate_limit_error":
		return 429
	case "api_error":
		return 500
	case "timeout_error":
		return 504
	case "overloaded_error":
		return 529
	}
	return 0
}

// messagesStreamError is the error of a Messages error event. A rate limit,
// a timeout, an overload or an internal error is transient: it permits a
// retry and failover and counts against the provider. The stream got no
// status of its own, so the classification carries none.
func messagesStreamError(kind, message string) error {
	text := "the Messages stream reported an error"
	if kind != "" {
		text += ": " + kind
	}
	if message = strings.TrimSpace(message); message != "" {
		if len(message) > 300 {
			message = strings.ToValidUTF8(message[:300], "")
		}
		text += ": " + message
	}
	status := messagesErrorStatus(kind)
	transient := status == 408 || status == 429 || status >= 500
	return &core.ProviderError{
		Message: text,
		Class:   core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: status}).ErrorClass,
		Classification: core.ProviderErrorClassification{
			Retryable: transient, FailoverEligible: transient, CircuitFailure: transient,
		},
	}
}

// cutShort is the failure of a Messages stream that ended before
// message_stop. Another target may serve the request and it counts against
// the provider, as an unusable answer does; the part already delivered
// cannot be taken back, so it is not retried.
func cutShort() error {
	return &core.ProviderError{
		Message: "the Messages stream ended before message_stop",
		Class:   core.ProviderErrorUpstream,
		Classification: core.ProviderErrorClassification{
			FailoverEligible: true, CircuitFailure: true,
		},
	}
}

func (s *chatStream) closing() bool {
	select {
	case <-s.stop:
		return true
	default:
		return false
	}
}

func (s *chatStream) fail(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err == nil {
		s.err = err
	}
}

func (s *chatStream) failed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err != nil
}

// emit forwards one converted chunk. After a failure the converter still
// emits its closing chunk; it is dropped, so a failed stream never looks
// complete.
func (s *chatStream) emit(chunk string) {
	if s.failed() {
		return
	}
	if s.id == "" {
		var head struct {
			ID string `json:"id"`
		}
		if json.Unmarshal([]byte(chunk), &head) == nil {
			s.id = head.ID
		}
	}
	s.send([]byte("data: " + chunk + "\n\n"))
}

func (s *chatStream) send(frame []byte) {
	select {
	case s.frames <- frame:
	case <-s.stop:
	}
}

// Next returns the next Chat chunk record, the stream's failure once the
// chunks before it are delivered, or io.EOF after data: [DONE].
func (s *chatStream) Next() ([]byte, error) {
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
func (s *chatStream) Close() error {
	s.stopOnce.Do(func() { close(s.stop) })
	err := s.upstream.Close()
	for range s.frames {
	}
	<-s.finished
	return err
}

// Losses implements core.LossReporter.
func (s *chatStream) Losses() []core.Loss {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]core.Loss(nil), s.losses...)
}

// usageChunk is the last Chat chunk of a stream that asks for usage: no
// choices, and the usage the Messages stream reported.
func usageChunk(id, model string, usage messagesUsage) string {
	body := map[string]any{"object": "chat.completion.chunk", "model": model, "choices": []any{}, "usage": usage.chat()}
	if id != "" {
		body["id"] = id
	}
	encoded, _ := json.Marshal(body)
	return string(encoded)
}

// withoutUsageLoss drops the loss that says the stream carries no usage,
// once it does.
func withoutUsageLoss(losses []core.Loss) []core.Loss {
	kept := losses[:0:0]
	for _, loss := range losses {
		if loss.Path == "usage" && loss.Class == translate.LossDropped {
			continue
		}
		kept = append(kept, loss)
	}
	return kept
}

// includesUsage reports whether a Chat request asks for usage in its stream.
func includesUsage(payload map[string]any) bool {
	options, _ := payload["stream_options"].(map[string]any)
	include, _ := options["include_usage"].(bool)
	return include
}
