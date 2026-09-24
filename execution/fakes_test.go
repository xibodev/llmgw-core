package execution_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

// clock is a manual clock shared by a test's tracker and executor. Sleep
// advances it instead of waiting, so repeats take no real time.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func newClock() *clock {
	return &clock{now: time.Unix(1_700_000_000, 0).UTC()}
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func (c *clock) Sleep(ctx context.Context, d time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.Advance(d)
	return nil
}

// tracker returns a HealthTracker on clock with one policy for every key.
func tracker(clock *clock, policy execution.HealthPolicy) *execution.HealthTracker {
	return execution.NewHealthTracker(execution.HealthOptions{
		Policy: func(string) execution.HealthPolicy { return policy },
		Now:    clock.Now,
	})
}

// Failures with the routing metadata the tests need.

func overloaded() error {
	return &core.ProviderError{Message: "upstream overloaded", Class: core.ProviderErrorUpstream,
		Classification: core.ProviderErrorClassification{StatusCode: 503, Retryable: true, FailoverEligible: true, CircuitFailure: true}}
}

func disconnected() error {
	return &core.ProviderError{Message: "upstream connection reset", Class: core.ProviderErrorTransport,
		Classification: core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}}
}

func rejected(status int) error {
	return &core.ProviderError{Message: "upstream rejected the request", Class: core.ProviderErrorInvalidRequest,
		Classification: core.ProviderErrorClassification{StatusCode: status}}
}

func rateLimited(retryAfter time.Duration) error {
	return &core.ProviderError{Message: "upstream rate limited the request", Class: core.ProviderErrorRateLimited,
		Classification: core.ProviderErrorClassification{StatusCode: 429, Retryable: true, FailoverEligible: true, RetryAfter: retryAfter}}
}

func unsupported() error {
	return &core.SurfaceError{Surface: core.ModelSurfaceResponses, Model: "model"}
}

func assertAvailable(t *testing.T, health execution.Health, key string) {
	t.Helper()
	if available, until := health.Available(key); !available || !until.IsZero() {
		t.Fatalf("%s: available=%v until=%v, want available", key, available, until)
	}
}

func assertUnavailableUntil(t *testing.T, health execution.Health, key string, want time.Time) {
	t.Helper()
	if available, until := health.Available(key); available || !until.Equal(want) {
		t.Fatalf("%s: available=%v until=%v, want unavailable until %v", key, available, until, want)
	}
}

// stream is a scripted core.StreamIter: it yields its frames, then fails
// with err, or ends with io.EOF when err is nil. errWithLast delivers the
// last frame together with err, as some streams do.
type stream struct {
	mu          sync.Mutex
	frames      []string
	err         error
	errWithLast bool
	closes      int
}

func (s *stream) Next() ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closes > 0 {
		return nil, io.ErrClosedPipe
	}
	if len(s.frames) > 0 {
		frame := s.frames[0]
		s.frames = s.frames[1:]
		if len(s.frames) == 0 && s.errWithLast {
			return []byte(frame), s.err
		}
		return []byte(frame), nil
	}
	if s.err != nil {
		return nil, s.err
	}
	return nil, io.EOF
}

func (s *stream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closes++
	return nil
}

func (s *stream) closed() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closes
}

// lossyStream is a stream that translates and reports its losses.
type lossyStream struct {
	*stream
	losses []core.Loss
}

func (s lossyStream) Losses() []core.Loss { return s.losses }

// upstreams scripts what each candidate's open returns and counts the opens.
type upstreams struct {
	mu      sync.Mutex
	opened  []string
	streams map[string]*stream
	errs    map[string]error
}

func (u *upstreams) open(_ context.Context, candidate string) (core.StreamIter, error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.opened = append(u.opened, candidate)
	if err := u.errs[candidate]; err != nil {
		return nil, err
	}
	return u.streams[candidate], nil
}

func (u *upstreams) opens() []string {
	u.mu.Lock()
	defer u.mu.Unlock()
	return append([]string(nil), u.opened...)
}

// Chat Completions frames as the gateway's characterization upstream writes
// them.

func chunk(model, delta, finish, usage string) string {
	return fmt.Sprintf(`data: {"id":"chatcmpl_fixture","object":"chat.completion.chunk","created":1700000000,"model":%q,"choices":[{"index":0,"delta":%s,"finish_reason":%s}]%s}`+"\n\n",
		model, delta, finish, usage)
}

const (
	done      = "data: [DONE]\n\n"
	keepalive = ": keepalive\n\n"
)

// chatOutput is the product's predicate for Chat Completions frames: a frame
// carries output when a delta has content, reasoning or tool calls.
// Keepalives, role-only frames, finish frames and [DONE] carry none.
func chatOutput(frame []byte) bool {
	for line := range strings.SplitSeq(string(frame), "\n") {
		data, ok := strings.CutPrefix(strings.TrimRight(line, "\r"), "data:")
		if !ok {
			continue
		}
		var decoded struct {
			Choices []struct {
				Delta struct {
					Content          string          `json:"content"`
					ReasoningContent string          `json:"reasoning_content"`
					Reasoning        string          `json:"reasoning"`
					ToolCalls        json.RawMessage `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(strings.TrimSpace(data)), &decoded) != nil {
			continue
		}
		for _, choice := range decoded.Choices {
			delta := choice.Delta
			toolCalls := len(delta.ToolCalls) > 0 && string(delta.ToolCalls) != "null" && string(delta.ToolCalls) != "[]"
			if delta.Content != "" || delta.ReasoningContent != "" || delta.Reasoning != "" || toolCalls {
				return true
			}
		}
	}
	return false
}

// drain reads a stream to its end and returns its frames and final error.
func drain(t *testing.T, s core.StreamIter) ([]string, error) {
	t.Helper()
	var frames []string
	for range 1000 {
		frame, err := s.Next()
		if len(frame) > 0 {
			frames = append(frames, string(frame))
		}
		if err != nil {
			return frames, err
		}
	}
	t.Fatal("the stream never ended")
	return nil, nil
}
