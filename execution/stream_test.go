package execution_test

import (
	"context"
	"errors"
	"io"
	"slices"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

// TestExecuteStreamFailsOverBeforeOutput mirrors the gateway's
// failover-before-output-stream characterization: the first route member
// answers 503 before any output, and the second serves the whole stream.
func TestExecuteStreamFailsOverBeforeOutput(t *testing.T) {
	t.Parallel()
	served := []string{
		chunk("model-b", `{"role":"assistant","content":"Hello"}`, "null", ""),
		chunk("model-b", `{"content":" from chat"}`, "null", ""),
		chunk("model-b", `{}`, `"stop"`, `,"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}`),
		done,
	}
	upstream := &upstreams{
		errs:    map[string]error{"fixture-a": core.NewProviderOperationError("chat", 503, "", nil)},
		streams: map[string]*stream{"fixture-b": {frames: served}},
	}
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 5, OpenDuration: time.Minute})
	executor := execution.Executor[string]{Health: health, Key: identity}

	result, err := execution.ExecuteStream(context.Background(), executor, []string{"fixture-a", "fixture-b"}, upstream.open, chatOutput)
	if err != nil || result.Candidate != "fixture-b" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	defer result.Value.Close()
	frames, end := drain(t, result.Value)
	if !slices.Equal(frames, served) || end != io.EOF {
		t.Fatalf("frames=%q end=%v, want the second member's stream", frames, end)
	}
	if got := upstream.opens(); !slices.Equal(got, []string{"fixture-a", "fixture-b"}) {
		t.Fatalf("opens=%v: failover before output must try both members once", got)
	}
	if first := result.Attempts[0]; first.Disposition != core.DispositionRetryable || first.Classification.StatusCode != 503 {
		t.Fatalf("first member's entry=%+v", first)
	}
	if health.State("fixture-a").Streak != 1 || health.State("fixture-b").Streak != 0 {
		t.Fatal("health did not record the failure before output and the success")
	}
}

// TestExecuteStreamNeverFailsOverAfterOutput mirrors the gateway's
// failover-after-output-stream characterization: the first member's
// connection breaks after a frame with content, so its failure reaches the
// caller and the second member is never tried.
func TestExecuteStreamNeverFailsOverAfterOutput(t *testing.T) {
	t.Parallel()
	partial := chunk("model-a", `{"role":"assistant","content":"Partial"}`, "null", "")
	cause := disconnected()
	upstream := &upstreams{streams: map[string]*stream{
		"fixture-a": {frames: []string{partial}, err: cause},
		"fixture-b": {frames: []string{chunk("model-b", `{"content":"unused"}`, "null", ""), done}},
	}}
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute})
	executor := execution.Executor[string]{Health: health, Key: identity, Retry: execution.Retry{Attempts: 3}}

	result, err := execution.ExecuteStream(context.Background(), executor, []string{"fixture-a", "fixture-b"}, upstream.open, chatOutput)
	if err != nil || result.Candidate != "fixture-a" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	frames, end := drain(t, result.Value)
	if !slices.Equal(frames, []string{partial}) {
		t.Fatalf("frames=%q, want the partial output", frames)
	}
	var afterOutput *execution.AfterOutputError
	if !errors.As(end, &afterOutput) || !errors.Is(end, cause) {
		t.Fatalf("end=%v, want the failure after output", end)
	}
	if disposition := core.ClassifyError(end).Disposition(); disposition != core.DispositionTerminal {
		t.Fatalf("a failure after output permits %q; no one may repeat the request", disposition)
	}
	if err := result.Value.Close(); err != nil {
		t.Fatal(err)
	}
	if got := upstream.opens(); !slices.Equal(got, []string{"fixture-a"}) {
		t.Fatalf("opens=%v: failover after output reached the caller must not try the next member", got)
	}
	// The stream served output: it committed as a success, and the failure
	// after it is delivered, not recorded.
	assertAvailable(t, health, "fixture-a")
	if len(result.Attempts) != 1 || result.Attempts[0].Err != nil {
		t.Fatalf("trace=%+v", result.Attempts)
	}
}

func TestExecuteStreamFailsOverWhenAStreamBreaksBeforeOutput(t *testing.T) {
	t.Parallel()
	broken := &stream{frames: []string{keepalive, chunk("model-a", `{"role":"assistant"}`, "null", "")}, err: disconnected()}
	served := []string{chunk("model-b", `{"role":"assistant","content":"Hello"}`, "null", ""), done}
	upstream := &upstreams{streams: map[string]*stream{"a": broken, "b": {frames: served}}}
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 5, OpenDuration: time.Minute})
	executor := execution.Executor[string]{Health: health, Key: identity}

	result, err := execution.ExecuteStream(context.Background(), executor, []string{"a", "b"}, upstream.open, chatOutput)
	if err != nil || result.Candidate != "b" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	frames, end := drain(t, result.Value)
	if !slices.Equal(frames, served) || end != io.EOF {
		t.Fatalf("frames=%q end=%v: the broken stream's keepalive and role frames leaked", frames, end)
	}
	if broken.closed() != 1 {
		t.Fatalf("the broken stream was closed %d times, want once", broken.closed())
	}
	if first := result.Attempts[0]; first.Disposition != core.DispositionRetryable || first.Class != core.ProviderErrorTransport {
		t.Fatalf("first entry=%+v", first)
	}
	if health.State("a").Streak != 1 {
		t.Fatal("a failure before output was not recorded")
	}
}

func TestExecuteStreamReplaysHeldFramesInOrder(t *testing.T) {
	t.Parallel()
	frames := []string{
		keepalive,
		chunk("model", `{"role":"assistant"}`, "null", ""),
		chunk("model", `{"reasoning_content":"thinking"}`, "null", ""),
		chunk("model", `{"content":"answer"}`, "null", ""),
		done,
	}
	upstream := &upstreams{streams: map[string]*stream{"a": {frames: frames}}}
	result, err := execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"a"}, upstream.open, chatOutput)
	if err != nil {
		t.Fatal(err)
	}
	got, end := drain(t, result.Value)
	if !slices.Equal(got, frames) || end != io.EOF {
		t.Fatalf("frames=%q end=%v", got, end)
	}
}

func TestExecuteStreamDeliversAStreamThatEndsWithoutOutput(t *testing.T) {
	t.Parallel()
	frames := []string{chunk("model", `{"role":"assistant"}`, "null", ""), chunk("model", `{}`, `"stop"`, ""), done}
	upstream := &upstreams{streams: map[string]*stream{
		"a": {frames: frames},
		"b": {frames: []string{chunk("model", `{"content":"unused"}`, "null", "")}},
	}}
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 2, OpenDuration: time.Minute})
	health.Record("a", overloaded())
	executor := execution.Executor[string]{Health: health, Key: identity}

	result, err := execution.ExecuteStream(context.Background(), executor, []string{"a", "b"}, upstream.open, chatOutput)
	if err != nil || result.Candidate != "a" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if state := health.State("a"); state.Streak != 0 {
		t.Fatalf("a stream that ended cleanly did not commit as a success: %+v", state)
	}
	got, end := drain(t, result.Value)
	if !slices.Equal(got, frames) || end != io.EOF {
		t.Fatalf("frames=%q end=%v", got, end)
	}
	if _, again := result.Value.Next(); again != io.EOF {
		t.Fatalf("a finished stream did not stay finished: %v", again)
	}
	if got := upstream.opens(); !slices.Equal(got, []string{"a"}) {
		t.Fatalf("opens=%v: an empty answer is an answer", got)
	}
}

func TestExecuteStreamCommitsAtTheFirstFrameWithoutAPredicate(t *testing.T) {
	t.Parallel()
	cause := disconnected()
	upstream := &upstreams{streams: map[string]*stream{
		"a": {frames: []string{keepalive}, err: cause},
		"b": {frames: []string{done}},
	}}
	result, err := execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"a", "b"}, upstream.open, nil)
	if err != nil || result.Candidate != "a" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	frames, end := drain(t, result.Value)
	if !slices.Equal(frames, []string{keepalive}) || !errors.Is(end, cause) {
		t.Fatalf("frames=%q end=%v", frames, end)
	}

	upstream = &upstreams{streams: map[string]*stream{"a": {err: cause}, "b": {frames: []string{done}}}}
	result, err = execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"a", "b"}, upstream.open, nil)
	if err != nil || result.Candidate != "b" {
		t.Fatalf("a stream that failed before any frame did not fail over: result=%+v err=%v", result, err)
	}
}

func TestExecuteStreamFailsOverWhenOutputArrivesWithAFailure(t *testing.T) {
	t.Parallel()
	upstream := &upstreams{streams: map[string]*stream{
		"a": {frames: []string{chunk("model", `{"content":"lost"}`, "null", "")}, err: disconnected(), errWithLast: true},
		"b": {frames: []string{chunk("model", `{"content":"served"}`, "null", ""), done}},
	}}
	result, err := execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"a", "b"}, upstream.open, chatOutput)
	if err != nil || result.Candidate != "b" {
		t.Fatalf("result=%+v err=%v: output that came with its stream's failure never reached the caller", result, err)
	}
}

func TestExecuteStreamKeepsAFrameThatArrivesWithTheEnd(t *testing.T) {
	t.Parallel()
	last := chunk("model", `{"content":"last"}`, "null", "")
	upstream := &upstreams{streams: map[string]*stream{"a": {frames: []string{last}, err: io.EOF, errWithLast: true}}}
	result, err := execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"a"}, upstream.open, func([]byte) bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	if frames, end := drain(t, result.Value); !slices.Equal(frames, []string{last}) || end != io.EOF {
		t.Fatalf("frames=%q end=%v", frames, end)
	}
}

// hanging is a stream whose reads wait for its context, as an HTTP body's do.
type hanging struct {
	ctx    context.Context
	closed chan struct{}
}

func (s *hanging) Next() ([]byte, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *hanging) Close() error {
	close(s.closed)
	return nil
}

func TestExecuteStreamStopsWhenTheCallerGivesUp(t *testing.T) {
	t.Parallel()
	health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	var opened []string
	var waiting *hanging
	open := func(ctx context.Context, candidate string) (core.StreamIter, error) {
		opened = append(opened, candidate)
		waiting = &hanging{ctx: ctx, closed: make(chan struct{})}
		return waiting, nil
	}
	executor := execution.Executor[string]{Health: health, Key: identity}
	_, err := execution.ExecuteStream(ctx, executor, []string{"a", "b"}, open, chatOutput)
	if !errors.Is(err, context.DeadlineExceeded) || !slices.Equal(opened, []string{"a"}) {
		t.Fatalf("err=%v opened=%v", err, opened)
	}
	select {
	case <-waiting.closed:
	default:
		t.Fatal("the abandoned stream was not closed")
	}
	assertAvailable(t, health, "a")

	// A caller that leaves between frames stops the execution too.
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	leaving := &stream{frames: []string{keepalive, keepalive}}
	output := func([]byte) bool {
		cancel()
		return false
	}
	upstream := &upstreams{streams: map[string]*stream{"a": leaving}}
	if _, err := execution.ExecuteStream(ctx, execution.Executor[string]{}, []string{"a"}, upstream.open, output); !errors.Is(err, context.Canceled) || leaving.closed() != 1 {
		t.Fatalf("err=%v closes=%d", err, leaving.closed())
	}
}

func TestExecuteStreamClosesAStreamOpenedWithAnError(t *testing.T) {
	t.Parallel()
	leaked := &stream{}
	open := func(_ context.Context, candidate string) (core.StreamIter, error) {
		if candidate == "a" {
			return leaked, overloaded()
		}
		return &stream{frames: []string{done}}, nil
	}
	result, err := execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"a", "b"}, open, chatOutput)
	if err != nil || result.Candidate != "b" || leaked.closed() != 1 {
		t.Fatalf("result=%+v err=%v closes=%d", result, err, leaked.closed())
	}
}

func TestExecuteStreamRejectsAnOpenWithoutAStream(t *testing.T) {
	t.Parallel()
	open := func(context.Context, string) (core.StreamIter, error) { return nil, nil }
	result, err := execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"a", "b"}, open, chatOutput)
	if err == nil || len(result.Attempts) != 1 || result.Attempts[0].Disposition != core.DispositionTerminal {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestExecuteStreamRepeatsARetryableOpen(t *testing.T) {
	t.Parallel()
	failures := 0
	open := func(context.Context, string) (core.StreamIter, error) {
		if failures == 0 {
			failures++
			return nil, overloaded()
		}
		return &stream{frames: []string{chunk("model", `{"content":"served"}`, "null", "")}}, nil
	}
	executor := execution.Executor[string]{Retry: execution.Retry{Attempts: 2}}
	result, err := execution.ExecuteStream(context.Background(), executor, []string{"a", "b"}, open, chatOutput)
	if err != nil || result.Candidate != "a" || result.Attempts[0].Tries != 2 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestExecuteStreamReportsTheLossesOfAStreamThatTranslates(t *testing.T) {
	t.Parallel()
	losses := []core.Loss{{Path: "temperature"}}
	open := func(context.Context, string) (core.StreamIter, error) {
		return lossyStream{stream: &stream{frames: []string{chunk("model", `{"content":"x"}`, "null", "")}}, losses: losses}, nil
	}
	result, err := execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"a"}, open, chatOutput)
	if err != nil {
		t.Fatal(err)
	}
	if got := core.StreamLosses(result.Value); len(got) != 1 || got[0].Path != "temperature" {
		t.Fatalf("losses=%+v", got)
	}
}

func TestExecuteStreamClosesTheCommittedStreamOnce(t *testing.T) {
	t.Parallel()
	inner := &stream{frames: []string{chunk("model", `{"content":"x"}`, "null", "")}}
	open := func(context.Context, string) (core.StreamIter, error) { return inner, nil }
	result, err := execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"a"}, open, chatOutput)
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err := result.Value.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if inner.closed() != 1 {
		t.Fatalf("closes=%d, want one", inner.closed())
	}
}

func TestExecuteStreamDoesNotWrapAnAfterOutputErrorTwice(t *testing.T) {
	t.Parallel()
	cause := &execution.AfterOutputError{Err: disconnected()}
	inner := &stream{frames: []string{chunk("model", `{"content":"x"}`, "null", "")}, err: cause}
	open := func(context.Context, string) (core.StreamIter, error) { return inner, nil }
	result, err := execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"a"}, open, chatOutput)
	if err != nil {
		t.Fatal(err)
	}
	if _, end := drain(t, result.Value); end != cause {
		t.Fatalf("end=%v, want the stream's own AfterOutputError", end)
	}
}

func TestAfterOutputError(t *testing.T) {
	t.Parallel()
	cause := overloaded()
	err := &execution.AfterOutputError{Err: cause}
	if err.Error() != cause.Error() || !errors.Is(err, cause) {
		t.Fatalf("message=%q is=%v", err.Error(), errors.Is(err, cause))
	}
	// Its empty classification shadows the retryable one it wraps.
	if classification := core.ClassifyError(err); classification != (core.ProviderErrorClassification{}) {
		t.Fatalf("classification=%+v", classification)
	}
	var missing *execution.AfterOutputError
	if missing.Error() == "" || missing.Unwrap() != nil || (&execution.AfterOutputError{}).Error() == "" {
		t.Fatal("an empty AfterOutputError is not safe to use")
	}
}
