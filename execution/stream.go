package execution

import (
	"bytes"
	"context"
	"errors"
	"io"
	"sync"

	core "github.com/xibodev/llmgw-core"
)

var errNoStream = errors.New("execution: open returned no stream")

// ExecuteStream is Execute for streams: it opens candidates in order, and a
// candidate serves once its stream carries output.
//
// output reports whether a frame carries output the caller would see,
// content or reasoning; keepalives and role-only frames carry none. Nil
// counts every frame. Frames before the first that carries output are held
// back, and a failure while they are held is the candidate's failure: its
// stream is closed and execution goes on as for Execute. The frame that
// carries output commits: the returned stream replays the held frames, then
// yields the rest, and nothing fails over any more. A later failure reaches
// the caller as the stream's error, an *AfterOutputError, and is not
// recorded. A stream that ends before any output commits too, so an empty
// answer is delivered rather than retried elsewhere.
//
// Health records a success when a stream commits. The returned stream
// reports the losses of a stream that translates, and the caller closes it.
func ExecuteStream[C any](ctx context.Context, executor Executor[C], candidates []C, open func(context.Context, C) (core.StreamIter, error), output func(frame []byte) bool) (Result[C, core.StreamIter], error) {
	return Execute(ctx, executor, candidates, func(ctx context.Context, candidate C) (core.StreamIter, error) {
		stream, err := open(ctx, candidate)
		if err != nil {
			if stream != nil {
				_ = stream.Close()
			}
			return nil, err
		}
		if stream == nil {
			return nil, errNoStream
		}
		return commit(ctx, stream, output)
	})
}

// commit reads stream until a frame carries output or the stream ends, and
// returns a stream that replays what it read. It closes a stream that fails
// before that.
func commit(ctx context.Context, stream core.StreamIter, output func([]byte) bool) (core.StreamIter, error) {
	var held [][]byte
	for {
		if err := ctx.Err(); err != nil {
			_ = stream.Close()
			return nil, err
		}
		frame, err := stream.Next()
		switch {
		case err == nil:
			if len(frame) == 0 {
				continue
			}
			// A stream may reuse its buffer on the next read.
			frame = bytes.Clone(frame)
			held = append(held, frame)
			if output == nil || output(frame) {
				return &committed{held: held, stream: stream}, nil
			}
		case errors.Is(err, io.EOF):
			if len(frame) > 0 {
				held = append(held, bytes.Clone(frame))
			}
			return &committed{held: held, stream: stream, end: err}, nil
		default:
			// Nothing has reached the caller, not even a frame that came with
			// the failure, so the next candidate may still serve.
			_ = stream.Close()
			return nil, err
		}
	}
}

// committed is a stream that has served output: it replays the frames held
// before the output, then reads on.
type committed struct {
	held   [][]byte
	stream core.StreamIter
	// end is how the stream ended, when that happened before any output.
	end error

	closeOnce sync.Once
	closeErr  error
}

func (s *committed) Next() ([]byte, error) {
	if len(s.held) > 0 {
		frame := s.held[0]
		s.held[0] = nil
		s.held = s.held[1:]
		return frame, nil
	}
	if s.end != nil {
		return nil, s.end
	}
	frame, err := s.stream.Next()
	if err != nil && !errors.Is(err, io.EOF) {
		var afterOutput *AfterOutputError
		if !errors.As(err, &afterOutput) {
			err = &AfterOutputError{Err: err}
		}
	}
	return frame, err
}

func (s *committed) Close() error {
	s.closeOnce.Do(func() { s.closeErr = s.stream.Close() })
	return s.closeErr
}

// Losses implements core.LossReporter for a stream that translates.
func (s *committed) Losses() []core.Loss {
	return core.StreamLosses(s.stream)
}

// AfterOutputError is a stream failure after output reached the caller.
//
// Its classification is empty, so it is terminal and neutral for health
// whatever Err says: no candidate can take back output the caller has seen,
// so neither an execution nor a product's own retry may repeat the request.
// A product whose streaming pushes output instead of returning a stream
// returns one from Execute's run, to stop there.
type AfterOutputError struct {
	// Err is the stream's own error, for errors.Is and errors.As.
	Err error
}

// Error is Err's message.
func (e *AfterOutputError) Error() string {
	if e == nil || e.Err == nil {
		return "stream failed after output"
	}
	return e.Err.Error()
}

// Unwrap returns Err.
func (e *AfterOutputError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// ProviderErrorClassification is empty, shadowing Err's: terminal, and not
// a circuit failure.
func (e *AfterOutputError) ProviderErrorClassification() core.ProviderErrorClassification {
	return core.ProviderErrorClassification{}
}
