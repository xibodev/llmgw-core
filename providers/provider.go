package providers

import (
	"bufio"
	"context"
	"fmt"
	"io"

	core "github.com/xibodev/llmgw-core"
)

// StreamIter is the shared core stream contract.
type StreamIter = core.StreamIter

// Provider represents a backend LLM client.
type Provider interface {
	// Complete executes a non-streaming chat completion.
	Complete(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (map[string]any, error)

	// Stream starts a streaming chat completion, yielding complete SSE frames.
	Stream(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (StreamIter, error)

	// ListModels queries the provider's active catalog.
	ListModels(ctx context.Context, cred *core.Credential) ([]core.ModelInfo, error)
}

// ResponsesProvider is the optional native, non-streaming Responses surface.
// Callers may use it after PlanTransport selects ModelSurfaceResponses without
// depending on a concrete provider implementation.
type ResponsesProvider interface {
	CompleteResponses(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (map[string]any, error)
}

// ResponsesStreamProvider is the optional native streaming Responses surface.
// Returned chunks are complete SSE frames in the upstream wire format.
type ResponsesStreamProvider interface {
	StreamResponses(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (StreamIter, error)
}

const maxSSEFrameSize = 1 << 20

// ByteStreamIter frames an HTTP response body into complete, bounded SSE records.
type ByteStreamIter struct {
	reader     io.ReadCloser
	scanner    *bufio.Scanner
	ctx        context.Context
	incomplete bool
}

func NewByteStreamIter(r io.ReadCloser) *ByteStreamIter {
	return newByteStreamIter(context.Background(), r)
}

func newByteStreamIter(ctx context.Context, r io.ReadCloser) *ByteStreamIter {
	s := &ByteStreamIter{reader: r, ctx: ctx}
	s.scanner = bufio.NewScanner(r)
	s.scanner.Buffer(make([]byte, 4096), maxSSEFrameSize)
	s.scanner.Split(s.splitFrame)
	return s
}

func (s *ByteStreamIter) Next() ([]byte, error) {
	if s.scanner.Scan() {
		if s.incomplete {
			return nil, invocationError(s.ctx, "upstream SSE stream ended with an incomplete record", 0, io.ErrUnexpectedEOF)
		}
		return bytesClone(s.scanner.Bytes()), nil
	}
	if err := s.scanner.Err(); err != nil {
		return nil, invocationError(s.ctx, fmt.Sprintf("read upstream SSE stream failed (records are limited to %d bytes)", maxSSEFrameSize), 0, err)
	}
	return nil, io.EOF
}

func (s *ByteStreamIter) splitFrame(data []byte, atEOF bool) (advance int, token []byte, err error) {
	for i := 0; i < len(data); i++ {
		if data[i] != '\n' {
			continue
		}
		if i+1 < len(data) && data[i+1] == '\n' {
			return i + 2, data[:i+2], nil
		}
		if i >= 1 && data[i-1] == '\r' && i+2 < len(data) && data[i+1] == '\r' && data[i+2] == '\n' {
			return i + 3, data[:i+3], nil
		}
	}
	if atEOF && len(data) > 0 {
		s.incomplete = true
		return len(data), data, nil
	}
	return 0, nil, nil
}

func bytesClone(data []byte) []byte {
	out := make([]byte, len(data))
	copy(out, data)
	return out
}

func (s *ByteStreamIter) Close() error {
	if s.reader != nil {
		return s.reader.Close()
	}
	return nil
}
