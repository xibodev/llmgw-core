package providers

import (
	"context"
	"io"

	core "github.com/xibodev/llmgw-core"
)

// StreamIter yields tokens or error chunks from an active stream.
type StreamIter interface {
	Next() ([]byte, error)
	Close() error
}

// Provider represents a backend LLM client.
type Provider interface {
	// Complete executes a non-streaming chat completion.
	Complete(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (map[string]any, error)

	// Stream starts a streaming chat completion, yielding SSE/delta bytes.
	Stream(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (StreamIter, error)

	// ListModels queries the provider's active catalog.
	ListModels(ctx context.Context, cred *core.Credential) ([]core.ModelInfo, error)
}

// ByteStreamIter wraps an io.ReadCloser into a StreamIter.
type ByteStreamIter struct {
	reader io.ReadCloser
	buf    []byte
}

func NewByteStreamIter(r io.ReadCloser) *ByteStreamIter {
	return &ByteStreamIter{
		reader: r,
		buf:    make([]byte, 4096),
	}
}

func (s *ByteStreamIter) Next() ([]byte, error) {
	n, err := s.reader.Read(s.buf)
	if n > 0 {
		out := make([]byte, n)
		copy(out, s.buf[:n])
		return out, nil
	}
	return nil, err
}

func (s *ByteStreamIter) Close() error {
	if s.reader != nil {
		return s.reader.Close()
	}
	return nil
}
