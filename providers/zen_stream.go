package providers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"slices"

	core "github.com/xibodev/llmgw-core"
)

// zenStreamFailure reports a stream that broke after it opened. Zen's
// streams are read with sseRecordReader, as the gateway reads them.
func zenStreamFailure(ctx context.Context, err error) error {
	var tooLarge *streamRecordTooLargeError
	if errors.As(err, &tooLarge) {
		return zenUpstreamError("an OpenCode Zen stream record exceeds the size limit", false, err)
	}
	return zenFailure(ctx, zenTransportError(ctx, "the OpenCode Zen stream failed", err))
}

// zenStream reports the request's losses through core.LossReporter.
type zenStream struct {
	events core.StreamIter
	losses []core.Loss
}

var _ core.LossReporter = (*zenStream)(nil)

func (s *zenStream) Next() ([]byte, error) { return s.events.Next() }
func (s *zenStream) Close() error          { return s.events.Close() }

// Losses implements core.LossReporter. They are known before the first
// frame.
func (s *zenStream) Losses() []core.Loss { return slices.Clone(s.losses) }

// zenChatStream passes every Chat record through byte for byte, [DONE]
// included, until Zen closes the stream.
type zenChatStream struct {
	ctx    context.Context
	body   io.ReadCloser
	reader *sseRecordReader
}

func (s *zenChatStream) Next() ([]byte, error) {
	record, err := s.reader.Next()
	if err == io.EOF {
		return nil, io.EOF
	}
	if err != nil {
		return nil, zenStreamFailure(s.ctx, err)
	}
	return record.frame, nil
}

func (s *zenChatStream) Close() error { return s.body.Close() }

// zenResponsesStream passes Responses events through as the gateway relays
// them: each record whose data is a JSON object with a type, byte for byte,
// up to and including response.completed, response.incomplete or
// response.failed. Any other record is skipped, and an error event passes
// through without ending the stream. A stream that ends before its
// terminal event fails, where the gateway writes response.failed itself.
type zenResponsesStream struct {
	ctx    context.Context
	body   io.ReadCloser
	reader *sseRecordReader
	done   bool
}

func (s *zenResponsesStream) Next() ([]byte, error) {
	for !s.done {
		record, err := s.reader.Next()
		if err == io.EOF {
			return nil, zenUpstreamError("the OpenCode Zen stream ended without a terminal event", false, nil)
		}
		if err != nil {
			return nil, zenStreamFailure(s.ctx, err)
		}
		var event struct {
			Type string `json:"type"`
		}
		if json.Unmarshal([]byte(record.data), &event) != nil || event.Type == "" {
			continue
		}
		switch event.Type {
		case "response.completed", "response.incomplete", "response.failed":
			s.done = true
		}
		return record.frame, nil
	}
	return nil, io.EOF
}

func (s *zenResponsesStream) Close() error { return s.body.Close() }

// zenFrames replays frames already rendered.
type zenFrames struct{ frames [][]byte }

func (s *zenFrames) Next() ([]byte, error) {
	if len(s.frames) == 0 {
		return nil, io.EOF
	}
	frame := s.frames[0]
	s.frames = s.frames[1:]
	return frame, nil
}

func (s *zenFrames) Close() error { return nil }
