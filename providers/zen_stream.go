package providers

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"slices"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

var errZenRecordTooLarge = errors.New("an OpenCode Zen stream record exceeds 4 MiB")

// zenRecord is one SSE record that carries data: its bytes as Zen sent
// them, and its data lines joined.
type zenRecord struct {
	frame []byte
	data  string
}

// zenSSEReader reads Zen's streams as the gateway reads them. A record
// without data, such as a comment or a keepalive, is skipped, as is one
// whose data is empty. A record is bounded to 4 MiB of wire bytes, its
// ignored fields and blank line included. A last record without its blank
// line still counts, and its frame gets the line ending it lacks, so every
// frame is a complete record.
type zenSSEReader struct{ reader *bufio.Reader }

func newZenSSEReader(body io.Reader) *zenSSEReader {
	return &zenSSEReader{reader: bufio.NewReader(body)}
}

func (r *zenSSEReader) Next() (zenRecord, error) {
	var frame []byte
	var data []string
	for {
		line, err := r.line(zenMaxRecordBytes - len(frame))
		if err != nil && err != io.EOF {
			return zenRecord{}, err
		}
		frame = append(frame, line...)
		if content, ended := strings.CutSuffix(string(line), "\n"); len(line) > 0 {
			if ended {
				content = strings.TrimSuffix(content, "\r")
			}
			switch {
			case content == "":
				if payload := strings.Join(data, "\n"); payload != "" {
					return zenRecord{frame: frame, data: payload}, nil
				}
				frame, data = nil, nil
			case content[0] != ':':
				field, value, _ := strings.Cut(content, ":")
				if field == "data" {
					data = append(data, strings.TrimPrefix(value, " "))
				}
			}
		}
		if err == io.EOF {
			if payload := strings.Join(data, "\n"); payload != "" {
				if !strings.HasSuffix(string(frame), "\n") {
					frame = append(frame, '\n')
				}
				return zenRecord{frame: append(frame, '\n'), data: payload}, nil
			}
			return zenRecord{}, io.EOF
		}
	}
}

// line reads one line of at most limit bytes.
func (r *zenSSEReader) line(limit int) ([]byte, error) {
	var line []byte
	for {
		fragment, err := r.reader.ReadSlice('\n')
		if len(fragment) > limit-len(line) {
			return nil, errZenRecordTooLarge
		}
		line = append(line, fragment...)
		if err != bufio.ErrBufferFull {
			return line, err
		}
	}
}

// zenStreamFailure reports a stream that broke after it opened.
func zenStreamFailure(ctx context.Context, err error) error {
	if errors.Is(err, errZenRecordTooLarge) {
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
	reader *zenSSEReader
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
	reader *zenSSEReader
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
