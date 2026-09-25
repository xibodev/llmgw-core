package providers

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// maxStreamRecordWireSize bounds all the wire bytes of one stream record,
// its ignored fields and blank delimiter included, as the gateway bounds
// them.
const maxStreamRecordWireSize = 4 << 20

// streamRecordTooLargeError describes an oversized stream record without
// keeping any of it.
type streamRecordTooLargeError struct {
	format string
	limit  int
}

func (e *streamRecordTooLargeError) Error() string {
	return fmt.Sprintf("upstream %s record exceeds %d-byte wire limit", e.format, e.limit)
}

// readBoundedLine reads one line of at most limit bytes, its newline
// included. A longer line fails with a *streamRecordTooLargeError naming
// format, such as SSE or NDJSON, before more of it is held than the reader
// buffers.
func readBoundedLine(reader *bufio.Reader, limit int, format string) ([]byte, error) {
	var line []byte
	for {
		fragment, err := reader.ReadSlice('\n')
		if len(fragment) > limit-len(line) {
			return nil, &streamRecordTooLargeError{format: format, limit: maxStreamRecordWireSize}
		}
		line = append(line, fragment...)
		if err != bufio.ErrBufferFull {
			return line, err
		}
	}
}

// sseRecord is one SSE record that carries data: its bytes as the upstream
// sent them, and its data lines joined.
type sseRecord struct {
	frame []byte
	data  string
}

// sseRecordReader reads an SSE stream as the gateway reads it. A record
// without data, such as a comment or a keepalive, is skipped, as is one
// whose data is empty. A record is bounded to maxStreamRecordWireSize. A
// last record without its blank line still counts, and its frame gets the
// line ending it lacks, so every frame is a complete record.
type sseRecordReader struct{ reader *bufio.Reader }

func newSSERecordReader(body io.Reader) *sseRecordReader {
	return &sseRecordReader{reader: bufio.NewReader(body)}
}

// Next returns the next record that carries data, [DONE] included, or
// io.EOF once the stream ends.
func (r *sseRecordReader) Next() (sseRecord, error) {
	var frame []byte
	var data []string
	for {
		line, err := readBoundedLine(r.reader, maxStreamRecordWireSize-len(frame), "SSE")
		if err != nil && err != io.EOF {
			return sseRecord{}, err
		}
		frame = append(frame, line...)
		if content, ended := strings.CutSuffix(string(line), "\n"); len(line) > 0 {
			if ended {
				content = strings.TrimSuffix(content, "\r")
			}
			switch {
			case content == "":
				if payload := strings.Join(data, "\n"); payload != "" {
					return sseRecord{frame: frame, data: payload}, nil
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
				return sseRecord{frame: append(frame, '\n'), data: payload}, nil
			}
			return sseRecord{}, io.EOF
		}
	}
}

// NextData returns the data of the next record other than [DONE], as the
// gateway returns the events of a stream it parses rather than relays, or
// io.EOF once the stream ends.
func (r *sseRecordReader) NextData() (string, error) {
	for {
		record, err := r.Next()
		if err != nil || record.data != "[DONE]" {
			return record.data, err
		}
	}
}

// sseFrameStream relays an SSE stream's records byte for byte, [DONE]
// included, until the upstream ends it.
type sseFrameStream struct {
	ctx    context.Context
	label  string
	body   io.ReadCloser
	reader *sseRecordReader
}

var _ core.StreamIter = (*sseFrameStream)(nil)

func newSSEFrameStream(ctx context.Context, label string, body io.ReadCloser) *sseFrameStream {
	return &sseFrameStream{ctx: ctx, label: label, body: body, reader: newSSERecordReader(body)}
}

func (s *sseFrameStream) Next() ([]byte, error) {
	record, err := s.reader.Next()
	if err == io.EOF {
		return nil, io.EOF
	}
	if err != nil {
		return nil, streamFailure(s.ctx, s.label, err)
	}
	return record.frame, nil
}

func (s *sseFrameStream) Close() error { return s.body.Close() }

// streamFailure reports a stream that broke after it opened, classified as
// the gateway classifies it. An oversized record ends the request: the
// upstream sent what will not be read, and nothing can serve the rest.
// Any other failure is the transport's, which another target may get
// past, unless the caller gave up.
func streamFailure(ctx context.Context, label string, err error) *core.ProviderError {
	var tooLarge *streamRecordTooLargeError
	if errors.As(err, &tooLarge) {
		return &core.ProviderError{
			Message: "the " + label + " stream sent a record over the size limit", Class: core.ProviderErrorUpstream, Cause: err,
		}
	}
	failure := &core.ProviderError{
		Message: "the " + label + " stream broke off", Class: core.ProviderErrorTransport, Cause: err,
		Classification: core.ProviderErrorClassification{FailoverEligible: true},
	}
	if ctx.Err() != nil {
		failure.Classification = core.ProviderErrorClassification{}
	}
	return failure
}
