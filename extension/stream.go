package extension

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"slices"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// maxStreamRecord bounds all the wire bytes of one stream record, its blank
// delimiter included, as core's providers bound them.
const maxStreamRecord = 4 << 20

var errRecordTooLarge = errors.New("extension: stream record exceeds its size limit")

// stream relays the SSE records of a stream answer that carry data, byte for
// byte, [DONE] included. A record without data, such as a comment or a
// keepalive, is skipped, as is one whose data is empty. A last record without
// its blank line still counts, and gets the line endings it lacks, so every
// frame is a complete record.
type stream struct {
	ctx      context.Context
	provider string
	body     io.ReadCloser
	reader   *bufio.Reader
	losses   []core.Loss
}

var (
	_ core.StreamIter   = (*stream)(nil)
	_ core.LossReporter = (*stream)(nil)
)

func newStream(ctx context.Context, provider string, response *http.Response) *stream {
	return &stream{
		ctx:      ctx,
		provider: provider,
		body:     response.Body,
		reader:   bufio.NewReader(response.Body),
		losses:   LossesFromHeader(response.Header),
	}
}

// Next returns the next record that carries data, or io.EOF once the stream
// ends.
func (s *stream) Next() ([]byte, error) {
	var frame []byte
	var data []string
	for {
		line, err := readBoundedLine(s.reader, maxStreamRecord-len(frame))
		if err != nil && err != io.EOF {
			return nil, s.failure(err)
		}
		frame = append(frame, line...)
		if content, ended := strings.CutSuffix(string(line), "\n"); len(line) > 0 {
			if ended {
				content = strings.TrimSuffix(content, "\r")
			}
			switch {
			case content == "":
				if strings.Join(data, "\n") != "" {
					return frame, nil
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
			if strings.Join(data, "\n") != "" {
				if !bytes.HasSuffix(frame, []byte("\n")) {
					frame = append(frame, '\n')
				}
				return append(frame, '\n'), nil
			}
			return nil, io.EOF
		}
	}
}

// Close releases the answer.
func (s *stream) Close() error { return s.body.Close() }

// Losses returns the losses the daemon put on its answer.
func (s *stream) Losses() []core.Loss { return slices.Clone(s.losses) }

// failure reports a stream that broke after it opened. An oversized record
// ends the request; any other failure is the transport's, which another
// target may get past, unless the caller gave up.
func (s *stream) failure(err error) error {
	if errors.Is(err, errRecordTooLarge) {
		return &core.ProviderError{
			Message: "the " + label(s.provider, "stream") + " sent a record over the size limit",
			Class:   core.ProviderErrorUpstream,
			Cause:   err,
		}
	}
	failure := &core.ProviderError{
		Message:        "the " + label(s.provider, "stream") + " broke off",
		Class:          core.ProviderErrorTransport,
		Classification: core.ProviderErrorClassification{FailoverEligible: true},
		Cause:          err,
	}
	if s.ctx.Err() != nil {
		failure.Classification = core.ProviderErrorClassification{}
	}
	return failure
}

// readBoundedLine reads one line of at most limit bytes, its newline
// included, and fails before it holds more of a longer line than the reader
// buffers.
func readBoundedLine(reader *bufio.Reader, limit int) ([]byte, error) {
	var line []byte
	for {
		fragment, err := reader.ReadSlice('\n')
		if len(fragment) > limit-len(line) {
			return nil, errRecordTooLarge
		}
		line = append(line, fragment...)
		if err != bufio.ErrBufferFull {
			return line, err
		}
	}
}
