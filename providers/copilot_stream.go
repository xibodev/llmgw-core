package providers

import (
	"bufio"
	"context"
	"errors"
	"io"
	"slices"

	core "github.com/xibodev/llmgw-core"
)

// copilotMaxRecordBytes bounds one SSE record on the wire, blank delimiter
// and ignored fields included, as the gateway bounds it.
const copilotMaxRecordBytes = 4 << 20

var errCopilotRecordTooLarge = errors.New("a Copilot SSE record exceeds its size limit")

// copilotStream yields Copilot's SSE records as sent, one complete record
// per frame, and reports the request's losses through core.LossReporter.
// Blank lines between records are skipped. A record the stream ends without
// terminating is completed, because the gateway still delivers its data.
type copilotStream struct {
	ctx    context.Context
	body   io.ReadCloser
	reader *bufio.Reader
	losses []core.Loss
}

var _ core.LossReporter = (*copilotStream)(nil)

func newCopilotStream(ctx context.Context, body io.ReadCloser, losses []core.Loss) *copilotStream {
	return &copilotStream{ctx: ctx, body: body, reader: bufio.NewReader(body), losses: losses}
}

func (s *copilotStream) Next() ([]byte, error) {
	var frame []byte
	for {
		line, err := copilotLine(s.reader, copilotMaxRecordBytes-len(frame))
		if errors.Is(err, errCopilotRecordTooLarge) {
			// The gateway ends the stream there; nothing can serve its rest.
			return nil, &core.ProviderError{Message: err.Error(), Class: core.ProviderErrorUpstream, Cause: err}
		}
		if err != nil && err != io.EOF {
			failure := &core.ProviderError{
				Message: "the Copilot stream broke off", Class: core.ProviderErrorTransport,
				Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
			}
			if callerCancellation(err) && s.ctx.Err() != nil {
				failure.Classification = core.ProviderErrorClassification{}
			}
			return nil, failure
		}
		switch {
		case len(line) == 0:
		case string(line) != "\n" && string(line) != "\r\n":
			frame = append(frame, line...)
		case len(frame) > 0:
			return append(frame, line...), nil
		}
		if err == io.EOF {
			if len(frame) == 0 {
				return nil, io.EOF
			}
			if frame[len(frame)-1] != '\n' {
				frame = append(frame, '\n')
			}
			return append(frame, '\n'), nil
		}
	}
}

func (s *copilotStream) Close() error { return s.body.Close() }

// Losses implements core.LossReporter. They are known before the first
// frame.
func (s *copilotStream) Losses() []core.Loss { return slices.Clone(s.losses) }

// copilotLine reads one line of at most limit bytes, its newline included.
func copilotLine(reader *bufio.Reader, limit int) ([]byte, error) {
	var line []byte
	for {
		fragment, err := reader.ReadSlice('\n')
		if len(fragment) > limit-len(line) {
			return nil, errCopilotRecordTooLarge
		}
		line = append(line, fragment...)
		if err != bufio.ErrBufferFull {
			return line, err
		}
	}
}

// copilotFrames is a stream whose frames are at hand: a Responses answer
// rendered as Chat chunks, the way the gateway streams Chat it served over
// Responses.
type copilotFrames struct {
	frames [][]byte
	losses []core.Loss
}

var _ core.LossReporter = (*copilotFrames)(nil)

func (s *copilotFrames) Next() ([]byte, error) {
	if len(s.frames) == 0 {
		return nil, io.EOF
	}
	frame := s.frames[0]
	s.frames = s.frames[1:]
	return frame, nil
}

func (s *copilotFrames) Close() error {
	s.frames = nil
	return nil
}

// Losses implements core.LossReporter.
func (s *copilotFrames) Losses() []core.Loss { return slices.Clone(s.losses) }
