package providers

import (
	"bufio"
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// Ported from the gateway's httpstream_test.go. NextData reads events as
// the gateway's reader does; Next keeps each record's bytes and [DONE].
func TestSSERecordReaderParsesLogicalEvents(t *testing.T) {
	t.Parallel()
	stream := strings.Join([]string{
		": comment\r",
		"event: message\r",
		"id: 1\r",
		"retry: 100\r",
		"unknown: ignored\r",
		"data:first\r",
		"data: second\r",
		"data:  two spaces remain\r",
		"\r",
		"data:\r",
		"\r",
		"event: no-data\r",
		"\r",
		"data: [DONE]\r",
		"\r",
		"data: final",
	}, "\n")
	reader := newSSERecordReader(strings.NewReader(stream))
	if got, err := reader.NextData(); err != nil || got != "first\nsecond\n two spaces remain" {
		t.Fatalf("first event = %q, %v", got, err)
	}
	if got, err := reader.NextData(); err != nil || got != "final" {
		t.Fatalf("EOF event = %q, %v", got, err)
	}
	if got, err := reader.NextData(); err != io.EOF || got != "" {
		t.Fatalf("end = %q, %v", got, err)
	}

	reader = newSSERecordReader(strings.NewReader(stream))
	for _, want := range []sseRecord{
		{frame: []byte(strings.Join(strings.Split(stream, "\n")[:9], "\n") + "\n"), data: "first\nsecond\n two spaces remain"},
		{frame: []byte("data: [DONE]\r\n\r\n"), data: "[DONE]"},
		{frame: []byte("data: final\n\n"), data: "final"},
	} {
		if got, err := reader.Next(); err != nil || string(got.frame) != string(want.frame) || got.data != want.data {
			t.Fatalf("record = %q %q, %v; want %q %q", got.frame, got.data, err, want.frame, want.data)
		}
	}
	if _, err := reader.Next(); err != io.EOF {
		t.Fatalf("end = %v, want io.EOF", err)
	}
}

// Ported from the gateway's httpstream_test.go.
func TestSSERecordReaderWireLimit(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name      string
		extraByte int
		wantOK    bool
	}{
		{name: "exact limit", wantOK: true},
		{name: "limit plus one", extraByte: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload := strings.Repeat("x", maxStreamRecordWireSize-len("data: \n\n")+tc.extraByte)
			reader := newSSERecordReader(strings.NewReader("data: " + payload + "\n\n"))
			got, err := reader.NextData()
			if (err == nil) != tc.wantOK {
				t.Fatalf("NextData err = %v, want ok %v", err, tc.wantOK)
			}
			if tc.wantOK && got != payload {
				t.Fatalf("payload length = %d, want %d", len(got), len(payload))
			}
			if !tc.wantOK {
				var sizeErr *streamRecordTooLargeError
				if !errors.As(err, &sizeErr) || sizeErr.format != "SSE" || sizeErr.limit != maxStreamRecordWireSize {
					t.Fatalf("error = %#v", err)
				}
				if strings.Contains(sizeErr.Error(), payload[:32]) {
					t.Fatalf("error exposes payload: %q", sizeErr)
				}
			}
		})
	}
}

// Ported from the gateway's httpstream_test.go.
func TestSSERecordReaderCountsIgnoredWireLines(t *testing.T) {
	t.Parallel()
	reader := newSSERecordReader(io.MultiReader(
		strings.NewReader(": "+strings.Repeat("x", maxStreamRecordWireSize-3)+"\n"),
		strings.NewReader("data: exposed\n\n"),
	))
	var sizeErr *streamRecordTooLargeError
	if _, err := reader.Next(); !errors.As(err, &sizeErr) {
		t.Fatalf("oversized ignored line: error = %v", err)
	}
}

func TestReadBoundedLineNamesItsFormat(t *testing.T) {
	t.Parallel()
	reader := bufio.NewReader(strings.NewReader(`{"done":false}` + "\n" + strings.Repeat("x", 64) + "\n"))
	if line, err := readBoundedLine(reader, 64, "NDJSON"); err != nil || string(line) != `{"done":false}`+"\n" {
		t.Fatalf("line = %q, %v", line, err)
	}
	var sizeErr *streamRecordTooLargeError
	if _, err := readBoundedLine(reader, 64, "NDJSON"); !errors.As(err, &sizeErr) || sizeErr.format != "NDJSON" {
		t.Fatalf("an NDJSON line over the limit: error = %v", err)
	}
}

// closeRecorder is a body that reports its Close.
type closeRecorder struct {
	io.Reader
	closed bool
}

func (b *closeRecorder) Close() error {
	b.closed = true
	return nil
}

// Ported from the gateway's TestHTTPStreamIterReturnsCompleteRecordsAndParserErrors.
// Records pass through byte for byte, and an oversized one ends the request.
func TestSSEFrameStreamRelaysRecordsAndEndsAtAnOversizedOne(t *testing.T) {
	t.Parallel()
	body := &closeRecorder{Reader: strings.NewReader(
		": keepalive\n\ndata: one\ndata: two\n\ndata: [DONE]\n\ndata: " + strings.Repeat("x", maxStreamRecordWireSize) + "\n\n",
	)}
	stream := newSSEFrameStream(context.Background(), "Fixture", body)
	for _, want := range []string{"data: one\ndata: two\n\n", "data: [DONE]\n\n"} {
		if frame, err := stream.Next(); err != nil || string(frame) != want {
			t.Fatalf("frame = %q, %v; want %q", frame, err, want)
		}
	}
	_, err := stream.Next()
	var failure *core.ProviderError
	var sizeErr *streamRecordTooLargeError
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorUpstream || !errors.As(err, &sizeErr) ||
		core.ClassifyError(err).Disposition() != core.DispositionTerminal ||
		failure.Error() != "the Fixture stream sent a record over the size limit" {
		t.Fatalf("oversized record: error = %v, classification = %+v", err, core.ClassifyError(err))
	}
	if err := stream.Close(); err != nil || !body.closed {
		t.Fatalf("Close = %v, closed = %v", err, body.closed)
	}
}

func TestSSEFrameStreamEndsWithTheUpstream(t *testing.T) {
	t.Parallel()
	stream := newSSEFrameStream(context.Background(), "Fixture", io.NopCloser(strings.NewReader("data: last")))
	if frame, err := stream.Next(); err != nil || string(frame) != "data: last\n\n" {
		t.Fatalf("frame = %q, %v", frame, err)
	}
	if _, err := stream.Next(); err != io.EOF {
		t.Fatalf("end = %v, want io.EOF", err)
	}
}

// A stream that breaks off may fail over to a target that has not been
// asked yet, as in the gateway, but is never repeated, and permits nothing
// once the caller gave up.
func TestSSEFrameStreamTransportFailure(t *testing.T) {
	t.Parallel()
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	for _, check := range []struct {
		ctx  context.Context
		want core.ProviderErrorClassification
	}{
		{context.Background(), core.ProviderErrorClassification{FailoverEligible: true}},
		{canceled, core.ProviderErrorClassification{}},
	} {
		stream := newSSEFrameStream(check.ctx, "Fixture", io.NopCloser(io.MultiReader(
			strings.NewReader("data: one\n\n"), &partialErrorReader{delivered: true},
		)))
		if _, err := stream.Next(); err != nil {
			t.Fatal(err)
		}
		_, err := stream.Next()
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorTransport || !errors.Is(err, io.ErrUnexpectedEOF) ||
			failure.Classification != check.want || failure.Error() != "the Fixture stream broke off" {
			t.Fatalf("error = %v, classification = %+v, want %+v", err, core.ClassifyError(err), check.want)
		}
	}
}
