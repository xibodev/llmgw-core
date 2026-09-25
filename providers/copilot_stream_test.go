package providers

import (
	"context"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// drainCopilot reads a stream to its end.
func drainCopilot(t *testing.T, stream core.StreamIter) []string {
	t.Helper()
	defer stream.Close()
	var frames []string
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			return frames
		}
		if err != nil {
			t.Fatal(err)
		}
		frames = append(frames, string(frame))
	}
}

// A Chat stream is Copilot's records as sent, whatever their line endings,
// fields and comments. Blank lines between records are not records, and a
// record the stream ends without terminating is completed.
func TestCopilotStreamsChatRecordsAsSent(t *testing.T) {
	t.Parallel()
	records := []string{
		": keepalive\r\n\r\n",
		"event: chunk\r\nid: 1\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\r\n\r\n",
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\r\ndata: \"finish_reason\":\"stop\"}]}\n\n",
		"data: [DONE]\n\n",
	}
	backend := newCopilotBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "\n"+records[0]+records[1]+"\r\n"+records[2]+records[3]+"data: {\"usage\":{}}")
	})
	stream, err := newFixtureCopilot(t, backend).Stream(context.Background(),
		copilotRequest(core.ModelSurfaceChatCompletions, "gpt-fixture", copilotRichChat, nil))
	if err != nil {
		t.Fatal(err)
	}
	want := append(append([]string(nil), records...), "data: {\"usage\":{}}\n\n")
	if frames := drainCopilot(t, stream); !reflect.DeepEqual(frames, want) {
		t.Fatalf("frames = %q\nwant %q", frames, want)
	}
	if losses := core.StreamLosses(stream); len(losses) != 4 {
		t.Fatalf("stream losses = %+v, want the request's four dropped fields", losses)
	}
	exchanges, calls := backend.take()
	if !reflect.DeepEqual(exchanges, []string{"product-oauth"}) || len(calls) != 1 {
		t.Fatalf("exchanges = %v calls = %+v", exchanges, calls)
	}
	// The gateway asks for a Chat stream with its usual Accept header.
	upstream := strings.Replace(copilotRichChatUpstream, `"stream":false`, `"stream":true`, 1)
	if calls[0].path != "/api/chat/completions" || calls[0].body != upstream {
		t.Fatalf("upstream = %s %s", calls[0].path, calls[0].body)
	}
	assertCopilotHeaders(t, calls[0].header, "session-1", "application/json", false)
}

func TestCopilotStreamRecordsAreBounded(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, func(w http.ResponseWriter, _ *http.Request, _ []byte) {
		_, _ = io.WriteString(w, "data: {}\n\ndata: "+strings.Repeat("x", copilotMaxRecordBytes)+"\n\n")
	})
	stream, err := newFixtureCopilot(t, backend).Stream(context.Background(),
		copilotRequest(core.ModelSurfaceChatCompletions, "gpt-fixture", `{"messages":[]}`, nil))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if frame, err := stream.Next(); err != nil || string(frame) != "data: {}\n\n" {
		t.Fatalf("first frame = %q err = %v", frame, err)
	}
	_, err = stream.Next()
	assertCopilotFailure(t, err, core.ProviderErrorUpstream, core.ProviderErrorClassification{})
}

// Copilot rejects images unless the request says it carries them.
func TestCopilotSendsTheVisionHeaderOnlyWithImages(t *testing.T) {
	t.Parallel()
	for name, body := range map[string]string{
		"image_url part":   `{"messages":[{"role":"user","content":[{"type":"text","text":"see"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}`,
		"input_image part": `{"messages":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AA=="}]}]}`,
		"image part":       `{"messages":[{"role":"user","content":[{"type":"image"}]}]}`,
		"text only":        `{"messages":[{"role":"user","content":[{"type":"text","text":"image_url"}]}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			backend := newCopilotBackend(t, answerCopilotChat)
			if _, err := newFixtureCopilot(t, backend).Invoke(context.Background(),
				copilotRequest(core.ModelSurfaceChatCompletions, "gpt-fixture", body, nil)); err != nil {
				t.Fatal(err)
			}
			_, calls := backend.take()
			assertCopilotHeaders(t, calls[0].header, "session-1", "application/json", name != "text only")
		})
	}
}
