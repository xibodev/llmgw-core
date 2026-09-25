package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers/zen"
)

func responsesRow(model string) (core.ModelInfo, bool) {
	return core.ModelInfo{ID: model, SupportedAPIs: []string{"/responses"}, Tags: []string{ModelTagFree}}, strings.HasPrefix(model, "gpt-")
}

func TestZenPassesKeyedResponsesThrough(t *testing.T) {
	t.Parallel()
	const answer = `{"id":"resp_keyed","object":"response","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"Hi <b>"}]}]}`
	backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, _ int) { _, _ = io.WriteString(w, answer) }}
	server := httptest.NewServer(backend)
	defer server.Close()
	provider := newTestZen(t, server, responsesRow)
	body := `{"model":"ignored","stream":true,"force_api_support":true,"input":"Say hello","instructions":"Be brief","temperature":0.5}`
	response, err := provider.Invoke(context.Background(), responsesRequest("gpt-fixture", body, &core.Credential{Token: "fixture-key"}))
	if err != nil || string(response.Body) != answer {
		t.Fatalf("response = %s, err = %v", response.Body, err)
	}
	calls := backend.take()
	const upstream = `{"input":"Say hello","instructions":"Be brief","model":"gpt-fixture","stream":false,"temperature":0.5}`
	if len(calls) != 1 || calls[0].path != "/zen/v1/responses" || calls[0].body != upstream {
		t.Fatalf("upstream = %+v", calls)
	}
	assertZenHeaders(t, calls[0].header, "Bearer fixture-key", "application/json", freshIdentity)
}

// zenResponsesUpstream is a Zen Responses stream with what the gateway
// skips: a comment, a record that is not JSON, one without a type, [DONE],
// and events after the terminal one.
const zenResponsesUpstream = ": keepalive\n\n" +
	"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_fixture\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
	"data: not json\n\n" +
	"data: {\"delta\":\"typeless\"}\n\n" +
	"data: {\"type\":\"error\",\"message\":\"transient\"}\n\n" +
	"event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\r\ndata: \"delta\":\"Hello\"}\r\n\r\n" +
	"data: [DONE]\n\n" +
	"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fixture\",\"status\":\"completed\",\"output\":[]}}\n\n" +
	"data: {\"type\":\"response.output_text.delta\",\"delta\":\"after\"}\n\n"

func TestZenStreamsResponsesAsTheGatewayRelaysThem(t *testing.T) {
	t.Parallel()
	backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, _ int) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, zenResponsesUpstream)
	}}
	server := httptest.NewServer(backend)
	defer server.Close()
	provider := newTestZen(t, server, responsesRow)
	stream, err := provider.Stream(context.Background(), responsesRequest("gpt-fixture", `{"input":"Say hello","instructions":"Be <brief>"}`, nil))
	if err != nil {
		t.Fatal(err)
	}
	frames, err := readZenFrames(t, stream)
	want := "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_fixture\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
		"data: {\"type\":\"error\",\"message\":\"transient\"}\n\n" +
		"event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\r\ndata: \"delta\":\"Hello\"}\r\n\r\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fixture\",\"status\":\"completed\",\"output\":[]}}\n\n"
	if err != nil || frames != want {
		t.Fatalf("frames = %q, err = %v", frames, err)
	}
	calls := backend.take()
	upstream := `{"input":"Say hello","instructions":` + zenJSON(t, zen.AnonymousAssistantPreamble+"\n\nBe <brief>") + `,"model":"gpt-fixture","stream":true}`
	if len(calls) != 1 || calls[0].body != upstream {
		t.Fatalf("upstream = %+v\nwant body %s", calls, upstream)
	}
	assertZenHeaders(t, calls[0].header, "Bearer public", "text/event-stream", freshIdentity)
}

func TestZenResponsesStreamFailsWithoutATerminalEvent(t *testing.T) {
	t.Parallel()
	for name, events := range map[string]string{
		"no terminal":      "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_fixture\"}}\n\n",
		"oversized record": "data: {\"type\":\"response.created\"}\n\ndata: \"" + strings.Repeat("x", zenMaxRecordBytes) + "\"\n\n",
	} {
		backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, _ int) { _, _ = io.WriteString(w, events) }}
		server := httptest.NewServer(backend)
		provider := newTestZen(t, server, responsesRow)
		stream, err := provider.Stream(context.Background(), responsesRequest("gpt-fixture", `{"input":"Say hello"}`, &core.Credential{APIKey: "fixture-key"}))
		if err != nil {
			t.Fatal(err)
		}
		frames, err := readZenFrames(t, stream)
		var failure *core.ProviderError
		if !strings.HasPrefix(frames, "data: {\"type\":\"response.created\"") || !errors.As(err, &failure) || failure.Class != core.ProviderErrorUpstream {
			t.Fatalf("%s: frames = %.80q, err = %v", name, frames, err)
		}
		server.Close()
	}
}

func TestZenAssemblesAnAnonymousResponse(t *testing.T) {
	t.Parallel()
	events := "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_fixture\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
		"data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fixture\",\"status\":\"completed\",\"output\":[]}}\n\n"
	backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, _ int) { _, _ = io.WriteString(w, events) }}
	server := httptest.NewServer(backend)
	defer server.Close()
	provider := newTestZen(t, server, responsesRow)
	response, err := provider.Invoke(context.Background(), responsesRequest("gpt-fixture", `{"input":[{"role":"user","content":"Say hello"}]}`, nil))
	const want = `{"id":"resp_fixture","output":[{"content":[{"text":"Hello","type":"output_text"}],"role":"assistant","status":"completed","type":"message"}],"status":"completed"}`
	if err != nil || string(response.Body) != want {
		t.Fatalf("response = %s, err = %v", response.Body, err)
	}
	calls := backend.take()
	upstream := `{"input":[{"content":"Say hello","role":"user"}],"instructions":` + zenJSON(t, zen.AnonymousAssistantPreamble) + `,"model":"gpt-fixture","stream":true}`
	if len(calls) != 1 || calls[0].body != upstream {
		t.Fatalf("upstream = %+v", calls)
	}
	assertZenHeaders(t, calls[0].header, "Bearer public", "application/json", freshIdentity)
}
