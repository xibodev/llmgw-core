package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers/zen"
)

// zenUpstreamCall is one request the synthetic Zen backend received.
type zenUpstreamCall struct {
	method, path, body string
	header             http.Header
}

// zenBackend records every request and answers with reply.
type zenBackend struct {
	mu    sync.Mutex
	calls []zenUpstreamCall
	reply func(w http.ResponseWriter, r *http.Request, call int)
}

func (b *zenBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	b.mu.Lock()
	b.calls = append(b.calls, zenUpstreamCall{method: r.Method, path: r.URL.Path, body: string(body), header: r.Header.Clone()})
	call := len(b.calls)
	b.mu.Unlock()
	b.reply(w, r, call)
}

func (b *zenBackend) take() []zenUpstreamCall {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

var zenFixtureNow = time.Date(2026, time.September, 24, 12, 0, 0, 0, time.UTC)

func newTestZen(t *testing.T, server *httptest.Server, models func(string) (core.ModelInfo, bool)) *Zen {
	t.Helper()
	provider, err := NewZen(ZenConfig{
		BaseURL: server.URL + "/zen/v1/", MetadataURL: server.URL + "/metadata",
		Client: server.Client(), CatalogClient: server.Client(), Models: models,
		Now:   func() time.Time { return zenFixtureNow },
		NewID: func(prefix string) (string, error) { return prefix + "_fixture", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

// zenJSON encodes a value as the gateway encodes request bodies.
func zenJSON(t *testing.T, value any) string {
	t.Helper()
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		t.Fatal(err)
	}
	return strings.TrimSuffix(buffer.String(), "\n")
}

func chatRequest(model, body string, credential *core.Credential) core.Request {
	return core.Request{Surface: core.ModelSurfaceChatCompletions, Model: model, Body: []byte(body), ContentType: core.ContentTypeJSON, Credential: credential}
}

func responsesRequest(model, body string, credential *core.Credential) core.Request {
	return core.Request{Surface: core.ModelSurfaceResponses, Model: model, Body: []byte(body), ContentType: core.ContentTypeJSON, Credential: credential}
}

// assertZenHeaders checks the headers the gateway sends Zen: JSON, the
// bearer, the invocation identity, and accept only where the gateway sets
// one.
func assertZenHeaders(t *testing.T, header http.Header, authorization, accept string, identity zen.InvocationIdentity) {
	t.Helper()
	for key, want := range map[string]string{
		"Content-Type": "application/json", "Authorization": authorization, "Accept": accept,
		"X-Opencode-Project": identity.Project, "X-Opencode-Session": identity.Session, "X-Opencode-Request": identity.Request,
		"X-Opencode-Client": identity.Client, "User-Agent": identity.UserAgent, "Copilot-Vision-Request": "",
	} {
		if got := header.Get(key); got != want {
			t.Errorf("%s = %q, want %q", key, got, want)
		}
	}
}

func readZenFrames(t *testing.T, stream core.StreamIter) (string, error) {
	t.Helper()
	defer stream.Close()
	var frames strings.Builder
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			return frames.String(), nil
		}
		if err != nil {
			return frames.String(), err
		}
		frames.Write(frame)
	}
}

// freshIdentity is the identity Zen makes for a context without one.
var freshIdentity = zen.InvocationIdentity{Project: "global", Session: "ses_fixture", Request: "msg_fixture", Client: "cli", UserAgent: zen.AnonymousUserAgent}

const zenChatUpstreamStream = ": keepalive\n\n" +
	"data: {\"id\":\"chatcmpl-fixture\",\"object\":\"chat.completion.chunk\",\"created\":1750000000,\"model\":\"big-pickle\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n" +
	"data: {\"id\":\"chatcmpl-fixture\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" from zen\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7}}\r\n\r\n" +
	"data: [DONE]\n\n"

func TestZenShapesAnonymousChatAsTheGatewayDoes(t *testing.T) {
	t.Parallel()
	backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, _ int) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, zenChatUpstreamStream)
	}}
	server := httptest.NewServer(backend)
	defer server.Close()
	provider := newTestZen(t, server, nil)
	body := `{"model":"ignored","stream":false,"max_tokens":64,"response_format":{"type":"text"},"_affinity_key":"a","tools":[],"messages":[{"role":"user","content":"Say <hello> & more"}]}`
	// The gateway admits a completion twice, and the second admission drops
	// the empty tools list; the stream is admitted once and keeps it.
	upstream := `{"max_tokens":64,"messages":[{"content":` + zenJSON(t, zen.AnonymousAssistantPreamble) + `,"role":"system"},{"content":"Say <hello> & more","role":"user"}],"model":"big-pickle","stream":true}`
	streamed := strings.Replace(upstream, `"stream":true}`, `"stream":true,"tools":[]}`, 1)

	response, err := provider.Invoke(context.Background(), chatRequest("big-pickle", body, nil))
	if err != nil {
		t.Fatal(err)
	}
	const completion = `{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"Hello from zen","role":"assistant"}}],"created":1750000000,"id":"chatcmpl-fixture","model":"big-pickle","object":"chat.completion","usage":{"completion_tokens":4,"prompt_tokens":3,"total_tokens":7}}`
	if string(response.Body) != completion {
		t.Fatalf("completion = %s", response.Body)
	}
	if len(response.Losses) != 1 || response.Losses[0].Path != "response_format" || response.Losses[0].Class != translate.LossDropped ||
		response.Losses[0].Severity != translate.LossAdvisory {
		t.Fatalf("losses = %+v", response.Losses)
	}
	calls := backend.take()
	if len(calls) != 1 || calls[0].method != http.MethodPost || calls[0].path != "/zen/v1/chat/completions" || calls[0].body != upstream {
		t.Fatalf("upstream = %+v\nwant body %s", calls, upstream)
	}
	assertZenHeaders(t, calls[0].header, "Bearer public", "", freshIdentity)

	stream, err := provider.Stream(context.Background(), chatRequest("big-pickle", body, &core.Credential{APIKey: "free"}))
	if err != nil {
		t.Fatal(err)
	}
	frames, err := readZenFrames(t, stream)
	if err != nil || frames != strings.TrimPrefix(zenChatUpstreamStream, ": keepalive\n\n") {
		t.Fatalf("frames = %q, err = %v", frames, err)
	}
	if losses := core.StreamLosses(stream); len(losses) != 1 || losses[0].Path != "response_format" {
		t.Fatalf("stream losses = %+v", losses)
	}
	if calls := backend.take(); len(calls) != 1 || calls[0].body != streamed {
		t.Fatalf("stream upstream = %+v\nwant body %s", calls, streamed)
	}
}
