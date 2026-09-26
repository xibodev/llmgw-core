package runtime_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
	"github.com/xibodev/llmgw-core/translation"
)

// anthropicUpstreamBody is what reaches Anthropic for "Say hello" to
// claude-fixture. The gateway's anthropic-messages-native goldens record
// the same body for the same requests, stream flag aside.
const anthropicUpstreamBody = `{"max_tokens":64,"messages":[{"content":"Say hello","role":"user"}],"model":"claude-fixture","stream":%t}`

// anthropicUpstreamMessage and anthropicUpstreamEvents are the answers of
// the gateway's characterization upstream, as it writes them.
const anthropicUpstreamMessage = `{"content":[{"text":"Hello from messages","type":"text"}],"id":"msg_fixture","model":"claude-fixture","role":"assistant","stop_reason":"end_turn","stop_sequence":null,"type":"message","usage":{"input_tokens":3,"output_tokens":4}}`

const anthropicUpstreamEvents = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-fixture\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" from messages\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":4}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

// anthropicGoldenBody is the client body the anthropic-messages-native
// golden records.
const anthropicGoldenBody = `{
  "content": [{"text": "Hello from messages", "type": "text"}],
  "id": "msg_fixture", "model": "claude-fixture", "role": "assistant",
  "stop_reason": "end_turn", "stop_sequence": null, "type": "message",
  "usage": {"input_tokens": 3, "output_tokens": 4}
}`

type anthropicRuntimeCall struct{ path, apiKey, authorization, beta, version, body string }

// anthropicRuntimeBackend is a synthetic Anthropic mounted under /n, as the
// gateway's characterization mounts it. It accepts one API key and any
// setup token.
type anthropicRuntimeBackend struct {
	mu    sync.Mutex
	calls []anthropicRuntimeCall
}

func (b *anthropicRuntimeBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	b.mu.Lock()
	b.calls = append(b.calls, anthropicRuntimeCall{
		path: r.URL.Path, apiKey: r.Header.Get("x-api-key"), authorization: r.Header.Get("Authorization"),
		beta: strings.Join(r.Header.Values("anthropic-beta"), ","), version: r.Header.Get("anthropic-version"), body: string(body),
	})
	b.mu.Unlock()
	switch {
	case r.Header.Get("x-api-key") != "fixture-upstream-token":
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`)
	case r.URL.Path == "/n/v1/models":
		_, _ = io.WriteString(w, `{"data":[{"id":"claude-fixture","type":"model","display_name":"Claude Fixture","created_at":"2025-01-01T00:00:00Z"}],"has_more":false}`)
	case r.URL.Path == "/n/v1/messages/count_tokens":
		_, _ = io.WriteString(w, `{"input_tokens":3}`)
	case r.URL.Path == "/n/v1/messages" && strings.Contains(string(body), `"stream":true`):
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, anthropicUpstreamEvents)
	case r.URL.Path == "/n/v1/messages":
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, anthropicUpstreamMessage+"\n")
	default:
		http.NotFound(w, r)
	}
}

func (b *anthropicRuntimeBackend) take() []anthropicRuntimeCall {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

type anthropicSettings struct{ BaseURL string }

// newRuntimeAnthropic builds the provider as a product would: Anthropic
// behind a translation.Adapter.
func newRuntimeAnthropic(server *httptest.Server, baseURL string) (core.Provider, error) {
	anthropic, err := providers.NewAnthropic(providers.AnthropicConfig{
		BaseURL: baseURL, Client: server.Client(), CatalogClient: server.Client(),
	})
	if err != nil {
		return nil, err
	}
	return translation.Adapter{Provider: anthropic}, nil
}

func newAnthropicRuntime(t *testing.T, server *httptest.Server, store core.CredentialStore) *coreruntime.Runtime[anthropicSettings] {
	t.Helper()
	runtime, err := coreruntime.New(coreruntime.Options[anthropicSettings]{
		Settings: coreruntime.NewMemorySettings(anthropicSettings{BaseURL: server.URL + "/n"}),
		Providers: func(settings anthropicSettings, _ string) (core.Provider, error) {
			return newRuntimeAnthropic(server, settings.BaseURL)
		},
		Credentials: store,
	})
	if err != nil {
		t.Fatal(err)
	}
	return runtime
}

func anthropicSameJSON(t *testing.T, got []byte, want string) bool {
	t.Helper()
	var decodedGot, decodedWant any
	if err := json.Unmarshal(got, &decodedGot); err != nil {
		t.Fatalf("body %q: %v", got, err)
	}
	if err := json.Unmarshal([]byte(want), &decodedWant); err != nil {
		t.Fatal(err)
	}
	return reflect.DeepEqual(decodedGot, decodedWant)
}

func readAnthropicRuntimeStream(t *testing.T, stream core.StreamIter) string {
	t.Helper()
	defer stream.Close()
	var frames strings.Builder
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			return frames.String()
		}
		if err != nil {
			t.Fatal(err)
		}
		frames.Write(frame)
	}
}

// TestRuntimeServesTheAnthropicVertical replays the gateway's
// anthropic-messages-native goldens. Both send the golden's upstream body.
// The client gets Anthropic's answer as sent: the golden body for a
// completion, and Anthropic's own events for a stream, where the gateway
// streamed through Chat and rendered events of its own.
func TestRuntimeServesTheAnthropicVertical(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	backend := &anthropicRuntimeBackend{}
	server := httptest.NewServer(backend)
	defer server.Close()
	store := core.NewMemoryCredentialStore()
	owner, stale := core.Caller{ID: "owner", Kind: core.CallerHuman}, core.Caller{ID: "stale", Kind: core.CallerHuman}
	for caller, record := range map[core.Caller]tokenstore.Record{
		owner: core.APIKeyRecord("fixture-upstream-token"),
		stale: core.APIKeyRecord("fixture-revoked-token"),
	} {
		if _, err := store.Save(ctx, caller.ID+"-anthropic", record); err != nil {
			t.Fatal(err)
		}
		store.Bind(caller, "native", caller.ID+"-anthropic")
	}
	runtime := newAnthropicRuntime(t, server, store)
	messages := func(stream bool) core.Request {
		return core.Request{
			Surface: core.ModelSurfaceMessages, Model: "claude-fixture", ContentType: core.ContentTypeJSON,
			Body: fmt.Appendf(nil, `{"max_tokens":64,"messages":[{"content":"Say hello","role":"user"}],"model":"native/claude-fixture","stream":%t}`, stream),
		}
	}
	keyed := func(stream bool) anthropicRuntimeCall {
		return anthropicRuntimeCall{path: "/n/v1/messages", apiKey: "fixture-upstream-token", version: "2023-06-01", body: fmt.Sprintf(anthropicUpstreamBody, stream)}
	}

	t.Run("anthropic-messages-native", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "native", messages(false))
		if err != nil {
			t.Fatal(err)
		}
		if string(response.Body) != anthropicUpstreamMessage || !anthropicSameJSON(t, response.Body, anthropicGoldenBody) {
			t.Fatalf("response = %s", response.Body)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []anthropicRuntimeCall{keyed(false)}) {
			t.Fatalf("upstream = %+v", calls)
		}
		// The golden labels the answer native. A product reads the label
		// from the declaration, which the Adapter passes through.
		provider, err := newRuntimeAnthropic(server, server.URL+"/n")
		if err != nil || !core.PreservesWire(provider, "claude-fixture", core.ModelSurfaceMessages) {
			t.Fatalf("Messages is not preserved through the adapter: %v", err)
		}
	})

	t.Run("anthropic-messages-native-stream", func(t *testing.T) {
		stream, err := runtime.Stream(ctx, owner, "native", messages(true))
		if err != nil {
			t.Fatal(err)
		}
		if frames := readAnthropicRuntimeStream(t, stream); frames != anthropicUpstreamEvents {
			t.Fatalf("frames = %q", frames)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []anthropicRuntimeCall{keyed(true)}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("catalog", func(t *testing.T) {
		record, err := runtime.ListModels(ctx, owner, "native")
		models := record.Evidence.Models
		if err != nil || record.Evidence.Status != core.CatalogDiscovered || len(models) != 1 || models[0].ID != "claude-fixture" ||
			models[0].DisplayName != "Claude Fixture" || !reflect.DeepEqual(models[0].SupportedAPIs, []string{"/v1/messages"}) ||
			models[0].Capabilities == nil || models[0].Capabilities.Surfaces.Messages != core.SupportSupported ||
			models[0].Capabilities.Provenance.Source != core.ModelCapabilitySourceRegistryStatic {
			t.Fatalf("catalog = %+v, err = %v", record, err)
		}
		want := anthropicRuntimeCall{path: "/n/v1/models", apiKey: "fixture-upstream-token", version: "2023-06-01"}
		if calls := backend.take(); !reflect.DeepEqual(calls, []anthropicRuntimeCall{want}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("token count through the adapter", func(t *testing.T) {
		provider, err := newRuntimeAnthropic(server, server.URL+"/n")
		if err != nil {
			t.Fatal(err)
		}
		request := messages(false)
		request.Body = []byte(`{"model":"native/claude-fixture","messages":[{"role":"user","content":"Say hello"}]}`)
		request.Credential = core.CredentialFromRecord("owner-anthropic", core.APIKeyRecord("fixture-upstream-token"))
		count, err := core.CountTokens(ctx, provider, core.TokenCountRequest{Request: request})
		if err != nil || count.InputTokens != 3 {
			t.Fatalf("count = %+v, err = %v", count, err)
		}
		want := anthropicRuntimeCall{
			path: "/n/v1/messages/count_tokens", apiKey: "fixture-upstream-token", version: "2023-06-01",
			body: `{"messages":[{"content":"Say hello","role":"user"}],"model":"claude-fixture"}`,
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []anthropicRuntimeCall{want}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("a rejected key fails with 401 and is never replayed", func(t *testing.T) {
		_, err := runtime.Invoke(ctx, stale, "native", messages(false))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || core.ClassifyError(err).StatusCode != http.StatusUnauthorized ||
			failure.Message != "anthropic returned HTTP 401 (type=authentication_error)" {
			t.Fatalf("err = %#v", err)
		}
		if calls := backend.take(); len(calls) != 1 || calls[0].apiKey != "fixture-revoked-token" {
			t.Fatalf("upstream = %+v", calls)
		}
		if health := runtime.Health("native"); health.ErrorClass != core.ProviderErrorAuth {
			t.Fatalf("health = %+v", health)
		}
	})
}
