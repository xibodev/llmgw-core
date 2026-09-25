package runtime_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	"github.com/xibodev/llmgw-core/providers/zen"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
	"github.com/xibodev/llmgw-core/translation"
)

const zenRuntimeMetadata = `{"opencode":{"npm":"@ai-sdk/openai-compatible","models":{
"chat-fixture-free":{"id":"chat-fixture-free","name":"Chat Fixture Free","tool_call":true,"cost":{"input":0,"output":0},"limit":{"context":200000,"output":32000}},
"muse-fixture-free":{"id":"muse-fixture-free","name":"Muse Fixture Free","provider":{"npm":"@ai-sdk/openai"},"cost":{"input":0,"output":0}},
"paid-fixture":{"id":"paid-fixture","name":"Paid Fixture","cost":{"input":1,"output":4}}}}}`

const zenRuntimeLive = `{"object":"list","data":[{"id":"chat-fixture-free","object":"model","created":1750000000,"owned_by":"opencode"},{"id":"muse-fixture-free"},{"id":"paid-fixture"}]}`

const zenRuntimeChatStream = "data: {\"id\":\"chatcmpl-zen\",\"created\":1750000000,\"model\":\"chat-fixture-free\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello from zen\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"

const zenRuntimeEvents = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_zen\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
	"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello from muse\"}\n\n" +
	"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_zen\",\"status\":\"completed\",\"output\":[]}}\n\n"

type zenRuntimeCall struct{ path, authorization, session, body string }

// zenRuntimeBackend is a synthetic OpenCode Zen with its models.dev
// catalog. Anonymous access is refused for the paid model.
type zenRuntimeBackend struct {
	mu    sync.Mutex
	calls []zenRuntimeCall
}

func (b *zenRuntimeBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	b.mu.Lock()
	b.calls = append(b.calls, zenRuntimeCall{path: r.URL.Path, authorization: r.Header.Get("Authorization"), session: r.Header.Get("x-opencode-session"), body: string(body)})
	b.mu.Unlock()
	anonymous := r.Header.Get("Authorization") == "Bearer public"
	switch {
	case r.URL.Path == "/metadata":
		_, _ = io.WriteString(w, zenRuntimeMetadata)
	case r.URL.Path == "/zen/v1/models":
		_, _ = io.WriteString(w, zenRuntimeLive)
	case anonymous && strings.Contains(string(body), `"model":"paid-fixture"`):
		w.WriteHeader(http.StatusUnauthorized)
	case r.URL.Path == "/zen/v1/chat/completions" && strings.Contains(string(body), `"stream":true`):
		_, _ = io.WriteString(w, zenRuntimeChatStream)
	case r.URL.Path == "/zen/v1/chat/completions":
		_, _ = io.WriteString(w, `{"id":"chatcmpl-keyed","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello with a key"},"finish_reason":"stop"}]}`)
	case r.URL.Path == "/zen/v1/responses":
		_, _ = io.WriteString(w, zenRuntimeEvents)
	default:
		http.NotFound(w, r)
	}
}

func (b *zenRuntimeBackend) take() []zenRuntimeCall {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

type zenSettings struct{ BaseURL string }

// newZenRuntime wires Zen to the catalog the Runtime keeps, so a model is
// served on the surfaces its anonymous catalog row lists.
func newZenRuntime(t *testing.T, server *httptest.Server, store core.CredentialStore) *coreruntime.Runtime[zenSettings] {
	t.Helper()
	catalogs := core.NewMemoryCatalogStore()
	runtime, err := coreruntime.New(coreruntime.Options[zenSettings]{
		Settings: coreruntime.NewMemorySettings(zenSettings{BaseURL: server.URL}),
		Providers: func(settings zenSettings, instance string) (core.Provider, error) {
			provider, err := providers.NewZen(providers.ZenConfig{
				BaseURL: settings.BaseURL + "/zen/v1", MetadataURL: settings.BaseURL + "/metadata",
				Client: server.Client(), CatalogClient: server.Client(),
				NewID: func(prefix string) (string, error) { return prefix + "_fixture", nil },
				Models: func(model string) (core.ModelInfo, bool) {
					record, err := catalogs.Load(context.Background(), core.CatalogKey{Instance: instance})
					if err != nil {
						return core.ModelInfo{}, false
					}
					index := slices.IndexFunc(record.Evidence.Models, func(row core.ModelInfo) bool { return row.ID == model })
					if index < 0 {
						return core.ModelInfo{}, false
					}
					return record.Evidence.Models[index], true
				},
			})
			if err != nil {
				return nil, err
			}
			return translation.Adapter{Provider: provider}, nil
		},
		Credentials: store, Catalogs: catalogs,
	})
	if err != nil {
		t.Fatal(err)
	}
	return runtime
}

func zenPreambled(t *testing.T, instructions string) string {
	t.Helper()
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(instructions); err != nil {
		t.Fatal(err)
	}
	return strings.TrimSuffix(buffer.String(), "\n")
}

func TestRuntimeServesTheZenVertical(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	backend := &zenRuntimeBackend{}
	server := httptest.NewServer(backend)
	defer server.Close()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(ctx, "owner-zen", core.APIKeyRecord("fixture-key")); err != nil {
		t.Fatal(err)
	}
	owner, visitor := core.Caller{ID: "owner", Kind: core.CallerHuman}, core.Caller{ID: "visitor", Kind: core.CallerHuman}
	store.Bind(owner, "zen", "owner-zen")
	runtime := newZenRuntime(t, server, store)
	preamble := zenPreambled(t, zen.AnonymousAssistantPreamble)
	chat := func(model string) core.Request {
		return core.Request{
			Surface: core.ModelSurfaceChatCompletions, Model: model, ContentType: core.ContentTypeJSON,
			Body: []byte(`{"model":"` + model + `","messages":[{"role":"user","content":"Say hello"}]}`),
		}
	}

	t.Run("catalog", func(t *testing.T) {
		record, err := runtime.ListModels(ctx, visitor, "zen")
		models := record.Evidence.Models
		if err != nil || record.Evidence.Status != core.CatalogDiscovered || len(models) != 2 ||
			models[0].ID != "chat-fixture-free" || !slices.Equal(models[0].SupportedAPIs, []string{"/chat/completions"}) ||
			models[1].ID != "muse-fixture-free" || !slices.Equal(models[1].SupportedAPIs, []string{"/responses"}) ||
			!slices.Equal(models[1].Tags, []string{providers.ModelTagFree}) || models[1].Capabilities.Surfaces.Responses != core.SupportSupported {
			t.Fatalf("anonymous catalog = %+v, err = %v", record, err)
		}
		if calls := backend.take(); len(calls) != 2 || calls[0].path != "/metadata" || calls[0].authorization != "" ||
			calls[1].path != "/zen/v1/models" || calls[1].authorization != "Bearer public" || calls[1].session != "ses_fixture" {
			t.Fatalf("anonymous catalog upstream = %+v", calls)
		}
		record, err = runtime.ListModels(ctx, owner, "zen")
		if err != nil || len(record.Evidence.Models) != 3 || record.Evidence.Models[2].ID != "paid-fixture" || len(record.Evidence.Models[2].Tags) != 0 {
			t.Fatalf("keyed catalog = %+v, err = %v", record, err)
		}
		if calls := backend.take(); len(calls) != 1 || calls[0].authorization != "Bearer fixture-key" || calls[0].session != "" {
			t.Fatalf("keyed catalog upstream = %+v", calls)
		}
	})

	t.Run("anonymous chat", func(t *testing.T) {
		identity := zen.InvocationIdentity{Project: "global", Session: "ses_caller", Request: "msg_caller", Client: "cli", UserAgent: zen.AnonymousUserAgent}
		response, err := runtime.Invoke(zen.WithInvocationIdentity(ctx, identity), visitor, "zen", chat("chat-fixture-free"))
		if err != nil || !strings.Contains(string(response.Body), `"content":"Hello from zen"`) {
			t.Fatalf("response = %s, err = %v", response.Body, err)
		}
		want := zenRuntimeCall{
			path: "/zen/v1/chat/completions", authorization: "Bearer public", session: "ses_caller",
			body: `{"messages":[{"content":` + preamble + `,"role":"system"},{"content":"Say hello","role":"user"}],"model":"chat-fixture-free","stream":true}`,
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []zenRuntimeCall{want}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("anonymous responses stream", func(t *testing.T) {
		stream, err := runtime.Stream(ctx, visitor, "zen", core.Request{
			Surface: core.ModelSurfaceResponses, Model: "muse-fixture-free", ContentType: core.ContentTypeJSON,
			Body: []byte(`{"model":"muse-fixture-free","input":"Say hello"}`),
		})
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Close()
		var frames strings.Builder
		for {
			frame, err := stream.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			frames.Write(frame)
		}
		if frames.String() != zenRuntimeEvents {
			t.Fatalf("frames = %q", frames.String())
		}
		want := zenRuntimeCall{
			path: "/zen/v1/responses", authorization: "Bearer public", session: "ses_fixture",
			body: `{"input":"Say hello","instructions":` + preamble + `,"model":"muse-fixture-free","stream":true}`,
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []zenRuntimeCall{want}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("keyed chat", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "zen", chat("paid-fixture"))
		if err != nil || !strings.Contains(string(response.Body), "Hello with a key") {
			t.Fatalf("response = %s, err = %v", response.Body, err)
		}
		want := zenRuntimeCall{
			path: "/zen/v1/chat/completions", authorization: "Bearer fixture-key", session: "ses_fixture",
			body: `{"messages":[{"content":"Say hello","role":"user"}],"model":"paid-fixture","stream":false}`,
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []zenRuntimeCall{want}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("a paid model without a key fails with 401", func(t *testing.T) {
		_, err := runtime.Invoke(ctx, visitor, "zen", chat("paid-fixture"))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Classification.StatusCode != http.StatusUnauthorized ||
			!strings.Contains(failure.Message, "configure an OpenCode Zen API key") {
			t.Fatalf("err = %#v", err)
		}
		if health := runtime.Health("zen"); health.ErrorClass != core.ProviderErrorAuth {
			t.Fatalf("health = %+v", health)
		}
	})
}
