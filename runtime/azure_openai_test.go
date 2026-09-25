package runtime_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
	"github.com/xibodev/llmgw-core/translation"
)

// azureUpstreamBody is what reaches the resource for "Say hello" as
// Messages to the chat-model deployment. The gateway's
// openai-messages-via-chat goldens record the same body for a Chat target.
const azureUpstreamBody = `{"max_tokens":64,"messages":[{"content":"Say hello","role":"user"}],"model":"chat-model","stream":%t}`

// azureUpstreamCompletion and azureUpstreamChunks are the answers of the
// gateway's characterization upstream to a Chat request.
const azureUpstreamCompletion = `{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"Hello from chat","role":"assistant"}}],"created":1700000000,"id":"chatcmpl_fixture","model":"chat-model","object":"chat.completion","usage":{"completion_tokens":4,"prompt_tokens":3,"total_tokens":7}}`

const azureUpstreamChunks = "data: {\"id\":\"chatcmpl_fixture\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"chat-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n" +
	"data: {\"id\":\"chatcmpl_fixture\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"chat-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" from chat\"},\"finish_reason\":null}]}\n\n" +
	"data: {\"id\":\"chatcmpl_fixture\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"chat-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7}}\n\n" +
	"data: [DONE]\n\n"

// azureMessagesGoldenBody is the client body the openai-messages-via-chat
// golden records.
const azureMessagesGoldenBody = `{"content":[{"text":"Hello from chat","type":"text"}],"id":"chatcmpl_fixture","model":"chat-model","role":"assistant",
"stop_reason":"end_turn","stop_sequence":null,"type":"message","usage":{"input_tokens":3,"output_tokens":4}}`

type azureRuntimeCall struct{ method, path, query, apiKey, body string }

// azureRuntimeBackend is a synthetic Azure OpenAI resource with one chat
// deployment and one embedding deployment.
type azureRuntimeBackend struct {
	mu    sync.Mutex
	calls []azureRuntimeCall
}

func (b *azureRuntimeBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	b.mu.Lock()
	b.calls = append(b.calls, azureRuntimeCall{method: r.Method, path: r.URL.Path, query: r.URL.RawQuery, apiKey: r.Header.Get("api-key"), body: string(body)})
	b.mu.Unlock()
	switch {
	case r.Header.Get("api-key") != "fixture-azure-key":
		w.WriteHeader(http.StatusUnauthorized)
	case r.URL.Path == "/openai/deployments":
		_, _ = io.WriteString(w, `{"data":[{"id":"chat-model","model":"gpt-4o","status":"succeeded"},{"id":"vectors","model":"text-embedding-3-large","status":"succeeded"}],"object":"list"}`)
	case r.URL.Path == "/openai/v1/chat/completions" && strings.Contains(string(body), `"stream":true`):
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, azureUpstreamChunks)
	case r.URL.Path == "/openai/v1/chat/completions":
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, azureUpstreamCompletion)
	default:
		http.NotFound(w, r)
	}
}

func (b *azureRuntimeBackend) take() []azureRuntimeCall {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

type azureSettings struct{ Endpoint string }

// newRuntimeAzure builds the provider as a product would: Azure OpenAI
// behind a translation.Adapter, from the endpoint the portal shows.
func newRuntimeAzure(server *httptest.Server, endpoint string) (core.Provider, error) {
	azure, err := providers.NewAzureOpenAI(providers.AzureOpenAIConfig{BaseURL: endpoint, Client: server.Client(), CatalogClient: server.Client()})
	if err != nil {
		return nil, err
	}
	return translation.Adapter{Provider: azure}, nil
}

// TestRuntimeServesTheAzureOpenAIVertical replays the gateway's
// openai-messages-via-chat goldens against an Azure resource, and Chat and
// the catalog with them. The gateway labels every Azure answer translated,
// Chat included, and so does a label read from core.PreservesWire.
func TestRuntimeServesTheAzureOpenAIVertical(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	backend := &azureRuntimeBackend{}
	server := httptest.NewServer(backend)
	defer server.Close()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(ctx, "owner-azure", core.APIKeyRecord("fixture-azure-key")); err != nil {
		t.Fatal(err)
	}
	owner := core.Caller{ID: "owner", Kind: core.CallerHuman}
	store.Bind(owner, "azure", "owner-azure")
	runtime, err := coreruntime.New(coreruntime.Options[azureSettings]{
		Settings: coreruntime.NewMemorySettings(azureSettings{Endpoint: server.URL}),
		Providers: func(settings azureSettings, _ string) (core.Provider, error) {
			return newRuntimeAzure(server, settings.Endpoint)
		},
		Credentials: store,
	})
	if err != nil {
		t.Fatal(err)
	}
	messages := func(stream bool) core.Request {
		return core.Request{
			Surface: core.ModelSurfaceMessages, Model: "chat-model", ContentType: core.ContentTypeJSON,
			Body: fmt.Appendf(nil, `{"model":"azure/chat-model","stream":%t,"max_tokens":64,"messages":[{"role":"user","content":"Say hello"}]}`, stream),
		}
	}
	chat := func(stream bool) azureRuntimeCall {
		return azureRuntimeCall{method: http.MethodPost, path: "/openai/v1/chat/completions", apiKey: "fixture-azure-key", body: fmt.Sprintf(azureUpstreamBody, stream)}
	}

	t.Run("openai-messages-via-chat", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "azure", messages(false))
		if err != nil {
			t.Fatal(err)
		}
		var got, want any
		if json.Unmarshal(response.Body, &got) != nil || json.Unmarshal([]byte(azureMessagesGoldenBody), &want) != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("response = %s", response.Body)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []azureRuntimeCall{chat(false)}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("openai-messages-via-chat-stream", func(t *testing.T) {
		stream, err := runtime.Stream(ctx, owner, "azure", messages(true))
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
		for _, event := range []string{"event: message_start\n", `"text":"Hello"`, `"text":" from chat"`, `"stop_reason":"end_turn"`, "event: message_stop\n"} {
			if !strings.Contains(frames.String(), event) {
				t.Fatalf("frames lack %q: %q", event, frames.String())
			}
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []azureRuntimeCall{chat(true)}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("chat", func(t *testing.T) {
		request := messages(false)
		request.Surface, request.Body = core.ModelSurfaceChatCompletions, []byte(`{"model":"azure/chat-model","max_tokens":64,"messages":[{"role":"user","content":"Say hello"}]}`)
		response, err := runtime.Invoke(ctx, owner, "azure", request)
		if err != nil || string(response.Body) != azureUpstreamCompletion {
			t.Fatalf("response = %s, err = %v", response.Body, err)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []azureRuntimeCall{chat(false)}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("the transport label is translated", func(t *testing.T) {
		provider, err := newRuntimeAzure(server, server.URL)
		if err != nil {
			t.Fatal(err)
		}
		for _, surface := range []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceMessages} {
			if core.PreservesWire(provider, "chat-model", surface) {
				t.Fatalf("%s reads as native", surface)
			}
		}
	})

	t.Run("catalog", func(t *testing.T) {
		record, err := runtime.ListModels(ctx, owner, "azure")
		models := record.Evidence.Models
		if err != nil || len(models) != 1 || models[0].ID != "chat-model" || models[0].DisplayName != "gpt-4o" ||
			!reflect.DeepEqual(models[0].SupportedAPIs, []string{"/v1/chat/completions", "/v1/messages"}) ||
			models[0].Capabilities == nil || models[0].Capabilities.Surfaces.ChatCompletions != core.SupportSupported {
			t.Fatalf("catalog = %+v, err = %v", record, err)
		}
		want := azureRuntimeCall{method: http.MethodGet, path: "/openai/deployments", query: "api-version=2023-03-15-preview", apiKey: "fixture-azure-key"}
		if calls := backend.take(); !reflect.DeepEqual(calls, []azureRuntimeCall{want}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})
}
