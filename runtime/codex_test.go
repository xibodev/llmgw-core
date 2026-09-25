package runtime_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	codexauth "github.com/xibodev/llm-provider-auth/codex"
	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
	"github.com/xibodev/llmgw-core/translation"
)

// codexUpstreamBody is what reaches Codex for "Say hello" to gpt-fixture,
// over Responses or Chat alike. The gateway's characterization goldens
// record the same body for the same requests.
const codexUpstreamBody = `{"input":[{"content":"Say hello","role":"user"}],"instructions":"Follow the caller's request.","model":"gpt-fixture","store":false,"stream":true}`

const codexUpstreamEvents = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_fixture\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
	"data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello from codex\"}\n\n" +
	"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fixture\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"gpt-fixture\",\"output\":[{\"type\":\"message\",\"id\":\"msg_fixture\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello from codex\"}]}],\"usage\":{\"input_tokens\":3,\"output_tokens\":4,\"total_tokens\":7}}}\n\n"

type codexCall struct{ path, authorization, account, beta, body string }

// codexBackend is a synthetic Codex backend with its OAuth token endpoint.
// It accepts one access token at a time; a refresh issues the next one.
type codexBackend struct {
	mu       sync.Mutex
	accepted string
	calls    []codexCall
}

func (b *codexBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	b.mu.Lock()
	b.calls = append(b.calls, codexCall{
		path: r.URL.Path, authorization: r.Header.Get("Authorization"), account: r.Header.Get("ChatGPT-Account-ID"),
		beta: r.Header.Get("OpenAI-Beta"), body: string(body),
	})
	accepted := r.Header.Get("Authorization") == "Bearer "+b.accepted
	b.mu.Unlock()
	switch {
	case r.URL.Path == "/oauth/token":
		claims, _ := json.Marshal(map[string]any{"https://api.openai.com/auth": map[string]string{"chatgpt_account_id": "account-fixture"}})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": "fixture-access-rotated", "refresh_token": "fixture-refresh-rotated", "expires_in": 3600,
			"id_token": "e30." + base64.RawURLEncoding.EncodeToString(claims) + ".fixture-signature",
		})
	case !accepted:
		w.WriteHeader(http.StatusUnauthorized)
	case r.URL.Path == "/backend-api/codex/responses":
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, codexUpstreamEvents)
	case r.URL.Path == "/backend-api/codex/models" && r.URL.Query().Get("client_version") == "fixture-client/1.0":
		// Current catalogs omit supported_endpoints.
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"models":[`+
			`{"slug":"gpt-fixture","display_name":"GPT Fixture","description":"GPT Fixture","visibility":"list","supported_in_api":true,"context_window":272000},`+
			`{"slug":"gpt-hidden","visibility":"hide","supported_in_api":true}]}`)
	default:
		http.NotFound(w, r)
	}
}

// take returns the calls since the last take.
func (b *codexBackend) take() []codexCall {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

// revoke makes the backend reject the current access token and accept only
// the one the next refresh issues.
func (b *codexBackend) revoke() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.accepted = "fixture-access-rotated"
}

type codexSettings struct{ BaseURL string }

func newCodexRuntime(t *testing.T, server *httptest.Server, store core.CredentialStore, evidence core.EvidenceSink) *coreruntime.Runtime[codexSettings] {
	t.Helper()
	runtime, err := coreruntime.New(coreruntime.Options[codexSettings]{
		Settings: coreruntime.NewMemorySettings(codexSettings{BaseURL: server.URL}),
		Providers: func(settings codexSettings, _ string) (core.Provider, error) {
			codex, err := providers.NewCodex(providers.CodexConfig{
				Instructions: "Follow the caller's request.", ClientVersion: "fixture-client/1.0",
				ResponsesURL: settings.BaseURL + "/backend-api/codex/responses",
				ModelsURL:    settings.BaseURL + "/backend-api/codex/models", Client: server.Client(),
			})
			if err != nil {
				return nil, err
			}
			return translation.Adapter{Provider: codex}, nil
		},
		Refresh: func(settings codexSettings, _ string) tokenstore.RefreshFunc {
			return providers.NewCodexRefresh(codexauth.Config{
				ClientID: "fixture-client", HTTPClient: server.Client(),
				Endpoints: codexauth.Endpoints{OAuthTokenURL: settings.BaseURL + "/oauth/token"},
			})
		},
		Credentials: store, Evidence: evidence,
	})
	if err != nil {
		t.Fatal(err)
	}
	return runtime
}

func TestRuntimeServesTheCodexVertical(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	backend := &codexBackend{accepted: "fixture-access"}
	server := httptest.NewServer(backend)
	defer server.Close()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(ctx, "owner-codex", tokenstore.Record{
		AccessToken: "fixture-access", RefreshToken: "fixture-refresh", TokenType: "Bearer",
		AccountID: "account-fixture", Expiry: time.Now().Add(time.Hour),
	}); err != nil {
		t.Fatal(err)
	}
	owner := core.Caller{ID: "owner", Kind: core.CallerHuman}
	store.Bind(owner, "codex", "owner-codex")
	evidence := &core.MemoryEvidenceSink{}
	runtime := newCodexRuntime(t, server, store, evidence)
	responses := core.Request{
		Surface: core.ModelSurfaceResponses, Model: "gpt-fixture", ContentType: core.ContentTypeJSON,
		Body: []byte(`{"model":"gpt-fixture","stream":false,"input":"Say hello"}`),
	}
	inference := codexCall{
		path: "/backend-api/codex/responses", authorization: "Bearer fixture-access", account: "account-fixture",
		beta: "responses=experimental", body: codexUpstreamBody,
	}

	t.Run("responses", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "codex", responses)
		if err != nil {
			t.Fatal(err)
		}
		var body struct {
			ID     string `json:"id"`
			Output []struct {
				Content []struct{ Text string } `json:"content"`
			} `json:"output"`
		}
		if json.Unmarshal(response.Body, &body) != nil || body.ID != "resp_fixture" || len(body.Output) != 1 ||
			len(body.Output[0].Content) != 1 || body.Output[0].Content[0].Text != "Hello from codex" {
			t.Fatalf("response = %s", response.Body)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []codexCall{inference}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("responses stream", func(t *testing.T) {
		stream, err := runtime.Stream(ctx, owner, "codex", responses)
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
		if frames.String() != codexUpstreamEvents {
			t.Fatalf("frames = %q", frames.String())
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []codexCall{inference}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("chat through the adapter", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "codex", core.Request{
			Surface: core.ModelSurfaceChatCompletions, Model: "gpt-fixture", ContentType: core.ContentTypeJSON,
			Body: []byte(`{"model":"gpt-fixture","stream":false,"messages":[{"role":"user","content":"Say hello"}]}`),
		})
		if err != nil {
			t.Fatal(err)
		}
		var body struct {
			Object  string `json:"object"`
			Choices []struct {
				FinishReason string                   `json:"finish_reason"`
				Message      struct{ Content string } `json:"message"`
			} `json:"choices"`
			Usage struct {
				TotalTokens int `json:"total_tokens"`
			} `json:"usage"`
		}
		if json.Unmarshal(response.Body, &body) != nil || body.Object != "chat.completion" || len(body.Choices) != 1 ||
			body.Choices[0].Message.Content != "Hello from codex" || body.Choices[0].FinishReason != "stop" || body.Usage.TotalTokens != 7 {
			t.Fatalf("chat response = %s", response.Body)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []codexCall{inference}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("catalog", func(t *testing.T) {
		record, err := runtime.ListModels(ctx, owner, "codex")
		if err != nil {
			t.Fatal(err)
		}
		models := record.Evidence.Models
		if record.Evidence.Status != core.CatalogDiscovered || len(models) != 1 || models[0].ID != "gpt-fixture" ||
			!slices.Equal(models[0].SupportedAPIs, []string{"/responses"}) || models[0].Capabilities == nil ||
			models[0].Capabilities.Surfaces.Responses != core.SupportSupported {
			t.Fatalf("catalog = %+v", record)
		}
		calls := backend.take()
		if len(calls) != 1 || calls[0].authorization != "Bearer fixture-access" || calls[0].account != "account-fixture" || calls[0].beta != "" {
			t.Fatalf("catalog upstream = %+v", calls)
		}
	})

	t.Run("a rejected token is refreshed once and the request replayed", func(t *testing.T) {
		backend.revoke()
		response, err := runtime.Invoke(ctx, owner, "codex", responses)
		if err != nil || !strings.Contains(string(response.Body), "Hello from codex") {
			t.Fatalf("response = %s, err = %v", response.Body, err)
		}
		replay := inference
		replay.authorization = "Bearer fixture-access-rotated"
		refresh := codexCall{path: "/oauth/token", body: `{"client_id":"fixture-client","grant_type":"refresh_token","refresh_token":"fixture-refresh"}`}
		if calls := backend.take(); !reflect.DeepEqual(calls, []codexCall{inference, refresh, replay}) {
			t.Fatalf("upstream = %+v", calls)
		}
		stored, err := store.Load(ctx, "owner-codex")
		if err != nil || stored.AccessToken != "fixture-access-rotated" || stored.RefreshToken != "fixture-refresh-rotated" ||
			stored.AccountID != "account-fixture" || stored.Expiry.Before(time.Now().Add(50*time.Minute)) {
			t.Fatalf("stored = %v, err = %v", stored, err)
		}
		records := evidence.Records()
		if last := records[len(records)-1]; last.CredentialRevision != stored.Revision || last.AccountID != "account-fixture" ||
			last.Outcome.Status != core.ProviderHealthHealthy {
			t.Fatalf("evidence = %+v", last)
		}
	})

	if health := runtime.Health("codex"); health.Status != core.ProviderHealthHealthy {
		t.Fatalf("health = %+v", health)
	}
}
