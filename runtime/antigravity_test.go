package runtime_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	antigravityauth "github.com/xibodev/llm-provider-auth/antigravity"
	"github.com/xibodev/llm-provider-auth/tokenstore"
	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
	"github.com/xibodev/llmgw-core/translation"
)

// antigravityUpstreamChat is what reaches Cloud Code Assist for "Say hello"
// with max_tokens 64 and temperature 0.2, the body the gateway sends for
// the same Chat request. Only the random request ID is replaced.
const antigravityUpstreamChat = `{"model":"gemini-fixture","project":"fixture-project","request":{"contents":[{"role":"user","parts":[{"text":"Say hello"}]}],` +
	`"generationConfig":{"maxOutputTokens":64,"temperature":0.2}},"requestId":"<request>","requestType":"agent","userAgent":"antigravity"}`

const antigravityUpstreamEvents = `data: {"response":{"candidates":[{"content":{"parts":[{"text":"Hello from antigravity"}]},"finishReason":"STOP"}],` +
	`"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7}}}` + "\n\n"

type antigravityCall struct{ path, authorization, body string }

// antigravityBackend is a synthetic Cloud Code Assist with Google's token
// endpoint and account discovery. It accepts one access token at a time; a
// refresh issues the next one.
type antigravityBackend struct {
	mu       sync.Mutex
	accepted string
	calls    []antigravityCall
}

func (b *antigravityBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	normalized := regexp.MustCompile(`"requestId":"agent_[0-9a-f]{24}"`).ReplaceAllString(string(body), `"requestId":"<request>"`)
	b.mu.Lock()
	b.calls = append(b.calls, antigravityCall{path: r.URL.RequestURI(), authorization: r.Header.Get("Authorization"), body: normalized})
	accepted := r.Header.Get("Authorization") == "Bearer "+b.accepted
	b.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	switch {
	case r.URL.Path == "/token":
		// Google rotates the access token and keeps the refresh token.
		_, _ = io.WriteString(w, `{"access_token":"fixture-access-rotated","expires_in":3600,"token_type":"Bearer"}`)
	case !accepted:
		w.WriteHeader(http.StatusUnauthorized)
	case r.URL.Path == "/load":
		_, _ = io.WriteString(w, `{"cloudaicompanionProject":"fixture-project","paidTier":{"id":"fixture-tier"}}`)
	case r.URL.Path == "/v1internal:loadCodeAssist":
		_, _ = io.WriteString(w, `{"cloudaicompanionProject":"fixture-project"}`)
	case r.URL.Path == "/v1internal:fetchAvailableModels":
		_, _ = io.WriteString(w, `{"models":{"gemini-fixture":{"displayName":"Gemini Fixture","supportsThinking":true,"quotaInfo":{"remainingFraction":1}},`+
			`"gemini-fixture-image":{"displayName":"Gemini Fixture Image"}},"imageGenerationModelIds":["gemini-fixture-image"]}`)
	case r.URL.Path == "/v1internal:streamGenerateContent":
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, antigravityUpstreamEvents)
	default:
		http.NotFound(w, r)
	}
}

// take returns the calls since the last take.
func (b *antigravityBackend) take() []antigravityCall {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

// revoke makes the backend reject the current access token and accept only
// the one the next refresh issues.
func (b *antigravityBackend) revoke() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.accepted = "fixture-access-rotated"
}

// storeProject stores a discovered project as the gateway will: only while
// the record still holds the token it was discovered with and names no
// project, with a revision-fenced write.
func storeProject(store core.CredentialStore, stored *atomic.Int32) func(context.Context, *core.Credential, string) {
	return func(ctx context.Context, credential *core.Credential, projectID string) {
		record, err := store.Load(ctx, credential.ConnectionID)
		if err != nil || record.AccessToken != credential.Token || record.Metadata[core.CredentialMetadataProjectID] != "" {
			return
		}
		record.Metadata = maps.Clone(record.Metadata)
		if record.Metadata == nil {
			record.Metadata = map[string]string{}
		}
		record.Metadata[core.CredentialMetadataProjectID] = projectID
		if _, err := store.ReplaceIfCurrent(ctx, credential.ConnectionID, record.Revision, record); err == nil {
			stored.Add(1)
		}
	}
}

type antigravitySettings struct{ BaseURL string }

func newAntigravityRuntime(t *testing.T, server *httptest.Server, store core.CredentialStore, evidence core.EvidenceSink, stored *atomic.Int32) *coreruntime.Runtime[antigravitySettings] {
	t.Helper()
	runtime, err := coreruntime.New(coreruntime.Options[antigravitySettings]{
		Settings: coreruntime.NewMemorySettings(antigravitySettings{BaseURL: server.URL}),
		Providers: func(settings antigravitySettings, _ string) (core.Provider, error) {
			antigravity, err := providers.NewAntigravity(providers.AntigravityConfig{
				BaseURL: settings.BaseURL, Client: server.Client(), ProjectResolved: storeProject(store, stored),
			})
			if err != nil {
				return nil, err
			}
			return translation.Adapter{Provider: antigravity}, nil
		},
		Refresh: func(settings antigravitySettings, _ string) tokenstore.RefreshFunc {
			return providers.NewAntigravityRefresh(antigravityauth.Config{
				ClientID: "fixture-client", ClientSecret: "fixture-secret", ClientAuthMode: antigravityauth.ClientAuthModeClientSecretPost,
				HTTPClient: server.Client(), Endpoints: antigravityauth.Endpoints{TokenURL: settings.BaseURL + "/token", LoadCodeAssistURL: settings.BaseURL + "/load"},
			})
		},
		Credentials: store, Evidence: evidence,
	})
	if err != nil {
		t.Fatal(err)
	}
	return runtime
}

func TestRuntimeServesTheAntigravityVertical(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	backend := &antigravityBackend{accepted: "fixture-access"}
	server := httptest.NewServer(backend)
	defer server.Close()
	store := core.NewMemoryCredentialStore()
	// The login stored no project, as a login whose discovery failed does.
	if _, err := store.Save(ctx, "owner-antigravity", tokenstore.Record{
		AccessToken: "fixture-access", RefreshToken: "fixture-refresh", TokenType: "Bearer", AccountID: "account-fixture",
		Expiry: time.Now().Add(time.Hour), Metadata: map[string]string{
			core.CredentialMetadataOAuthProfile: providers.AntigravityOAuthProfileRuntimeSecret, core.CredentialMetadataOAuthClientID: "fixture-client",
		},
	}); err != nil {
		t.Fatal(err)
	}
	owner := core.Caller{ID: "owner", Kind: core.CallerHuman}
	store.Bind(owner, "antigravity", "owner-antigravity")
	evidence := &core.MemoryEvidenceSink{}
	stored := &atomic.Int32{}
	runtime := newAntigravityRuntime(t, server, store, evidence, stored)
	chat := core.Request{
		Surface: core.ModelSurfaceChatCompletions, Model: "gemini-fixture", ContentType: core.ContentTypeJSON,
		Body: []byte(`{"model":"gemini-fixture","stream":false,"max_tokens":64,"temperature":0.2,"top_p":0.9,"messages":[{"role":"user","content":"Say hello"}]}`),
	}
	inference := antigravityCall{path: "/v1internal:streamGenerateContent?alt=sse", authorization: "Bearer fixture-access", body: antigravityUpstreamChat}
	assertChat := func(t *testing.T, response core.Response) {
		t.Helper()
		var body struct {
			Object, Model string
			Choices       []struct {
				FinishReason string                   `json:"finish_reason"`
				Message      struct{ Content string } `json:"message"`
			} `json:"choices"`
			Usage struct {
				TotalTokens int `json:"total_tokens"`
			} `json:"usage"`
		}
		if json.Unmarshal(response.Body, &body) != nil || body.Object != "chat.completion" || body.Model != "gemini-fixture" || len(body.Choices) != 1 ||
			body.Choices[0].Message.Content != "Hello from antigravity" || body.Choices[0].FinishReason != "stop" || body.Usage.TotalTokens != 7 {
			t.Fatalf("chat response = %s", response.Body)
		}
	}

	t.Run("project discovery through the hook", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "antigravity", chat)
		if err != nil {
			t.Fatal(err)
		}
		assertChat(t, response)
		discovery := antigravityCall{
			path: "/v1internal:loadCodeAssist", authorization: "Bearer fixture-access",
			body: `{"metadata":{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}}`,
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []antigravityCall{discovery, inference}) {
			t.Fatalf("upstream = %+v", calls)
		}
		record, err := store.Load(ctx, "owner-antigravity")
		if err != nil || record.Metadata[core.CredentialMetadataProjectID] != "fixture-project" || stored.Load() != 1 {
			t.Fatalf("stored = %v %+v, err = %v, writes = %d", record, record.Metadata, err, stored.Load())
		}
	})

	// The stored project serves every later operation without discovery.
	t.Run("chat", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "antigravity", chat)
		if err != nil {
			t.Fatal(err)
		}
		assertChat(t, response)
		if len(response.Losses) != 1 || response.Losses[0].Path != "top_p" || response.Losses[0].Class != translate.LossDropped ||
			response.Losses[0].Severity != translate.LossAdvisory {
			t.Fatalf("losses = %+v, want top_p dropped as the gateway drops it", response.Losses)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []antigravityCall{inference}) || stored.Load() != 1 {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("messages through the adapter over native chat", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "antigravity", core.Request{
			Surface: core.ModelSurfaceMessages, Model: "gemini-fixture", ContentType: core.ContentTypeJSON,
			Body: []byte(`{"model":"gemini-fixture","stream":false,"max_tokens":64,"temperature":0.2,"messages":[{"role":"user","content":"Say hello"}]}`),
		})
		if err != nil {
			t.Fatal(err)
		}
		var body struct {
			Type    string                  `json:"type"`
			Content []struct{ Text string } `json:"content"`
		}
		if json.Unmarshal(response.Body, &body) != nil || body.Type != "message" || len(body.Content) != 1 || body.Content[0].Text != "Hello from antigravity" {
			t.Fatalf("messages response = %s", response.Body)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []antigravityCall{inference}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	// Antigravity reads Cloud Code Assist's stream to its end, so it
	// streams nothing, as the gateway does, and sends nothing trying.
	t.Run("a stream is refused", func(t *testing.T) {
		stream := chat
		stream.Body = []byte(`{"model":"gemini-fixture","stream":true,"messages":[{"role":"user","content":"Say hello"}]}`)
		if _, err := runtime.Stream(ctx, owner, "antigravity", stream); !errors.Is(err, providers.ErrExperimentalAntigravityStreamingUnsupported) ||
			core.ClassifyError(err).Disposition() != core.DispositionFailover {
			t.Fatalf("err = %v, want a refusal that permits failover", err)
		}
		if calls := backend.take(); len(calls) != 0 {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("catalog", func(t *testing.T) {
		record, err := runtime.ListModels(ctx, owner, "antigravity")
		if err != nil {
			t.Fatal(err)
		}
		models := record.Evidence.Models
		if record.Evidence.Status != core.CatalogDiscovered || len(models) != 2 || models[0].ID != "gemini-fixture" || models[1].ID != "gemini-fixture-image" ||
			!slices.Equal(models[0].SupportedAPIs, []string{"/v1/chat/completions"}) ||
			!slices.Equal(models[1].SupportedAPIs, []string{"/v1/chat/completions", "/v1/images/generations"}) ||
			models[0].Capabilities.Reasoning != core.SupportSupported || models[0].Capabilities.Streaming != core.SupportUnsupported ||
			models[1].Capabilities.Operations.Image != core.SupportSupported {
			t.Fatalf("catalog = %+v", record)
		}
		catalog := antigravityCall{path: "/v1internal:fetchAvailableModels", authorization: "Bearer fixture-access", body: `{"project":"fixture-project"}`}
		if calls := backend.take(); !reflect.DeepEqual(calls, []antigravityCall{catalog}) {
			t.Fatalf("catalog upstream = %+v", calls)
		}
	})

	t.Run("a rejected token is refreshed once and the request replayed", func(t *testing.T) {
		backend.revoke()
		response, err := runtime.Invoke(ctx, owner, "antigravity", chat)
		if err != nil {
			t.Fatal(err)
		}
		assertChat(t, response)
		refresh := antigravityCall{path: "/token", body: "client_id=fixture-client&client_secret=fixture-secret&grant_type=refresh_token&refresh_token=fixture-refresh"}
		discovery := antigravityCall{path: "/load", authorization: "Bearer fixture-access-rotated", body: `{"metadata":{"ideType":"ANTIGRAVITY"}}`}
		replay := inference
		replay.authorization = "Bearer fixture-access-rotated"
		if calls := backend.take(); !reflect.DeepEqual(calls, []antigravityCall{inference, refresh, discovery, replay}) {
			t.Fatalf("upstream = %+v", calls)
		}
		record, err := store.Load(ctx, "owner-antigravity")
		if err != nil || record.AccessToken != "fixture-access-rotated" || record.RefreshToken != "fixture-refresh" || record.AccountID != "account-fixture" ||
			record.Metadata[core.CredentialMetadataProjectID] != "fixture-project" ||
			record.Metadata[core.CredentialMetadataOAuthProfile] != providers.AntigravityOAuthProfileRuntimeSecret ||
			record.Expiry.Before(time.Now().Add(50*time.Minute)) {
			t.Fatalf("stored = %v %+v, err = %v", record, record.Metadata, err)
		}
		records := evidence.Records()
		if last := records[len(records)-1]; last.CredentialRevision != record.Revision || last.AccountID != "account-fixture" ||
			last.Outcome.Status != core.ProviderHealthHealthy {
			t.Fatalf("evidence = %+v", last)
		}
	})

	if health := runtime.Health("antigravity"); health.Status != core.ProviderHealthHealthy || health.ErrorClass != core.ProviderErrorNone {
		t.Fatalf("health = %+v", health)
	}
}
