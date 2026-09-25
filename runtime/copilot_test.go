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
	"time"

	copilotauth "github.com/xibodev/llm-provider-auth/copilot"
	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
	"github.com/xibodev/llmgw-core/translation"
)

const (
	copilotCatalog = `{"object":"list","data":[` +
		`{"id":"gpt-fixture","name":"GPT Fixture","vendor":"Fixture Vendor","supported_endpoints":["/chat/completions","/responses"],` +
		`"capabilities":{"type":"chat","limits":{"max_context_window_tokens":128000},"supports":{"streaming":true,"tool_calls":true,"vision":true}}},` +
		`{"id":"legacy-fixture","name":"Legacy Fixture","vendor":"Fixture Vendor","capabilities":{"type":"chat","supports":{"streaming":true}}}]}`
	copilotChatAnswer = `{"id":"chatcmpl_fixture","object":"chat.completion","model":"gpt-fixture",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"Hello from copilot"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`
	copilotChatEvents = "data: {\"id\":\"chatcmpl_fixture\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n" +
		"data: {\"id\":\"chatcmpl_fixture\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" from copilot\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	copilotResponsesAnswer = `{"id":"resp_fixture","object":"response","status":"completed","model":"gpt-fixture",` +
		`"output":[{"type":"message","id":"msg_fixture","role":"assistant","content":[{"type":"output_text","text":"Hello from copilot"}]}]}`
)

type copilotAPICall struct{ path, authorization, body string }

// copilotUpstream is a synthetic GitHub session-token exchange and, on its
// own server, the Copilot API that the sessions name as their base. Each
// session is accepted until revokeSessions, and each OAuth token until it
// is revoked.
type copilotUpstream struct {
	github, api *httptest.Server

	mu        sync.Mutex
	issued    int
	exchanges []string
	calls     []copilotAPICall
	sessions  map[string]bool
	revoked   map[string]bool
}

func newCopilotUpstream(t *testing.T) *copilotUpstream {
	t.Helper()
	upstream := &copilotUpstream{sessions: map[string]bool{}, revoked: map[string]bool{}}
	upstream.github = httptest.NewServer(http.HandlerFunc(upstream.exchange))
	upstream.api = httptest.NewServer(http.HandlerFunc(upstream.serve))
	t.Cleanup(upstream.github.Close)
	t.Cleanup(upstream.api.Close)
	return upstream
}

func (u *copilotUpstream) exchange(w http.ResponseWriter, r *http.Request) {
	oauth := strings.TrimPrefix(r.Header.Get("Authorization"), "token ")
	u.mu.Lock()
	u.exchanges = append(u.exchanges, oauth)
	u.issued++
	session, revoked := fmt.Sprintf("%s-session-%d", oauth, u.issued), u.revoked[oauth]
	u.sessions[session] = !revoked
	u.mu.Unlock()
	if revoked || r.URL.Path != "/copilot_internal/v2/token" {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"token": session, "expires_at": time.Now().Add(30 * time.Minute).Unix(), "endpoints": map[string]any{"api": u.api.URL},
	})
}

func (u *copilotUpstream) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	authorization := r.Header.Get("Authorization")
	u.mu.Lock()
	u.calls = append(u.calls, copilotAPICall{path: r.URL.Path, authorization: authorization, body: string(body)})
	accepted := u.sessions[strings.TrimPrefix(authorization, "Bearer ")]
	u.mu.Unlock()
	switch {
	case !accepted:
		w.WriteHeader(http.StatusUnauthorized)
	case r.URL.Path == "/models":
		_, _ = io.WriteString(w, copilotCatalog)
	case r.URL.Path == "/chat/completions" && strings.Contains(string(body), `"stream":true`):
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, copilotChatEvents)
	case r.URL.Path == "/chat/completions":
		_, _ = io.WriteString(w, copilotChatAnswer)
	case r.URL.Path == "/responses":
		_, _ = io.WriteString(w, copilotResponsesAnswer)
	default:
		http.NotFound(w, r)
	}
}

// take returns the exchanges and API calls since the last take.
func (u *copilotUpstream) take() ([]string, []copilotAPICall) {
	u.mu.Lock()
	defer u.mu.Unlock()
	exchanges, calls := u.exchanges, u.calls
	u.exchanges, u.calls = nil, nil
	return exchanges, calls
}

// revokeSessions makes the API reject every session issued so far.
func (u *copilotUpstream) revokeSessions() {
	u.mu.Lock()
	defer u.mu.Unlock()
	clear(u.sessions)
}

func (u *copilotUpstream) revokeOAuth(token string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.revoked[token] = true
}

type copilotSettings struct{ ExchangeURL string }

func newCopilotRuntime(t *testing.T, upstream *copilotUpstream, store core.CredentialStore, evidence core.EvidenceSink) *coreruntime.Runtime[copilotSettings] {
	t.Helper()
	runtime, err := coreruntime.New(coreruntime.Options[copilotSettings]{
		Settings: coreruntime.NewMemorySettings(copilotSettings{ExchangeURL: upstream.github.URL + "/copilot_internal/v2/token"}),
		Providers: func(settings copilotSettings, _ string) (core.Provider, error) {
			auth := copilotauth.New(copilotauth.Config{
				AllowProxy: true, OAuthToken: "product-oauth", HTTPClient: upstream.github.Client(),
				Endpoints: copilotauth.Endpoints{SessionTokenURL: settings.ExchangeURL},
			})
			copilot, err := providers.NewCopilot(providers.CopilotConfig{
				Auth: auth, EditorPluginVersion: "fixture-product/1.0", UserAgent: "FixtureCopilotChat/1.0", Client: upstream.api.Client(),
			})
			if err != nil {
				return nil, err
			}
			return translation.Adapter{Provider: copilot}, nil
		},
		// A product whose GitHub tokens rotate supplies the rotation; the
		// Runtime calls it when Copilot rejects a caller's token.
		Refresh: func(copilotSettings, string) tokenstore.RefreshFunc {
			return func(_ context.Context, record tokenstore.Record) (tokenstore.Record, error) {
				return tokenstore.Record{AccessToken: record.AccessToken + "-rotated", TokenType: "Bearer"}, nil
			}
		},
		Credentials: store, Evidence: evidence,
	})
	if err != nil {
		t.Fatal(err)
	}
	return runtime
}

func assertCopilotUpstream(t *testing.T, upstream *copilotUpstream, exchanges []string, calls ...copilotAPICall) {
	t.Helper()
	gotExchanges, gotCalls := upstream.take()
	if !reflect.DeepEqual(gotExchanges, exchanges) || !reflect.DeepEqual(gotCalls, calls) {
		t.Fatalf("exchanges = %v calls = %+v\nwant %v %+v", gotExchanges, gotCalls, exchanges, calls)
	}
}

func readCopilotStream(t *testing.T, stream core.StreamIter) string {
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
func TestRuntimeServesTheCopilotVertical(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	upstream := newCopilotUpstream(t)
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(ctx, "owner-copilot", tokenstore.Record{AccessToken: "owner-oauth", RefreshToken: "owner-refresh", TokenType: "Bearer"}); err != nil {
		t.Fatal(err)
	}
	owner, guest := core.Caller{ID: "owner", Kind: core.CallerHuman}, core.Caller{ID: "guest", Kind: core.CallerHuman}
	store.Bind(owner, "copilot", "owner-copilot")
	evidence := &core.MemoryEvidenceSink{}
	runtime := newCopilotRuntime(t, upstream, store, evidence)
	request := func(surface core.ModelSurface, model, body string) core.Request {
		return core.Request{Surface: surface, Model: model, ContentType: core.ContentTypeJSON, Body: []byte(body)}
	}
	chat := request(core.ModelSurfaceChatCompletions, "gpt-fixture", `{"model":"gpt-fixture","stream":false,"messages":[{"role":"user","content":"Say hello"}]}`)
	// The bodies the gateway's characterization goldens record for the same
	// requests to its OpenAI transport, which Copilot shares.
	chatUpstream := `{"messages":[{"content":"Say hello","role":"user"}],"model":"gpt-fixture","stream":false}`
	messagesUpstream := `{"max_tokens":64,"messages":[{"content":"Say hello","role":"user"}],"model":"gpt-fixture","stream":false}`
	session := "Bearer owner-oauth-session-1"

	t.Run("catalog", func(t *testing.T) {
		record, err := runtime.ListModels(ctx, owner, "copilot")
		if err != nil {
			t.Fatal(err)
		}
		models := record.Evidence.Models
		if record.Evidence.Status != core.CatalogDiscovered || len(models) != 2 || models[0].ID != "gpt-fixture" ||
			!reflect.DeepEqual(models[0].SupportedAPIs, []string{"/chat/completions", "/responses"}) ||
			models[0].Capabilities.Surfaces.Responses != core.SupportSupported || models[0].Capabilities.Inputs.Image != core.SupportSupported ||
			models[1].SupportedAPIs != nil || models[1].Capabilities.Surfaces.ChatCompletions != core.SupportUnknown {
			t.Fatalf("catalog = %+v", record)
		}
		assertCopilotUpstream(t, upstream, []string{"owner-oauth"}, copilotAPICall{path: "/models", authorization: session})
	})

	// The owner's session serves every request below without an exchange.
	t.Run("chat", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "copilot", chat)
		if err != nil || string(response.Body) != copilotChatAnswer {
			t.Fatalf("response = %s err = %v", response.Body, err)
		}
		assertCopilotUpstream(t, upstream, nil, copilotAPICall{path: "/chat/completions", authorization: session, body: chatUpstream})
	})

	t.Run("chat stream", func(t *testing.T) {
		stream, err := runtime.Stream(ctx, owner, "copilot", chat)
		if err != nil {
			t.Fatal(err)
		}
		if frames := readCopilotStream(t, stream); frames != copilotChatEvents {
			t.Fatalf("frames = %q", frames)
		}
		streamed := strings.Replace(chatUpstream, `"stream":false`, `"stream":true`, 1)
		assertCopilotUpstream(t, upstream, nil, copilotAPICall{path: "/chat/completions", authorization: session, body: streamed})
	})

	t.Run("responses, native for a model the catalog lists", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "copilot", request(core.ModelSurfaceResponses, "gpt-fixture", `{"model":"gpt-fixture","input":"Say hello"}`))
		if err != nil || string(response.Body) != copilotResponsesAnswer {
			t.Fatalf("response = %s err = %v", response.Body, err)
		}
		assertCopilotUpstream(t, upstream, nil, copilotAPICall{path: "/responses", authorization: session, body: `{"input":"Say hello","model":"gpt-fixture","stream":false}`})
	})

	t.Run("responses through the adapter for a model it does not", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "copilot", request(core.ModelSurfaceResponses, "legacy-fixture",
			`{"model":"legacy-fixture","stream":false,"input":"Say hello","max_output_tokens":64}`))
		if err != nil || !strings.Contains(string(response.Body), `"text":"Hello from copilot"`) {
			t.Fatalf("response = %s err = %v", response.Body, err)
		}
		// The gateway sends a Responses request's output limit to a Chat
		// model as max_completion_tokens.
		legacy := `{"max_completion_tokens":64,"messages":[{"content":"Say hello","role":"user"}],"model":"legacy-fixture","stream":false}`
		assertCopilotUpstream(t, upstream, nil, copilotAPICall{path: "/chat/completions", authorization: session, body: legacy})
	})

	t.Run("messages through the adapter", func(t *testing.T) {
		messages := request(core.ModelSurfaceMessages, "gpt-fixture", `{"model":"gpt-fixture","stream":false,"max_tokens":64,"messages":[{"role":"user","content":"Say hello"}]}`)
		response, err := runtime.Invoke(ctx, owner, "copilot", messages)
		if err != nil || !strings.Contains(string(response.Body), `"type":"message"`) || !strings.Contains(string(response.Body), "Hello from copilot") {
			t.Fatalf("response = %s err = %v", response.Body, err)
		}
		stream, err := runtime.Stream(ctx, owner, "copilot", messages)
		if err != nil {
			t.Fatal(err)
		}
		if frames := readCopilotStream(t, stream); !strings.Contains(frames, "message_stop") || !strings.Contains(frames, " from copilot") {
			t.Fatalf("frames = %q", frames)
		}
		streamed := strings.Replace(messagesUpstream, `"stream":false`, `"stream":true`, 1)
		assertCopilotUpstream(t, upstream, nil,
			copilotAPICall{path: "/chat/completions", authorization: session, body: messagesUpstream},
			copilotAPICall{path: "/chat/completions", authorization: session, body: streamed})
	})
	t.Run("a caller without a credential uses the product's token", func(t *testing.T) {
		if _, err := runtime.Invoke(ctx, guest, "copilot", chat); err != nil {
			t.Fatal(err)
		}
		assertCopilotUpstream(t, upstream, []string{"product-oauth"},
			copilotAPICall{path: "/chat/completions", authorization: "Bearer product-oauth-session-2", body: chatUpstream})
	})

	t.Run("a rejected session is replaced once and the request replayed", func(t *testing.T) {
		upstream.revokeSessions()
		if response, err := runtime.Invoke(ctx, owner, "copilot", chat); err != nil || string(response.Body) != copilotChatAnswer {
			t.Fatalf("response = %s err = %v", response.Body, err)
		}
		assertCopilotUpstream(t, upstream, []string{"owner-oauth"},
			copilotAPICall{path: "/chat/completions", authorization: session, body: chatUpstream},
			copilotAPICall{path: "/chat/completions", authorization: "Bearer owner-oauth-session-3", body: chatUpstream})
	})

	// GitHub rejecting the OAuth token fails with 401, which the Runtime
	// answers by refreshing the credential once and replaying.
	t.Run("a rejected OAuth token is refreshed by the Runtime", func(t *testing.T) {
		upstream.revokeOAuth("owner-oauth")
		upstream.revokeSessions()
		if response, err := runtime.Invoke(ctx, owner, "copilot", chat); err != nil || string(response.Body) != copilotChatAnswer {
			t.Fatalf("response = %s err = %v", response.Body, err)
		}
		assertCopilotUpstream(t, upstream, []string{"owner-oauth", "owner-oauth-rotated"},
			copilotAPICall{path: "/chat/completions", authorization: "Bearer owner-oauth-session-3", body: chatUpstream},
			copilotAPICall{path: "/chat/completions", authorization: "Bearer owner-oauth-rotated-session-5", body: chatUpstream})
		stored, err := store.Load(ctx, "owner-copilot")
		if err != nil || stored.AccessToken != "owner-oauth-rotated" {
			t.Fatalf("stored = %v err = %v", stored, err)
		}
		records := evidence.Records()
		if last := records[len(records)-1]; last.CredentialRevision != stored.Revision || last.Outcome.Status != core.ProviderHealthHealthy {
			t.Fatalf("evidence = %+v", last)
		}
	})

	// The product's own token has no credential to refresh, so the Runtime
	// reports the rejection.
	t.Run("a rejected product token is reported", func(t *testing.T) {
		upstream.revokeOAuth("product-oauth")
		_, err := runtime.Invoke(ctx, guest, "copilot", chat)
		if !errors.Is(err, copilotauth.ErrOAuthTokenRejected) || core.ClassifyError(err) != (core.ProviderErrorClassification{StatusCode: http.StatusUnauthorized}) {
			t.Fatalf("err = %v classification = %+v", err, core.ClassifyError(err))
		}
		assertCopilotUpstream(t, upstream, []string{"product-oauth"})
		if health := runtime.Health("copilot"); health.Status != core.ProviderHealthUnhealthy || health.ErrorClass != core.ProviderErrorAuth {
			t.Fatalf("health = %+v", health)
		}
	})
}
