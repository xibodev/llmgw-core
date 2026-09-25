package runtime_test

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/xibodev/llm-provider-auth/tokenstore"
	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
	"github.com/xibodev/llmgw-core/translation"
)

// The bodies the gateway sends Google for "Say hello" with a system prompt,
// max_tokens 64 and temperature 0.2, encoded as the gateway encodes them.
const (
	googleUpstreamChat = `{"contents":[{"parts":[{"text":"Say hello"}],"role":"user"}],"generationConfig":{"maxOutputTokens":64,"temperature":0.2},` +
		`"systemInstruction":{"parts":[{"text":"Answer \u003cbriefly\u003e \u0026 kindly."}]}}`
	googleUpstreamAnswer = `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello from Gemini"}]},"finishReason":"STOP"}],` +
		`"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7},"modelVersion":"gemini-fixture-001"}`
	// What the gateway returns for that answer.
	googleChatCompletion = `{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"Hello from Gemini","role":"assistant"}}],` +
		`"id":"chatcmpl-google","model":"gemini-fixture-001","object":"chat.completion","usage":{"completion_tokens":4,"prompt_tokens":3,"total_tokens":7}}`
	googleVertexModel = "/v1/projects/fixture-project/locations/global/publishers/google/models/"
)

type googleRuntimeCall struct{ method, uri, authorization, requestType, body string }

// googleBackend is a synthetic AI Studio, Vertex AI and Google token
// endpoint. AI Studio accepts one API key and Vertex AI one minted token.
type googleBackend struct {
	mu    sync.Mutex
	calls []googleRuntimeCall
}

func (b *googleBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	authorization := r.Header.Get("Authorization")
	if key := r.Header.Get("x-goog-api-key"); key != "" {
		authorization = "key " + key
	}
	call := googleRuntimeCall{method: r.Method, uri: r.URL.RequestURI(), authorization: authorization, requestType: r.Header.Get("X-Vertex-AI-LLM-Request-Type"), body: string(body)}
	if r.URL.Path == "/token" {
		call.body = "" // A signed assertion, fresh each time.
	}
	b.mu.Lock()
	b.calls = append(b.calls, call)
	b.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	studio, vertex := authorization == "key fixture-studio-key", authorization == "Bearer ya29.fixture-minted"
	switch {
	case r.URL.Path == "/token":
		_, _ = io.WriteString(w, `{"access_token":"ya29.fixture-minted","expires_in":3600,"token_type":"Bearer"}`)
	case r.URL.Path == googleVertexModel+"gemini-rejected:generateContent":
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"code":401,"status":"UNAUTHENTICATED","message":"Request had invalid authentication credentials."}}`)
	case !studio && !vertex:
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, `{"error":{"code":403,"status":"PERMISSION_DENIED","message":"denied"}}`)
	case strings.HasSuffix(r.URL.Path, ":generateContent"):
		_, _ = io.WriteString(w, googleUpstreamAnswer)
	case strings.HasSuffix(r.URL.Path, ":embedContent"):
		_, _ = io.WriteString(w, `{"embedding":{"values":[0.25,-0.5]},"usageMetadata":{"promptTokenCount":2}}`)
	case strings.HasSuffix(r.URL.Path, ":predict"):
		_, _ = io.WriteString(w, `{"predictions":[{"embeddings":{"values":[0.75],"statistics":{"token_count":3}}}]}`)
	case r.URL.Path == "/v1beta/models" && r.URL.Query().Get("pageToken") == "":
		_, _ = io.WriteString(w, `{"models":[{"name":"models/gemini-fixture","displayName":"Gemini Fixture","supportedGenerationMethods":["generateContent","countTokens"]}],"nextPageToken":"page+2/=="}`)
	case r.URL.Path == "/v1beta/models":
		_, _ = io.WriteString(w, `{"models":[{"name":"models/gemini-embedding-fixture","supportedGenerationMethods":["embedContent"]}]}`)
	case r.URL.Path == "/v1beta1/publishers/google/models":
		_, _ = io.WriteString(w, `{"publisherModels":[{"name":"publishers/google/models/gemini-fixture","supportedActions":{"openGenerationAiStudio":{}}},`+
			`{"name":"publishers/google/models/deployable-fixture","supportedActions":{"deploy":{}}}]}`)
	default:
		http.NotFound(w, r)
	}
}

// take returns the calls since the last take.
func (b *googleBackend) take() []googleRuntimeCall {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

// googleServiceAccountKey is a synthetic service-account key for
// fixture-project whose token endpoint is tokenURI.
func googleServiceAccountKey(t *testing.T, tokenURI string) string {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(map[string]string{
		"type": "service_account", "project_id": "fixture-project", "private_key_id": "key-1",
		"private_key":  string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"client_email": "svc@fixture-project.iam.example.test", "token_uri": tokenURI,
	})
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

type googleSettings struct{ BaseURL string }

// newGoogleRuntime serves three instances: AI Studio, Vertex AI in the
// project a credential names, and Vertex AI configured for another one.
func newGoogleRuntime(t *testing.T, server *httptest.Server, store core.CredentialStore, evidence core.EvidenceSink) *coreruntime.Runtime[googleSettings] {
	t.Helper()
	runtime, err := coreruntime.New(coreruntime.Options[googleSettings]{
		Settings: coreruntime.NewMemorySettings(googleSettings{BaseURL: server.URL}),
		Providers: func(settings googleSettings, instance string) (core.Provider, error) {
			config := providers.GoogleConfig{Deployment: providers.GoogleVertexAI, BaseURL: settings.BaseURL + "/v1", RequestType: "paygo", Client: server.Client()}
			switch instance {
			case "ai-studio":
				config = providers.GoogleConfig{Deployment: providers.GoogleAIStudio, BaseURL: settings.BaseURL + "/v1beta", Client: server.Client()}
			case "vertex-other-project":
				config.Project = "another-project"
			}
			google, err := providers.NewGoogle(config)
			if err != nil {
				return nil, err
			}
			return translation.Adapter{Provider: google}, nil
		},
		Credentials: store, Evidence: evidence,
	})
	if err != nil {
		t.Fatal(err)
	}
	return runtime
}

// googleStore holds the AI Studio key of owner and a service-account key
// every caller of the Vertex AI instances shares, as a system connection.
func googleStore(t *testing.T, server *httptest.Server) *core.MemoryCredentialStore {
	t.Helper()
	ctx := context.Background()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(ctx, "owner-studio", core.APIKeyRecord("fixture-studio-key")); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Save(ctx, "system-vertex", tokenstore.Record{
		AccessToken: googleServiceAccountKey(t, server.URL+"/token"), TokenType: core.TokenTypeGCPServiceAccount,
	}); err != nil {
		t.Fatal(err)
	}
	store.Bind(core.Caller{ID: "owner", Kind: core.CallerHuman}, "ai-studio", "owner-studio")
	store.BindShared("vertex", "system-vertex")
	store.BindShared("vertex-other-project", "system-vertex")
	return store
}

func TestRuntimeServesTheGoogleVertical(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	backend := &googleBackend{}
	server := httptest.NewServer(backend)
	defer server.Close()
	evidence := &core.MemoryEvidenceSink{}
	runtime := newGoogleRuntime(t, server, googleStore(t, server), evidence)
	owner := core.Caller{ID: "owner", Kind: core.CallerHuman}
	// A gateway API key acts for no principal, so only the shared system
	// credential serves it.
	system := core.Caller{Kind: core.CallerService}
	chat := core.Request{
		Surface: core.ModelSurfaceChatCompletions, Model: "gemini-fixture", ContentType: core.ContentTypeJSON,
		Body: []byte(`{"model":"gemini-fixture","stream":false,"max_tokens":64,"temperature":0.2,"top_p":0.9,` +
			`"messages":[{"role":"system","content":"Answer <briefly> & kindly."},{"role":"user","content":"Say hello"}]}`),
	}
	expect := func(t *testing.T, want ...googleRuntimeCall) {
		t.Helper()
		if calls := backend.take(); !slices.Equal(calls, want) {
			t.Fatalf("upstream = %+v\nwant %+v", calls, want)
		}
	}
	studioKey, minted := "key fixture-studio-key", "Bearer ya29.fixture-minted"

	t.Run("chat on AI Studio", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "ai-studio", chat)
		if err != nil {
			t.Fatal(err)
		}
		if string(response.Body) != googleChatCompletion || len(response.Losses) != 1 || response.Losses[0].Path != "top_p" ||
			response.Losses[0].Severity != translate.LossAdvisory {
			t.Fatalf("response = %s, losses = %+v", response.Body, response.Losses)
		}
		expect(t, googleRuntimeCall{method: http.MethodPost, uri: "/v1beta/models/gemini-fixture:generateContent", authorization: studioKey, body: googleUpstreamChat})
	})

	t.Run("messages through the adapter over native chat", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "ai-studio", core.Request{
			Surface: core.ModelSurfaceMessages, Model: "gemini-fixture", ContentType: core.ContentTypeJSON,
			Body: []byte(`{"model":"gemini-fixture","max_tokens":64,"temperature":0.2,"system":"Answer <briefly> & kindly.","messages":[{"role":"user","content":"Say hello"}]}`),
		})
		if err != nil {
			t.Fatal(err)
		}
		var body struct {
			Type    string                  `json:"type"`
			Content []struct{ Text string } `json:"content"`
		}
		if json.Unmarshal(response.Body, &body) != nil || body.Type != "message" || len(body.Content) != 1 || body.Content[0].Text != "Hello from Gemini" {
			t.Fatalf("messages response = %s", response.Body)
		}
		expect(t, googleRuntimeCall{method: http.MethodPost, uri: "/v1beta/models/gemini-fixture:generateContent", authorization: studioKey, body: googleUpstreamChat})
	})

	t.Run("chat on Vertex AI with a service account", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, system, "vertex", chat)
		if err != nil {
			t.Fatal(err)
		}
		if string(response.Body) != googleChatCompletion {
			t.Fatalf("response = %s", response.Body)
		}
		// The project comes from the key, and the key is exchanged once.
		expect(t, googleRuntimeCall{method: http.MethodPost, uri: "/token"},
			googleRuntimeCall{method: http.MethodPost, uri: googleVertexModel + "gemini-fixture:generateContent", authorization: minted, requestType: "shared", body: googleUpstreamChat})
	})

	t.Run("embeddings", func(t *testing.T) {
		embed := func(caller core.Caller, instance, model, input string) string {
			t.Helper()
			response, err := runtime.Invoke(ctx, caller, instance, core.Request{
				Surface: core.ModelSurfaceEmbeddings, Model: model, ContentType: core.ContentTypeJSON,
				Body: []byte(`{"model":"` + model + `","input":` + input + `}`),
			})
			if err != nil {
				t.Fatal(err)
			}
			return string(response.Body)
		}
		if body := embed(owner, "ai-studio", "gemini-embedding-fixture", `"Say hello"`); body !=
			`{"data":[{"embedding":[0.25,-0.5],"index":0,"object":"embedding"}],"model":"gemini-embedding-fixture","object":"list","usage":{"prompt_tokens":2,"total_tokens":2}}` {
			t.Fatalf("AI Studio embeddings = %s", body)
		}
		expect(t, googleRuntimeCall{method: http.MethodPost, uri: "/v1beta/models/gemini-embedding-fixture:embedContent", authorization: studioKey,
			body: `{"content":{"parts":[{"text":"Say hello"}]}}`})
		if body := embed(system, "vertex", "text-embedding-fixture", `["Say hello"]`); body !=
			`{"data":[{"embedding":[0.75],"index":0,"object":"embedding"}],"model":"text-embedding-fixture","object":"list","usage":{"prompt_tokens":3,"total_tokens":3}}` {
			t.Fatalf("Vertex AI embeddings = %s", body)
		}
		expect(t, googleRuntimeCall{method: http.MethodPost, uri: googleVertexModel + "text-embedding-fixture:predict", authorization: minted, requestType: "shared",
			body: `{"instances":[{"content":"Say hello"}]}`})
	})

	t.Run("catalogs", func(t *testing.T) {
		record, err := runtime.ListModels(ctx, owner, "ai-studio")
		if err != nil {
			t.Fatal(err)
		}
		models := record.Evidence.Models
		if record.Evidence.Status != core.CatalogDiscovered || len(models) != 2 || models[0].ID != "gemini-fixture" || models[0].DisplayName != "Gemini Fixture" ||
			models[0].Vendor != "google" || !slices.Equal(models[0].SupportedAPIs, []string{"/v1/chat/completions", "/v1/messages"}) ||
			models[0].Capabilities.Surfaces.ChatCompletions != core.SupportSupported || models[0].Capabilities.Surfaces.Messages != core.SupportSupported ||
			models[1].ID != "gemini-embedding-fixture" || models[1].Capabilities.Operations.Embeddings != core.SupportSupported {
			t.Fatalf("AI Studio catalog = %+v", record)
		}
		expect(t, googleRuntimeCall{method: http.MethodGet, uri: "/v1beta/models?pageSize=1000", authorization: studioKey},
			googleRuntimeCall{method: http.MethodGet, uri: "/v1beta/models?pageSize=1000&pageToken=page%2B2%2F%3D%3D", authorization: studioKey})
		// Discovery needs the OAuth principal, and is no invocation.
		record, err = runtime.ListModels(ctx, system, "vertex")
		if err != nil {
			t.Fatal(err)
		}
		if models := record.Evidence.Models; len(models) != 1 || models[0].ID != "gemini-fixture" || models[0].Capabilities.Operations.Chat != core.SupportSupported {
			t.Fatalf("Vertex AI catalog = %+v", record)
		}
		expect(t, googleRuntimeCall{method: http.MethodGet, uri: "/v1beta1/publishers/google/models?pageSize=200", authorization: minted})
	})

	t.Run("a stream is refused and nothing is sent", func(t *testing.T) {
		stream := chat
		stream.Body = []byte(`{"model":"gemini-fixture","stream":true,"messages":[{"role":"user","content":"Say hello"}]}`)
		if _, err := runtime.Stream(ctx, owner, "ai-studio", stream); core.ClassifyError(err).Disposition() != core.DispositionFailover {
			t.Fatalf("err = %v, want a refusal that permits failover", err)
		}
		expect(t)
	})

	t.Run("a key for another project is refused before anything is sent", func(t *testing.T) {
		_, err := runtime.Invoke(ctx, system, "vertex-other-project", chat)
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration || core.ClassifyError(err).Disposition() != core.DispositionFailover ||
			!strings.Contains(err.Error(), `"another-project" does not match the service account project "fixture-project"`) {
			t.Fatalf("err = %v", err)
		}
		expect(t)
	})

	// A service account has no refresh token, so the Runtime cannot replay
	// a request Vertex AI rejects; the rejection keeps its status.
	t.Run("a rejected service account is not replayed", func(t *testing.T) {
		rejected := chat
		rejected.Model = "gemini-rejected"
		_, err := runtime.Invoke(ctx, system, "vertex", rejected)
		if classification := core.ClassifyError(err); classification.StatusCode != http.StatusUnauthorized || classification.Disposition() != core.DispositionTerminal ||
			!strings.Contains(err.Error(), "credential rejected") {
			t.Fatalf("err = %v, classification = %+v", err, classification)
		}
		expect(t, googleRuntimeCall{method: http.MethodPost, uri: googleVertexModel + "gemini-rejected:generateContent", authorization: minted, requestType: "shared", body: googleUpstreamChat})
	})

	// Only upstream outcomes change health.
	if health := runtime.Health("ai-studio"); health.Status != core.ProviderHealthHealthy {
		t.Fatalf("AI Studio health = %+v", health)
	}
	if health := runtime.Health("vertex"); health.Status != core.ProviderHealthUnhealthy || health.ErrorClass != core.ProviderErrorAuth {
		t.Fatalf("Vertex AI health = %+v", health)
	}
	if health := runtime.Health("vertex-other-project"); health.Status != core.ProviderHealthUnknown {
		t.Fatalf("misconfigured Vertex AI health = %+v", health)
	}
	for _, record := range evidence.Records() {
		if record.Instance != "ai-studio" && record.CredentialKey != "system-vertex" {
			t.Fatalf("evidence = %+v, want the shared service account", record)
		}
	}
}
