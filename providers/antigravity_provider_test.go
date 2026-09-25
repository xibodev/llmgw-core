package providers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"slices"
	"sync"
	"testing"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// antigravityCall is one request that reached a fake Cloud Code Assist.
type antigravityCall struct {
	path   string
	header http.Header
	body   string
}

const antigravityFixtureStream = `data: {"response":{"candidates":[{"content":{"parts":[{"text":"Hello from "},{"text":"antigravity"}]},"finishReason":"STOP"}],` +
	`"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7}}}` + "\n\ndata: [DONE]\n\n"

// antigravityUpstream serves a discovered project, the current catalog
// fixture and a completion.
func antigravityUpstream(t *testing.T) http.HandlerFunc {
	catalog := readAntigravityFixture(t, "antigravity", "catalog-current.json")
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1internal:loadCodeAssist":
			_, _ = io.WriteString(w, `{"cloudaicompanionProject":"fixture-project-discovered"}`)
		case "/v1internal:fetchAvailableModels":
			_, _ = w.Write(catalog)
		case "/v1internal:streamGenerateContent":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, antigravityFixtureStream)
		default:
			http.NotFound(w, r)
		}
	}
}

// recordingAntigravityServer records each request before handler answers
// it. The returned function takes the requests recorded so far.
func recordingAntigravityServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, func() []antigravityCall) {
	t.Helper()
	var mu sync.Mutex
	var calls []antigravityCall
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		calls = append(calls, antigravityCall{path: r.URL.RequestURI(), header: r.Header.Clone(), body: string(body)})
		mu.Unlock()
		handler(w, r)
	}))
	t.Cleanup(server.Close)
	return server, func() []antigravityCall {
		mu.Lock()
		defer mu.Unlock()
		taken := calls
		calls = nil
		return taken
	}
}

func newFixtureAntigravity(t *testing.T, server *httptest.Server, resolved func(context.Context, *core.Credential, string)) *Antigravity {
	t.Helper()
	provider, err := NewAntigravity(AntigravityConfig{BaseURL: server.URL, Client: server.Client(), ProjectResolved: resolved})
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func fixtureAntigravityCredential(project string) *core.Credential {
	credential := &core.Credential{ConnectionID: "fixture-connection", Token: "fixture-access", TokenType: "Bearer"}
	if project != "" {
		credential.Metadata = map[string]string{core.CredentialMetadataProjectID: project}
	}
	return credential
}

func antigravityChatRequest(body string, credential *core.Credential) core.Request {
	return core.Request{
		Surface: core.ModelSurfaceChatCompletions, Model: "gemini-fixture", Body: []byte(body),
		ContentType: core.ContentTypeJSON, Credential: credential,
	}
}

// gatewayAntigravityPayload is what the gateway's Chat facade hands the
// legacy adapter for a Chat body: it decodes the body without UseNumber and
// passes the messages and, when set, max_tokens, temperature and tools.
func gatewayAntigravityPayload(t *testing.T, body string) map[string]any {
	t.Helper()
	var request struct {
		Messages    []map[string]any `json:"messages"`
		MaxTokens   any              `json:"max_tokens"`
		Temperature any              `json:"temperature"`
		Tools       any              `json:"tools"`
	}
	if err := json.Unmarshal([]byte(body), &request); err != nil {
		t.Fatal(err)
	}
	messages := make([]any, len(request.Messages))
	for index, message := range request.Messages {
		messages[index] = message
	}
	payload := map[string]any{"messages": messages}
	for key, value := range map[string]any{"max_tokens": request.MaxTokens, "temperature": request.Temperature, "tools": request.Tools} {
		if value != nil {
			payload[key] = value
		}
	}
	return payload
}

// withoutRequestID replaces the request ID each generation draws at random.
func withoutRequestID(call antigravityCall) antigravityCall {
	call.body = regexp.MustCompile(`"requestId":"agent_[0-9a-f]{24}"`).ReplaceAllString(call.body, `"requestId":"<request>"`)
	return call
}

// A Chat request must reach Cloud Code Assist exactly as the gateway sends
// it through the legacy adapter, headers included. The number in the tool
// schema pins float64 decoding, and the extra fields the gateway's policy.
func TestAntigravitySendsWhatTheGatewaySends(t *testing.T) {
	t.Parallel()
	const chat = `{"model":"gemini-fixture","stream":false,"max_tokens":64,"temperature":0.2,"top_p":0.9,"stop":["END"],` +
		`"max_completion_tokens":128,"tool_choice":"auto","response_format":{"type":"json_object"},"user":null,"messages":[` +
		`{"role":"system","content":"Be brief."},{"role":"user","content":[{"type":"text","text":"Look it up"}]},` +
		`{"role":"assistant","content":null,"reasoning_content":"thinking","tool_calls":[{"id":"call-fixture","type":"function",` +
		`"function":{"name":"lookup","arguments":"{\"q\":\"fixture\"}","thought_signature":"fixture-signature"}}]},` +
		`{"role":"tool","tool_call_id":"call-fixture","content":"found"}],"tools":[{"type":"function","function":{"name":"lookup",` +
		`"description":"Look up","parameters":{"type":"object","properties":{"q":{"type":"string","maxLength":1.50}}}}}]}`
	server, calls := recordingAntigravityServer(t, antigravityUpstream(t))
	legacy := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return "fixture-access", "fixture-project", nil
	}, server.Client(), server.URL)
	legacyResponse, err := legacy.Complete(context.Background(), "gemini-fixture", gatewayAntigravityPayload(t, chat), nil)
	if err != nil {
		t.Fatal(err)
	}
	response, err := newFixtureAntigravity(t, server, nil).Invoke(context.Background(), antigravityChatRequest(chat, fixtureAntigravityCredential("fixture-project")))
	if err != nil {
		t.Fatal(err)
	}
	sent := calls()
	if len(sent) != 2 || sent[0].path != "/v1internal:streamGenerateContent?alt=sse" {
		t.Fatalf("upstream = %+v", sent)
	}
	if fromLegacy, fromAntigravity := withoutRequestID(sent[0]), withoutRequestID(sent[1]); !reflect.DeepEqual(fromAntigravity, fromLegacy) {
		t.Fatalf("Antigravity request drifted from the gateway's:\n got %+v\nwant %+v", fromAntigravity, fromLegacy)
	}
	var body map[string]any
	if json.Unmarshal(response.Body, &body) != nil || response.ContentType != core.ContentTypeJSON {
		t.Fatalf("response = %s %s", response.ContentType, response.Body)
	}
	for _, key := range []string{"id", "created"} {
		delete(body, key)
		delete(legacyResponse, key)
	}
	if want, _ := json.Marshal(legacyResponse); !reflect.DeepEqual(body, decodeJSONObject(t, want)) {
		t.Fatalf("response = %v, want the legacy %s", body, want)
	}
	var dropped []string
	for _, loss := range response.Losses {
		if loss.Class != translate.LossDropped {
			t.Fatalf("loss = %+v", loss)
		}
		dropped = append(dropped, loss.Path+":"+string(loss.Severity))
	}
	if want := []string{"max_completion_tokens:advisory", "response_format:material", "stop:advisory", "tool_choice:advisory", "top_p:advisory"}; !slices.Equal(dropped, want) {
		t.Fatalf("dropped = %v, want %v", dropped, want)
	}
}

func decodeJSONObject(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	return value
}
