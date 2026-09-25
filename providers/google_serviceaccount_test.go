package providers

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	gcp "github.com/xibodev/llm-provider-auth/gcp"
	core "github.com/xibodev/llmgw-core"
)

// googleTestRSAKey is a throwaway key, generated once because generation
// is slow. No test needs a real key.
var googleTestRSAKey = sync.OnceValue(func() *rsa.PrivateKey {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		panic(err)
	}
	return key
})

// googleServiceAccountKey is a synthetic service-account key whose token
// endpoint is tokenURI.
func googleServiceAccountKey(t *testing.T, tokenURI, project string) string {
	t.Helper()
	der, err := x509.MarshalPKCS8PrivateKey(googleTestRSAKey())
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(map[string]string{
		"type": "service_account", "project_id": project, "private_key_id": "key-1",
		"private_key":  string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"client_email": "svc@" + project + ".iam.example.test", "token_uri": tokenURI,
	})
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func googleServiceAccount(key string) *core.Credential {
	return &core.Credential{Token: key, TokenType: core.TokenTypeGCPServiceAccount}
}

// googleTokenEndpoint is a synthetic Google token endpoint that issues
// token-1, token-2 and so on, one per exchange.
func googleTokenEndpoint(t *testing.T) (*googleFake, string, *atomic.Int32) {
	t.Helper()
	exchanges := &atomic.Int32{}
	fake, base := newGoogleFake(t, func(*http.Request) (int, string) {
		return http.StatusOK, fmt.Sprintf(`{"access_token":"token-%d","expires_in":3600,"token_type":"Bearer"}`, exchanges.Add(1))
	})
	return fake, base + "/token", exchanges
}

// googleAssertionClaims decodes the claims of the JWT a token exchange
// presented.
func googleAssertionClaims(t *testing.T, form string) (string, map[string]any) {
	t.Helper()
	values, err := url.ParseQuery(form)
	if err != nil {
		t.Fatal(err)
	}
	parts := strings.Split(values.Get("assertion"), ".")
	if len(parts) != 3 {
		t.Fatalf("assertion = %q", values.Get("assertion"))
	}
	raw, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		t.Fatal(err)
	}
	var claims map[string]any
	if err := json.Unmarshal(raw, &claims); err != nil {
		t.Fatal(err)
	}
	return values.Get("grant_type"), claims
}

type googleClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *googleClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *googleClock) Advance(by time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(by)
}

// Ported from the gateway's TestVertexUsesStoredServiceAccountConnection
// and TestServiceAccountProviderSendsBearerToken: a service-account key
// authenticates as a bearer it mints, never as x-goog-api-key, and the
// project comes from the key when none is configured.
func TestGoogleVertexServiceAccountSendsAMintedBearer(t *testing.T) {
	t.Parallel()
	tokens, tokenURI, _ := googleTokenEndpoint(t)
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1"})
	key := googleServiceAccountKey(t, tokenURI, "fixture-project")
	if _, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleServiceAccount(key))); err != nil {
		t.Fatal(err)
	}
	calls := fake.take()
	if len(calls) != 1 || calls[0].authorization != "Bearer token-1" || calls[0].apiKey != "" ||
		calls[0].path != "/v1/projects/fixture-project/locations/global/publishers/google/models/gemini-3.5-flash:generateContent" {
		t.Fatalf("upstream = %+v", calls)
	}
	exchanges := tokens.take()
	if len(exchanges) != 1 || exchanges[0].path != "/token" || exchanges[0].authorization != "" {
		t.Fatalf("token exchanges = %+v", exchanges)
	}
	grant, claims := googleAssertionClaims(t, exchanges[0].body)
	if grant != "urn:ietf:params:oauth:grant-type:jwt-bearer" || claims["scope"] != gcp.CloudPlatformScope ||
		claims["iss"] != "svc@fixture-project.iam.example.test" || claims["aud"] != tokenURI {
		t.Fatalf("grant = %q, claims = %+v", grant, claims)
	}
}

// Ported from the gateway's TestVertexCachedProviderRefreshesServiceAccountToken:
// a token serves every operation until shortly before it expires, and the
// next operation mints another. Each instance keeps its own tokens.
func TestGoogleVertexServiceAccountTokensAreCachedPerInstance(t *testing.T) {
	t.Parallel()
	_, tokenURI, exchanges := googleTokenEndpoint(t)
	fake, base := newGoogleFake(t, func(r *http.Request) (int, string) {
		return http.StatusOK, `{"candidates":[{"content":{"parts":[{"text":"` + strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ") + `"}]}}]}`
	})
	clock := &googleClock{now: time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC)}
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Now: clock.Now})
	credential := googleServiceAccount(googleServiceAccountKey(t, tokenURI, "fixture-project"))
	answer := func(provider *Google) string {
		t.Helper()
		response, err := provider.Invoke(context.Background(), googleChatRequest("gemini-test", `{"messages":[{"role":"user","content":"hi"}]}`, credential))
		if err != nil {
			t.Fatal(err)
		}
		var body struct {
			Choices []struct{ Message struct{ Content string } } `json:"choices"`
		}
		if err := json.Unmarshal(response.Body, &body); err != nil || len(body.Choices) != 1 {
			t.Fatalf("response = %s", response.Body)
		}
		return body.Choices[0].Message.Content
	}
	for index, want := range []string{"token-1", "token-1", "token-2"} {
		if index == 2 {
			// A minute before expiry the cache mints again.
			clock.Advance(59*time.Minute + time.Second)
		}
		if got := answer(provider); got != want {
			t.Fatalf("request %d token=%q want=%q", index+1, got, want)
		}
	}
	if other := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Now: clock.Now}); answer(other) != "token-3" {
		t.Fatal("a second instance shared the first one's token")
	}
	if exchanges.Load() != 3 || len(fake.take()) != 4 {
		t.Fatalf("token exchanges=%d want=3", exchanges.Load())
	}
}

// The key names its project, so an unset project is filled in from it and
// a contradicting one fails before anything is sent, as in the gateway. A
// project the credential names bills calls to it instead of the key's own.
func TestGoogleVertexServiceAccountProject(t *testing.T) {
	t.Parallel()
	tokens, tokenURI, _ := googleTokenEndpoint(t)
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	key := googleServiceAccountKey(t, tokenURI, "fixture-project")
	mismatched := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "other-project"})
	_, err := mismatched.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleServiceAccount(key)))
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration ||
		err.Error() != `vertex_ai: configured project "other-project" does not match the service account project "fixture-project"` {
		t.Fatalf("error = %v", err)
	}
	if len(fake.take()) != 0 || len(tokens.take()) != 0 {
		t.Fatal("a request was sent for a project the key contradicts")
	}
	matching := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "FIXTURE-PROJECT"})
	billed := googleServiceAccount(key)
	billed.Metadata = map[string]string{core.CredentialMetadataProjectID: "billing-project"}
	unset := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1"})
	for provider, credential := range map[*Google]*core.Credential{matching: googleServiceAccount(key), unset: billed} {
		if _, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, credential)); err != nil {
			t.Fatal(err)
		}
	}
	paths := map[string]bool{}
	for _, call := range fake.take() {
		paths[strings.Split(call.path, "/locations/")[0]] = true
	}
	if !paths["/v1/projects/FIXTURE-PROJECT"] || !paths["/v1/projects/billing-project"] || len(paths) != 2 {
		t.Fatalf("projects = %v", paths)
	}
}

// A key that is not a service-account key fails before anything is sent,
// and the failure describes its shape, never its material.
func TestGoogleServiceAccountKeyFailures(t *testing.T) {
	t.Parallel()
	tokens, tokenURI, _ := googleTokenEndpoint(t)
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: "https://vertex.example.test/v1"})
	var document map[string]string
	if err := json.Unmarshal([]byte(googleServiceAccountKey(t, tokenURI, "fixture-project")), &document); err != nil {
		t.Fatal(err)
	}
	document["project_id"] = ""
	missing, _ := json.Marshal(document)
	for name, key := range map[string]string{
		"not json":        "not-json",
		"authorized user": `{"type":"authorized_user","client_secret":"fixture-private-secret"}`,
		"missing project": string(missing),
	} {
		_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleServiceAccount(key)))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration || !strings.HasPrefix(err.Error(), "vertex_ai: service account key: ") ||
			strings.Contains(err.Error(), "PRIVATE KEY") || strings.Contains(err.Error(), "fixture-private-secret") {
			t.Errorf("%s: %v", name, err)
		}
	}
	if len(tokens.take()) != 0 {
		t.Fatal("a malformed key was exchanged")
	}
}

// Ported from the gateway's token source in newVertexProvider: a refusal
// from the token endpoint permits failover unless it rejects the account,
// a transient one repeats, and an endpoint out of reach may repeat. A
// catalog maps them to the gateway's catalog codes.
func TestGoogleServiceAccountTokenFailures(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name, answer  string
		status        int
		want          core.ProviderErrorClassification
		catalogCode   string
		catalogStatus int
	}{
		{
			name: "invalid grant", status: 400, answer: `{"error":"invalid_grant","error_description":"fixture-private-description"}`,
			want:        core.ProviderErrorClassification{StatusCode: 400, FailoverEligible: true},
			catalogCode: CatalogCodeAuthenticationFailed, catalogStatus: 400,
		},
		{
			name: "account rejected", status: 401, answer: `{"error":"invalid_client"}`,
			want:        core.ProviderErrorClassification{StatusCode: 401},
			catalogCode: CatalogCodeAuthenticationFailed, catalogStatus: 401,
		},
		{
			name: "unavailable", status: 503, answer: `{}`,
			want:        core.ProviderErrorClassification{StatusCode: 503, Retryable: true, FailoverEligible: true, CircuitFailure: true},
			catalogCode: CatalogCodeTransportError, catalogStatus: 503,
		},
		{
			name: "no token", status: 200, answer: `{"token_type":"Bearer"}`,
			want:        core.ProviderErrorClassification{FailoverEligible: true},
			catalogCode: CatalogCodeAuthenticationFailed,
		},
		{
			name: "unreachable", want: core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true},
			catalogCode: CatalogCodeTransportError,
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			tokens, base := newGoogleFake(t, googleAnswer(testCase.status, testCase.answer))
			tokenURI := base + "/token"
			if testCase.status == 0 {
				tokenURI = googleHangUp(t)
			}
			fake, vertex := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
			provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: vertex + "/v1"})
			credential := googleServiceAccount(googleServiceAccountKey(t, tokenURI, "fixture-project"))
			_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, credential))
			var exchange *gcp.TokenError
			if classification := core.ClassifyError(err); classification != testCase.want || !errors.As(err, &exchange) ||
				!strings.HasPrefix(err.Error(), "vertex_ai: service account token refresh failed") || strings.Contains(err.Error(), "fixture-private") {
				t.Fatalf("error = %v, classification = %+v, want %+v", err, classification, testCase.want)
			}
			_, err = provider.ListModels(context.Background(), credential)
			if code, detail, status := googleCatalogCode(t, err); code != testCase.catalogCode || status != testCase.catalogStatus ||
				!strings.HasPrefix(detail, "Provider credential refresh") {
				t.Fatalf("catalog failure=(%q,%q,%d)", code, detail, status)
			}
			if calls := fake.take(); len(calls) != 0 {
				t.Fatalf("upstream = %+v", calls)
			}
			if testCase.status != 0 && len(tokens.take()) != 2 {
				t.Fatal("each operation did not try to exchange the key")
			}
		})
	}
}

// googleHangUp returns an endpoint that closes every connection unanswered.
func googleHangUp(t *testing.T) string {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		connection, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			_ = connection.Close()
		}
	}))
	t.Cleanup(server.Close)
	return server.URL + "/token"
}
