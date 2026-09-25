package providers

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// Ported from the gateway's googleai_serviceaccount_test.go: a token is a
// Bearer credential and never an x-goog-api-key, an API key keeps its
// header, and stray whitespace never reaches a header.
func TestGoogleAuthenticatesWithTheRequestCredential(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name                  string
		deployment            GoogleDeployment
		credential            *core.Credential
		authorization, apiKey string
	}{
		{name: "access token", deployment: GoogleVertexAI, credential: googleBearer("ya29.minted-token"), authorization: "Bearer ya29.minted-token"},
		{name: "api key", deployment: GoogleVertexAI, credential: googleKey("an-api-key"), apiKey: "an-api-key"},
		{name: "trimmed token", deployment: GoogleVertexAI, credential: googleBearer("  ya29.padded  "), authorization: "Bearer ya29.padded"},
		// Vertex AI refuses a request that presents both.
		{name: "both", deployment: GoogleVertexAI, credential: &core.Credential{APIKey: "k", Token: "ya29.both"}, authorization: "Bearer ya29.both"},
		{name: "key record in token", deployment: GoogleAIStudio, credential: &core.Credential{Token: " studio-key ", TokenType: core.TokenTypeAPIKey}, apiKey: "studio-key"},
		{name: "untyped key", deployment: GoogleAIStudio, credential: &core.Credential{APIKey: "studio-key"}, apiKey: "studio-key"},
		// AI Studio sends what it has, as the gateway does, for a proxy
		// that holds the key.
		{name: "no credential", deployment: GoogleAIStudio},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
			if testCase.deployment == GoogleVertexAI {
				base += "/v1"
			}
			provider := newGoogleTest(t, GoogleConfig{Deployment: testCase.deployment, BaseURL: base, Project: "proj"})
			if _, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, testCase.credential)); err != nil {
				t.Fatal(err)
			}
			calls := fake.take()
			if len(calls) != 1 || calls[0].authorization != testCase.authorization || calls[0].apiKey != testCase.apiKey || calls[0].query != "" {
				t.Fatalf("upstream = %+v", calls)
			}
		})
	}
}

// Ported from the gateway's TestVertexWithoutAnyCredentialFailsClearly:
// Vertex AI answers a missing credential with 401, which reads as a
// rejected key and sends the operator looking in the wrong place.
func TestGoogleVertexWithoutAnyCredentialFailsClearly(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "fixture-project"})
	for _, credential := range []*core.Credential{nil, {}, {APIKey: " ", Token: " "}} {
		_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, credential))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration ||
			!strings.Contains(err.Error(), "no credential") || !strings.Contains(err.Error(), "vertex_ai") {
			t.Fatalf("error = %v, want a configuration error that names the missing credential", err)
		}
	}
	if calls := fake.take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
}

// A credential's project fills an unset one and must match a set one, as a
// service-account key's project does in the gateway.
func TestGoogleVertexProjectComesFromTheCredential(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	named := func(project string) *core.Credential {
		credential := googleBearer("ya29.token")
		credential.Metadata = map[string]string{core.CredentialMetadataProjectID: project}
		return credential
	}
	unset := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1"})
	if _, err := unset.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, named(" billing-project "))); err != nil {
		t.Fatal(err)
	}
	set := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "Fixture-Project", Location: "us-central1"})
	if _, err := set.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, named("fixture-project"))); err != nil {
		t.Fatal(err)
	}
	calls := fake.take()
	if len(calls) != 2 || calls[0].path != "/v1/projects/billing-project/locations/global/publishers/google/models/gemini-3.5-flash:generateContent" ||
		calls[1].path != "/v1/projects/Fixture-Project/locations/us-central1/publishers/google/models/gemini-3.5-flash:generateContent" {
		t.Fatalf("upstream = %+v", calls)
	}
	for provider, project := range map[*Google]string{set: "another-project", unset: "bad/project"} {
		_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, named(project)))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration || !strings.Contains(err.Error(), "project") {
			t.Fatalf("project %q: %v, want a configuration error", project, err)
		}
	}
	// Without any project, the gateway's message.
	if _, err := unset.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleKey("k"))); err == nil ||
		err.Error() != "vertex_ai: project is required (set 'project' on the provider)" {
		t.Fatalf("error = %v", err)
	}
	if calls := fake.take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
}

// Material of another kind is no bearer token, so it is refused rather than
// sent: AI Studio takes no service-account key, as in the gateway, and
// neither deployment takes an Anthropic setup token.
func TestGoogleRefusesCredentialKindsItDoesNotServe(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	studio := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	vertex := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "p"})
	for provider, kind := range map[*Google]string{studio: core.TokenTypeGCPServiceAccount, vertex: core.TokenTypeAnthropicSetupToken} {
		_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`,
			&core.Credential{Token: `{"type":"service_account"}`, TokenType: kind}))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration || strings.Contains(err.Error(), "service_account\"") {
			t.Fatalf("%s with kind %s: %v, want a configuration error", provider.label(), kind, err)
		}
	}
	if calls := fake.take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
}

// Ported from the gateway's TestVertexRequestTypeHeaderIsExplicitAndInvocationOnly.
func TestGoogleVertexRequestTypeHeaderIsExplicitAndInvocationOnly(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct{ name, mode, want string }{
		{name: "unset"},
		{name: "default", mode: "default"},
		{name: "paygo", mode: " PayGo ", want: "shared"},
		{name: "dedicated", mode: "dedicated", want: "dedicated"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			fake, base := newGoogleFake(t, func(r *http.Request) (int, string) {
				if r.Method == http.MethodGet {
					return http.StatusOK, `{"publisherModels":[]}`
				}
				return http.StatusOK, googleHelloAnswer
			})
			provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "project", RequestType: testCase.mode})
			if _, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleBearer("ya29.t"))); err != nil {
				t.Fatal(err)
			}
			if _, err := provider.ListModels(context.Background(), googleBearer("ya29.t")); err != nil {
				t.Fatal(err)
			}
			// Discovery is no invocation, so it carries no request type.
			calls := fake.take()
			if len(calls) != 2 || calls[0].requestType != testCase.want || calls[1].method != http.MethodGet || calls[1].requestType != "" {
				t.Fatalf("upstream = %+v, want request type %q on the invocation only", calls, testCase.want)
			}
		})
	}
	// AI Studio has no request type, whatever the configuration says.
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base, RequestType: "dedicated"})
	if _, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleKey("k"))); err != nil {
		t.Fatal(err)
	}
	if calls := fake.take(); len(calls) != 1 || calls[0].requestType != "" {
		t.Fatalf("AI Studio received a Vertex request type: %+v", calls)
	}
}

func TestNewGoogleValidatesItsConfiguration(t *testing.T) {
	t.Parallel()
	for name, config := range map[string]GoogleConfig{
		"no deployment":        {},
		"unknown deployment":   {Deployment: "gemini"},
		"relative base":        {Deployment: GoogleAIStudio, BaseURL: "generativelanguage.googleapis.com/v1beta"},
		"unknown request type": {Deployment: GoogleVertexAI, RequestType: "priority"},
		"location with a path": {Deployment: GoogleVertexAI, Location: "us-central1/../eu"},
		"project with a query": {Deployment: GoogleVertexAI, Project: "p?alt=sse"},
	} {
		if _, err := NewGoogle(config); err == nil {
			t.Errorf("%s: NewGoogle accepted %+v", name, config)
		}
	}
}
