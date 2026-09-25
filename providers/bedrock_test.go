package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// The gateway templates Bedrock's OpenAI-compatible endpoint from the
// region, us-east-1 by default, unless a base URL is configured.
func TestNewBedrockTemplatesTheRegionEndpoint(t *testing.T) {
	t.Parallel()
	for _, test := range []struct{ region, baseURL, want string }{
		{"", "", "https://bedrock-runtime.us-east-1.amazonaws.com/v1"},
		{" eu-west-3 ", "", "https://bedrock-runtime.eu-west-3.amazonaws.com/v1"},
		{"US-GOV-WEST-1", "", "https://bedrock-runtime.us-gov-west-1.amazonaws.com/v1"},
		{"eu-west-3", " https://bedrock-mantle.example.test/v1/ ", "https://bedrock-mantle.example.test/v1"},
	} {
		provider, err := NewBedrock(test.region, test.baseURL, OpenAICompatibleConfig{BaseURL: "https://ignored.example.test"})
		if err != nil || provider.baseURL != test.want || provider.label != "Bedrock" {
			t.Fatalf("%q %q: provider = %+v, err = %v", test.region, test.baseURL, provider, err)
		}
	}
	for _, region := range []string{"us-east-1.evil.example", "x@evil.example/", "us--east-1", "-us-east-1", "us-east-1-", "us_east_1", "eu-west-3#"} {
		var failure *core.ProviderError
		if _, err := NewBedrock(region, "", OpenAICompatibleConfig{}); !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration {
			t.Fatalf("region %q: err = %v", region, err)
		}
	}
	if _, err := NewBedrock("", "bedrock.example.test/v1", OpenAICompatibleConfig{}); err == nil {
		t.Fatal("a relative base URL was accepted")
	}
}

// A Bedrock API key is a bearer token, and Bedrock is otherwise an
// OpenAI-compatible upstream.
func TestBedrockAuthenticatesWithABearerKey(t *testing.T) {
	t.Parallel()
	backend, server := newOpenAIBackend(t, func(w http.ResponseWriter, r *http.Request, attempt int) {
		if attempt > 1 {
			w.WriteHeader(http.StatusForbidden)
			_, _ = io.WriteString(w, `{"message":"fixture-secret"}`)
			return
		}
		answerOpenAIChat(w, r, attempt)
	})
	provider, err := NewBedrock("us-west-2", server.URL+"/openai/v1", OpenAICompatibleConfig{Client: server.Client(), ForceAPISupport: true})
	if err != nil {
		t.Fatal(err)
	}
	request := openAIRequest(core.ModelSurfaceChatCompletions, "fixture.model-v1", `{"messages":[{"role":"user","content":"hi"}]}`, &core.Credential{APIKey: "fixture-bedrock-key"})
	if _, err := provider.Invoke(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	_, err = provider.Invoke(context.Background(), request)
	if core.ClassifyError(err).StatusCode != http.StatusForbidden || !strings.HasPrefix(err.Error(), "Bedrock returned HTTP 403") ||
		strings.Contains(err.Error(), "fixture-secret") {
		t.Fatalf("err = %v", err)
	}
	calls := backend.take()
	if len(calls) != 2 || calls[0].path != "/openai/v1/chat/completions" || calls[0].authorization != "Bearer fixture-bedrock-key" ||
		calls[0].body != `{"messages":[{"content":"hi","role":"user"}],"model":"fixture.model-v1","stream":false}` {
		t.Fatalf("upstream = %+v", calls)
	}
}
