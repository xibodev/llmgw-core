package providers

import (
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
)

// ollamaRecorded is one request the fixture daemon received.
type ollamaRecorded struct{ method, path, contentType, body string }

// ollamaDaemon is a synthetic Ollama daemon that records each request and
// answers every one with status and reply.
func ollamaDaemon(t *testing.T, status int, reply string) (*Ollama, func() []ollamaRecorded) {
	t.Helper()
	var mu sync.Mutex
	var calls []ollamaRecorded
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		calls = append(calls, ollamaRecorded{r.Method, r.URL.Path, r.Header.Get("Content-Type"), string(body)})
		mu.Unlock()
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(server.Close)
	provider, err := NewOllama(OllamaConfig{BaseURL: server.URL + "/", Client: server.Client(), CatalogClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	return provider, func() []ollamaRecorded {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(calls)
	}
}

func ollamaChatRequest(model, body string) core.Request {
	return core.Request{Surface: core.ModelSurfaceChatCompletions, Model: model, Body: []byte(body), ContentType: core.ContentTypeJSON}
}

// Ported from the gateway's TestOllamaNativeRootDiagnostics, with the base
// URL a provider refuses to be built with.
func TestOllamaBaseURLIssueIsTheGatewaysDiagnostic(t *testing.T) {
	t.Parallel()
	for _, scenario := range []struct{ base, wantIssue string }{
		{base: "http://127.0.0.1:11434"},
		{base: "http://host.docker.internal:11434/"},
		{base: " https://ollama.example.test "},
		{base: "http://127.0.0.1:11434/v1", wantIssue: "not the OpenAI-compatible /v1 URL; remove /v1."},
		{base: "http://127.0.0.1:11434/V1/", wantIssue: "remove /v1."},
		{base: "http://127.0.0.1:11434/api", wantIssue: "must be the native daemon root with no path."},
		{base: "127.0.0.1:11434", wantIssue: "must be an http(s) native daemon root such as http://127.0.0.1:11434."},
		{base: "://fixture-secret", wantIssue: "must be an http(s)"},
		{base: "ftp://127.0.0.1:11434", wantIssue: "must be an http(s)"},
	} {
		issue := OllamaBaseURLIssue(scenario.base)
		if scenario.wantIssue == "" && issue != "" || scenario.wantIssue != "" && !strings.Contains(issue, scenario.wantIssue) {
			t.Fatalf("OllamaBaseURLIssue(%q) = %q, want %q", scenario.base, issue, scenario.wantIssue)
		}
		provider, err := NewOllama(OllamaConfig{BaseURL: scenario.base})
		var failure *core.ProviderError
		switch {
		case scenario.wantIssue == "" && (err != nil || provider == nil):
			t.Fatalf("NewOllama(%q) = %v", scenario.base, err)
		case scenario.wantIssue != "" && (!errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration || failure.Message != issue):
			t.Fatalf("NewOllama(%q) error = %#v, want the issue as a configuration error", scenario.base, err)
		}
	}
	provider, err := NewOllama(OllamaConfig{})
	if err != nil || provider.chatURL != "http://127.0.0.1:11434/api/chat" || provider.tagsURL != "http://127.0.0.1:11434/api/tags" {
		t.Fatalf("default provider = %+v, %v", provider, err)
	}
	if surfaces := provider.NativeSurfaces("any"); !reflect.DeepEqual(surfaces, []core.ModelSurface{core.ModelSurfaceChatCompletions}) {
		t.Fatalf("surfaces = %v", surfaces)
	}
}
