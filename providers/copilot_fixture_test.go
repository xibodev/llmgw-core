package providers

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	copilotauth "github.com/xibodev/llm-provider-auth/copilot"
	core "github.com/xibodev/llmgw-core"
)

const copilotExchangePath = "/copilot_internal/v2/token"

// copilotRecorded is one request to the synthetic Copilot API.
type copilotRecorded struct {
	method, path, authorization, accept, vision, body string
	header                                            http.Header
}

// copilotBackend is a synthetic GitHub session-token exchange and the
// Copilot API at the base URL its sessions name. It issues session-1,
// session-2 and so on, refuses the OAuth tokens and sessions a test rejects,
// and answers accepted API calls with answer. With hold set, an exchange
// sends on hold when it arrives and answers once it receives from it.
type copilotBackend struct {
	server *httptest.Server

	mu            sync.Mutex
	exchanges     []string
	calls         []copilotRecorded
	issued        int
	expiresAt     int64
	exchangeFails map[string]int
	rejected      map[string]bool
	rejectAll     bool
	hold          chan struct{}
	answer        func(w http.ResponseWriter, r *http.Request, body []byte)
}

func newCopilotBackend(t *testing.T, answer func(w http.ResponseWriter, r *http.Request, body []byte)) *copilotBackend {
	t.Helper()
	backend := &copilotBackend{exchangeFails: map[string]int{}, rejected: map[string]bool{}, answer: answer}
	backend.server = httptest.NewServer(backend)
	t.Cleanup(backend.server.Close)
	return backend
}

func (b *copilotBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	b.mu.Lock()
	if r.URL.Path == copilotExchangePath {
		oauth := strings.TrimPrefix(r.Header.Get("Authorization"), "token ")
		b.exchanges = append(b.exchanges, oauth)
		b.issued++
		token, status, expiresAt, hold := fmt.Sprintf("session-%d", b.issued), b.exchangeFails[oauth], b.expiresAt, b.hold
		b.mu.Unlock()
		if hold != nil {
			hold <- struct{}{}
			<-hold
		}
		if status != 0 {
			w.WriteHeader(status)
			return
		}
		if expiresAt == 0 {
			expiresAt = time.Now().Add(30 * time.Minute).Unix()
		}
		writeCopilotJSON(w, map[string]any{"token": token, "expires_at": expiresAt, "endpoints": map[string]any{"api": b.server.URL + "/api/"}})
		return
	}
	authorization := r.Header.Get("Authorization")
	b.calls = append(b.calls, copilotRecorded{
		method: r.Method, path: r.URL.Path, authorization: authorization, accept: r.Header.Get("Accept"),
		vision: r.Header.Get("Copilot-Vision-Request"), body: string(body), header: r.Header.Clone(),
	})
	rejected, answer := b.rejectAll || b.rejected[strings.TrimPrefix(authorization, "Bearer ")], b.answer
	b.mu.Unlock()
	if rejected {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	answer(w, r, body)
}

// take returns the exchanges and API calls since the last take.
func (b *copilotBackend) take() ([]string, []copilotRecorded) {
	b.mu.Lock()
	defer b.mu.Unlock()
	exchanges, calls := b.exchanges, b.calls
	b.exchanges, b.calls = nil, nil
	return exchanges, calls
}

func (b *copilotBackend) update(change func(*copilotBackend)) {
	b.mu.Lock()
	defer b.mu.Unlock()
	change(b)
}

func writeCopilotJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

// newFixtureCopilot returns a Copilot on backend. Auth's own token is
// product-oauth, and adjust may change either configuration.
func newFixtureCopilot(t *testing.T, backend *copilotBackend, adjust ...func(*CopilotConfig, *copilotauth.Config)) *Copilot {
	t.Helper()
	auth := copilotauth.Config{
		AllowProxy: true, OAuthToken: "product-oauth", HTTPClient: backend.server.Client(),
		Endpoints: copilotauth.Endpoints{SessionTokenURL: backend.server.URL + copilotExchangePath},
	}
	config := CopilotConfig{EditorPluginVersion: "fixture-plugin/1.0", UserAgent: "FixtureCopilotChat/1.0", Client: backend.server.Client()}
	for _, change := range adjust {
		change(&config, &auth)
	}
	config.Auth = copilotauth.New(auth)
	provider, err := NewCopilot(config)
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func copilotCredential(token string) *core.Credential {
	return &core.Credential{Token: token, TokenType: "Bearer"}
}

func copilotRequest(surface core.ModelSurface, model, body string, credential *core.Credential) core.Request {
	return core.Request{Surface: surface, Model: model, Body: []byte(body), ContentType: core.ContentTypeJSON, Credential: credential}
}

// assertCopilotFailure checks the canonical error a Copilot operation returns.
func assertCopilotFailure(t *testing.T, err error, class core.ProviderErrorClass, want core.ProviderErrorClassification) *core.ProviderError {
	t.Helper()
	var failure *core.ProviderError
	if !errors.As(err, &failure) {
		t.Fatalf("error = %T %v, want a *core.ProviderError", err, err)
	}
	if failure.Class != class || failure.Classification != want || core.ClassifyError(err) != want {
		t.Fatalf("error %q: class %q classification %+v, want %q %+v", err, failure.Class, failure.Classification, class, want)
	}
	return failure
}
