package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

func TestZenFailuresCarryTheirStatusAndRetryAfter(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name       string
		request    core.Request
		status     int
		retryAfter string
		stream     bool
		class      core.ProviderErrorClass
		retryable  bool
		message    string
	}{
		{"anonymous 401", chatRequest("paid-fixture", `{"messages":[{"role":"user","content":"hi"}]}`, nil), 401, "", false, core.ProviderErrorAuth, false,
			`OpenCode Zen model "paid-fixture" is not available through the current anonymous catalog; configure an OpenCode Zen API key for paid models.`},
		{"keyed 401", chatRequest("paid-fixture", `{"messages":[]}`, &core.Credential{APIKey: "fixture-key"}), 401, "", true, core.ProviderErrorAuth, false, "OpenCode Zen returned HTTP 401 (code=invalid_key)"},
		{"429", chatRequest("big-pickle", `{"messages":[]}`, &core.Credential{APIKey: "fixture-key"}), 429, "7", false, core.ProviderErrorRateLimited, true, "OpenCode Zen returned HTTP 429 (code=invalid_key)"},
		{"503 stream", responsesRequest("gpt-fixture", `{"input":"hi"}`, nil), 503, "2", true, core.ProviderErrorUpstream, true, "OpenCode Zen returned HTTP 503 (code=invalid_key)"},
	} {
		t.Run(test.name, func(t *testing.T) {
			backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, _ int) {
				if test.retryAfter != "" {
					w.Header().Set("Retry-After", test.retryAfter)
				}
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, `{"error":{"code":"invalid_key","message":"private upstream text"}}`)
			}}
			server := httptest.NewServer(backend)
			defer server.Close()
			provider := newTestZen(t, server, responsesRow)
			var err error
			if test.stream {
				_, err = provider.Stream(context.Background(), test.request)
			} else {
				_, err = provider.Invoke(context.Background(), test.request)
			}
			var failure *core.ProviderError
			if !errors.As(err, &failure) || failure.Class != test.class || failure.Message != test.message {
				t.Fatalf("err = %#v", err)
			}
			wantRetry := time.Duration(0)
			if test.retryAfter != "" {
				wantRetry = map[string]time.Duration{"7": 7 * time.Second, "2": 2 * time.Second}[test.retryAfter]
			}
			if c := failure.Classification; c.StatusCode != test.status || c.Retryable != test.retryable || c.RetryAfter != wantRetry {
				t.Fatalf("classification = %+v", c)
			}
		})
	}
}

func TestZenRefusesBeforeSending(t *testing.T) {
	t.Parallel()
	backend := &zenBackend{reply: func(w http.ResponseWriter, _ *http.Request, _ int) { t.Error("a refused request reached Zen") }}
	server := httptest.NewServer(backend)
	defer server.Close()
	rows := map[string]core.ModelInfo{
		"paid-fixture": {ID: "paid-fixture", SupportedAPIs: []string{"/chat/completions"}},
		"chat-fixture": {ID: "chat-fixture", Tags: []string{ModelTagFree}},
	}
	provider := newTestZen(t, server, func(model string) (core.ModelInfo, bool) { row, ok := rows[model]; return row, ok })
	var failure *core.ProviderError
	_, err := provider.Invoke(context.Background(), chatRequest("paid-fixture", `{"messages":[]}`, &core.Credential{APIKey: "public"}))
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorAuth || failure.Classification.StatusCode != 0 || !failure.Classification.FailoverEligible {
		t.Fatalf("anonymous paid model: err = %#v", err)
	}
	var surface *core.SurfaceError
	if _, err := provider.Stream(context.Background(), responsesRequest("chat-fixture", `{"input":"hi"}`, nil)); !errors.As(err, &surface) {
		t.Fatalf("Responses on a Chat model: err = %v", err)
	}
	messages := core.Request{Surface: core.ModelSurfaceMessages, Model: "chat-fixture", Body: []byte(`{}`), ContentType: core.ContentTypeJSON}
	if _, err := provider.Invoke(context.Background(), messages); !errors.As(err, &surface) {
		t.Fatalf("Messages: err = %v", err)
	}
	for body, contentType := range map[string]string{`{"messages":[]}`: "text/plain", `{"messages":[]} {}`: core.ContentTypeJSON, `{"messages":"hi"}`: core.ContentTypeJSON} {
		request := chatRequest("chat-fixture", body, nil)
		request.ContentType = contentType
		if _, err := provider.Invoke(context.Background(), request); !errors.As(err, &failure) || failure.Class != core.ProviderErrorInvalidRequest {
			t.Fatalf("%s as %s: err = %v", body, contentType, err)
		}
	}
}

func TestZenReportsTransportFailuresAndCallerCancellation(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(&zenBackend{reply: func(http.ResponseWriter, *http.Request, int) {}})
	provider := newTestZen(t, server, nil)
	server.Close()
	var failure *core.ProviderError
	_, err := provider.Invoke(context.Background(), chatRequest("big-pickle", `{"messages":[]}`, nil))
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorTransport || !failure.Classification.Retryable {
		t.Fatalf("unreachable: err = %#v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = provider.Invoke(ctx, chatRequest("big-pickle", `{"messages":[]}`, nil))
	if !errors.As(err, &failure) || failure.Classification != (core.ProviderErrorClassification{}) || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled: err = %#v", err)
	}
}

func TestZenNativeSurfacesFollowTheCatalogRow(t *testing.T) {
	t.Parallel()
	rows := map[string][]string{"responses": {"/responses"}, "both": {"/chat/completions", "/v1/responses"}, "chat": {"/v1/chat/completions"}, "unlisted": nil}
	provider, err := NewZen(ZenConfig{Models: func(model string) (core.ModelInfo, bool) {
		endpoints, ok := rows[model]
		return core.ModelInfo{ID: model, SupportedAPIs: endpoints}, ok
	}})
	if err != nil {
		t.Fatal(err)
	}
	chat, responses := core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses
	for model, want := range map[string][]core.ModelSurface{
		"responses": {responses, chat}, "both": {chat, responses}, "chat": {chat}, "unlisted": {chat},
		"muse-spark-cold": {responses, chat}, "Muse-Spark-Cold": {responses, chat}, "cold": {chat},
	} {
		if got := provider.NativeSurfaces(model); !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: surfaces = %v, want %v", model, got, want)
		}
	}
	if _, err := NewZen(ZenConfig{BaseURL: "zen.example/v1"}); err == nil {
		t.Fatal("a relative base URL was accepted")
	}
}
