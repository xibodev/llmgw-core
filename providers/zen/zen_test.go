package zen

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

type httpClientFunc func(*http.Request) (*http.Response, error)

func (f httpClientFunc) Do(request *http.Request) (*http.Response, error) { return f(request) }

func TestAnonymousHeaders(t *testing.T) {
	sequence := 0
	client, err := New(Config{NewID: func(prefix string) (string, error) {
		sequence++
		return fmt.Sprintf("%s_%d", prefix, sequence), nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	first, err := client.AnonymousHeaders()
	if err != nil {
		t.Fatal(err)
	}
	second, err := client.AnonymousHeaders()
	if err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]string{
		"Authorization": "Bearer public", "Content-Type": "application/json",
		"User-Agent": AnonymousUserAgent, "x-opencode-project": "global", "x-opencode-client": "cli",
		"x-opencode-session": "ses_1", "x-opencode-request": "msg_2",
	} {
		if got := first.Get(key); got != want {
			t.Errorf("%s=%q, want %q", key, got, want)
		}
	}
	if second.Get("x-opencode-session") != "ses_3" || second.Get("x-opencode-request") != "msg_4" {
		t.Fatalf("request identities were reused: %v", second)
	}
}

func TestAnonymousHeadersIDFailure(t *testing.T) {
	client, err := New(Config{NewID: func(string) (string, error) { return "", fmt.Errorf("entropy") }})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.AnonymousHeaders(); err == nil || !strings.Contains(err.Error(), "invocation identity") {
		t.Fatalf("error=%v", err)
	}
}

func TestInvocationIdentityLifetimesAndCallerPrecedence(t *testing.T) {
	sequence := 0
	newID := func(prefix string) (string, error) { sequence++; return fmt.Sprintf("%s_%d", prefix, sequence), nil }
	headers := http.Header{}
	headers.Set("x-opencode-project", "project caller")
	headers.Set("x-opencode-session", "session caller")
	headers.Set("x-opencode-client", "app caller")
	headers.Set("User-Agent", "caller/1")
	first, err := NewInvocationIdentity(headers, newID)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewInvocationIdentity(headers, newID)
	if err != nil {
		t.Fatal(err)
	}
	if first.Project != "project caller" || first.Session != "session caller" || first.Client != "app caller" || first.UserAgent != AnonymousUserAgent || first.Request != "msg_1" {
		t.Fatalf("first=%+v", first)
	}
	if second.Session != first.Session || second.Request != "msg_2" || sequence != 2 {
		t.Fatalf("second=%+v sequence=%d", second, sequence)
	}
	ctx := WithInvocationIdentity(context.Background(), first)
	stable, err := EnsureInvocationIdentity(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := InvocationIdentityFromContext(stable)
	if !ok || !reflect.DeepEqual(got, first) {
		t.Fatalf("identity=%+v ok=%v", got, ok)
	}
}

func TestDiscoverMatchesOpenCodePublicSnapshotSemantics(t *testing.T) {
	observedAt := time.Date(2026, time.September, 21, 12, 0, 0, 0, time.UTC)
	zenCatalog := `{"data":[
		{"id":"chat-free","object":"model","created":1},
		{"id":"responses-free","object":"model","created":2},
		{"id":"beta-free"},{"id":"alpha-free"},{"id":"deprecated-free"},
		{"id":"paid"},{"id":"missing-output"},{"id":"string-zero"},
		{"id":"unknown-cost"},{"id":"unknown-surface"},{"id":"mismatched-id"}
	]}`
	metadata := `{"opencode":{"npm":"@ai-sdk/openai-compatible","models":{
		"chat-free":{"id":"chat-free","name":"Chat Free","cost":{"input":0,"output":0,"cache_read":0},"tool_call":true,"reasoning":true,"structured_output":true,"modalities":{"input":["text","image"]},"limit":{"context":200000,"output":32000}},
		"responses-free":{"id":"responses-free","name":"Responses Free","provider":{"npm":"@ai-sdk/openai"},"cost":{"input":0.0,"output":0e0,"tiers":[{"tier":"small","input":0,"output":0}]},"modalities":{"input":["text"]}},
		"beta-free":{"id":"beta-free","status":"beta","cost":{"input":0,"output":0}},
		"alpha-free":{"id":"alpha-free","status":"alpha","cost":{"input":0,"output":0}},
		"deprecated-free":{"id":"deprecated-free","status":"deprecated","cost":{"input":0,"output":0}},
		"paid":{"id":"paid","cost":{"input":0,"output":0.01}},
		"missing-output":{"id":"missing-output","cost":{"input":0}},
		"missing-input":{"id":"missing-input","cost":{"output":1}},
		"missing-cost":{"id":"missing-cost"},
		"string-zero":{"id":"string-zero","cost":{"input":"0","output":0}},
		"unknown-cost":{"id":"unknown-cost","cost":{"input":0,"output":0,"future_price":0}},
		"unknown-surface":{"id":"unknown-surface","provider":{"npm":"@ai-sdk/anthropic"},"cost":{"input":0,"output":0}},
		"mismatched-id":{"id":"other","cost":{"input":0,"output":0}},
		"metadata-only":{"id":"metadata-only","cost":{"input":0,"output":0}}
	}}}`

	var zenHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/zen/models":
			zenHeaders = r.Header.Clone()
			_, _ = fmt.Fprint(w, zenCatalog)
		case "/metadata":
			if r.Header.Get("Authorization") != "" {
				t.Fatal("models.dev request received Zen authorization")
			}
			_, _ = fmt.Fprint(w, metadata)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	client := testClient(t, server.URL+"/zen", server.URL+"/metadata", observedAt)
	evidence, err := client.Discover(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if evidence.Status != core.CatalogDiscovered || !evidence.ObservedAt.Equal(observedAt) {
		t.Fatalf("evidence=%+v", evidence)
	}
	ids := make([]string, 0, len(evidence.Models))
	for _, model := range evidence.Models {
		ids = append(ids, model.ID)
	}
	if want := []string{"beta-free", "chat-free", "metadata-only", "missing-cost", "missing-input", "missing-output", "paid", "responses-free", "unknown-cost"}; !slices.Equal(ids, want) {
		t.Fatalf("models=%v, want %v", ids, want)
	}
	if zenHeaders != nil {
		t.Fatalf("public snapshot unexpectedly required live Zen: %v", zenHeaders)
	}

	chat := evidence.Models[1]
	if surface, ok := NativeSurface(chat); !ok || surface != core.ModelSurfaceChatCompletions {
		t.Fatalf("chat surface=%q ok=%v", surface, ok)
	}
	capabilities := chat.Capabilities
	if capabilities == nil || capabilities.Tools != core.SupportSupported || capabilities.Reasoning != core.SupportSupported ||
		capabilities.StructuredOutput != core.SupportSupported || capabilities.Inputs.Image != core.SupportSupported ||
		capabilities.Limits.ContextTokens == nil || *capabilities.Limits.ContextTokens != 200000 ||
		capabilities.Provenance.Source != core.ModelCapabilitySourceModelsDev ||
		capabilities.Freshness.ExpiresAt == nil || !capabilities.Freshness.ExpiresAt.Equal(observedAt.Add(time.Hour)) {
		t.Fatalf("chat capabilities=%+v", capabilities)
	}
	responses := evidence.Models[7]
	if surface, ok := NativeSurface(responses); !ok || surface != core.ModelSurfaceResponses {
		t.Fatalf("responses surface=%q ok=%v", surface, ok)
	}
}

func TestDiscoverVerifiedRequiresLiveExactZeroCost(t *testing.T) {
	metadataRequests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/zen/models" {
			_, _ = fmt.Fprint(w, `{"data":[{"id":"free"},{"id":"beta-free"},{"id":"unstated"},{"id":"input-only"}]}`)
			return
		}
		metadataRequests++
		_, _ = fmt.Fprint(w, `{"opencode":{"npm":"@ai-sdk/openai-compatible","models":{"free":{"id":"free","status":"active","cost":{"input":0,"output":0}},"beta-free":{"id":"beta-free","status":"beta","cost":{"input":0,"output":0}},"unstated":{"id":"unstated","cost":{"input":0,"output":0}},"input-only":{"id":"input-only","status":"active","cost":{"input":0,"output":1}},"snapshot-only":{"id":"snapshot-only","status":"active","cost":{"input":0,"output":0}}}}}`)
	}))
	defer server.Close()
	client := testClient(t, server.URL+"/zen", server.URL+"/metadata", time.Now())
	evidence, err := client.DiscoverVerified(context.Background())
	if err != nil || len(evidence.Models) != 2 || evidence.Models[0].ID != "beta-free" || evidence.Models[1].ID != "free" || metadataRequests != 1 {
		t.Fatalf("evidence=%+v err=%v", evidence, err)
	}
}

func TestDiscoverFailureAndEmptyEvidence(t *testing.T) {
	t.Run("HTTP failure", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { http.Error(w, "private", http.StatusUnauthorized) }))
		defer server.Close()
		client := testClient(t, server.URL, server.URL, time.Now())
		evidence, err := client.Discover(context.Background())
		if err == nil || evidence.Status != core.CatalogFailed || strings.Contains(err.Error(), "private") {
			t.Fatalf("evidence=%+v err=%v", evidence, err)
		}
	})
	t.Run("empty intersection", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.HasSuffix(r.URL.Path, "/models") {
				_, _ = fmt.Fprint(w, `{"data":[{"id":"paid"}]}`)
			} else {
				_, _ = fmt.Fprint(w, `{"opencode":{"npm":"@ai-sdk/openai-compatible","models":{"paid":{"id":"paid","cost":{"input":1,"output":0}}}}}`)
			}
		}))
		defer server.Close()
		client := testClient(t, server.URL, server.URL+"/metadata", time.Now())
		evidence, err := client.Discover(context.Background())
		if err != nil || evidence.Status != core.CatalogEmpty || len(evidence.Models) != 0 {
			t.Fatalf("evidence=%+v err=%v", evidence, err)
		}
	})
	t.Run("public snapshot does not require Zen shape", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.HasSuffix(r.URL.Path, "/models") {
				_, _ = fmt.Fprint(w, `{}`)
			} else {
				_, _ = fmt.Fprint(w, `{"opencode":{"npm":"@ai-sdk/openai-compatible","models":{}}}`)
			}
		}))
		defer server.Close()
		client := testClient(t, server.URL, server.URL+"/metadata", time.Now())
		evidence, err := client.Discover(context.Background())
		if err != nil || evidence.Status != core.CatalogEmpty {
			t.Fatalf("evidence=%+v err=%v", evidence, err)
		}
	})
}

func TestNativeSurfaceRejectsAmbiguousOrMissingCapabilities(t *testing.T) {
	for _, model := range []core.ModelInfo{
		{},
		{Capabilities: &core.ModelCapabilities{}},
		{Capabilities: &core.ModelCapabilities{Surfaces: core.ModelSurfaceCapabilities{ChatCompletions: core.SupportSupported, Responses: core.SupportSupported}}},
	} {
		if surface, ok := NativeSurface(model); ok || surface != "" {
			t.Fatalf("surface=%q ok=%v for %+v", surface, ok, model)
		}
	}
}

func TestCompleteNativeUsesSurfaceAndClassifiableErrors(t *testing.T) {
	var paths []string
	var identities []InvocationIdentity
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if r.Header.Get("Authorization") != "Bearer public" {
			t.Fatalf("headers=%v", r.Header)
		}
		identities = append(identities, InvocationIdentity{
			Project: r.Header.Get("x-opencode-project"), Session: r.Header.Get("x-opencode-session"),
			Request: r.Header.Get("x-opencode-request"), Client: r.Header.Get("x-opencode-client"),
			UserAgent: r.Header.Get("User-Agent"),
		})
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		if payload["model"] != "free" || payload["input"] != "hello" {
			t.Fatalf("payload=%+v", payload)
		}
		if r.URL.Path == "/zen/v1/responses" {
			_, _ = fmt.Fprint(w, `{"id":"response"}`)
			return
		}
		w.Header().Set("Retry-After", "7")
		http.Error(w, "do not expose this", http.StatusTooManyRequests)
	}))
	defer server.Close()
	client := testClient(t, server.URL+"/zen/v1", server.URL+"/metadata", time.Now())
	payload := map[string]any{"input": "hello"}
	identity := InvocationIdentity{Project: "project", Session: "session", Request: "request", Client: "desktop", UserAgent: AnonymousUserAgent}
	ctx := WithInvocationIdentity(context.Background(), identity)
	result, err := client.CompleteNative(ctx, "free", core.ModelSurfaceResponses, payload)
	if err != nil || result["id"] != "response" || payload["model"] != nil {
		t.Fatalf("result=%v payload=%v err=%v", result, payload, err)
	}
	_, err = client.CompleteNative(ctx, "free", core.ModelSurfaceChatCompletions, payload)
	var operationError *core.ProviderOperationError
	if !errors.As(err, &operationError) || operationError.Failure.StatusCode != http.StatusTooManyRequests || operationError.Failure.RetryAfter != "7" || strings.Contains(err.Error(), "do not expose") {
		t.Fatalf("error=%#v", err)
	}
	if !slices.Equal(paths, []string{"/zen/v1/responses", "/zen/v1/chat/completions"}) {
		t.Fatalf("paths=%v", paths)
	}
	if len(identities) != 2 || !reflect.DeepEqual(identities[0], identity) || !reflect.DeepEqual(identities[1], identity) {
		t.Fatalf("identities=%+v", identities)
	}
}

func testClient(t *testing.T, baseURL, metadataURL string, now time.Time) *Client {
	t.Helper()
	sequence := 0
	client, err := New(Config{
		Endpoints: Endpoints{BaseURL: baseURL, MetadataURL: metadataURL}, Now: func() time.Time { return now },
		NewID: func(prefix string) (string, error) { sequence++; return fmt.Sprintf("%s_%d", prefix, sequence), nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func TestNewValidatesEndpoints(t *testing.T) {
	for _, config := range []Config{
		{Endpoints: Endpoints{BaseURL: "://bad"}},
		{Endpoints: Endpoints{MetadataURL: "relative"}},
	} {
		if _, err := New(config); err == nil {
			t.Fatalf("New(%+v) succeeded", config)
		}
	}
}

func TestCapabilityJSONIsTyped(t *testing.T) {
	model := capabilities(metadataModel{}, core.ModelSurfaceChatCompletions, time.Now(), time.Hour)
	encoded, err := json.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"chat_completions":"supported"`) || !strings.Contains(string(encoded), `"responses":"unsupported"`) {
		t.Fatalf("capabilities=%s", encoded)
	}
}

func TestConnectReturnsNativeProbeEvidenceAndPublication(t *testing.T) {
	now := time.Date(2026, time.September, 21, 12, 0, 0, 0, time.UTC)
	var probePath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/zen/models":
			_, _ = fmt.Fprint(w, `{"data":[{"id":"responses-free"}]}`)
		case "/metadata":
			_, _ = fmt.Fprint(w, `{"opencode":{"npm":"@ai-sdk/openai-compatible","models":{"responses-free":{"id":"responses-free","status":"active","provider":{"npm":"@ai-sdk/openai"},"cost":{"input":0,"output":0},"tool_call":true}}}}`)
		case "/zen/responses":
			probePath = r.URL.Path
			if r.Header.Get("Accept") != "text/event-stream" {
				t.Fatalf("probe headers=%v", r.Header)
			}
			var payload map[string]any
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Fatal(err)
			}
			if payload["input"] != "Reply with: ok" || payload["messages"] != nil || payload["max_output_tokens"] != float64(16) || payload["tools"] != nil {
				t.Fatalf("probe payload=%+v", payload)
			}
			_, _ = fmt.Fprint(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	clockCalls := 0
	client, err := New(Config{
		ProviderID: "configured-zen",
		Endpoints:  Endpoints{BaseURL: server.URL + "/zen", MetadataURL: server.URL + "/metadata"},
		Now:        func() time.Time { clockCalls++; return now.Add(time.Duration(clockCalls) * time.Millisecond) },
		NewID:      func(prefix string) (string, error) { return prefix + "_id", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Connect(context.Background(), core.ProviderConnectRequest{
		Connection:        core.ProviderConnection{ProviderID: "configured-zen", Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous},
		PublicationPolicy: core.PublishVerifiedTargets,
	})
	if err != nil {
		t.Fatal(err)
	}
	if probePath != "/zen/responses" || result.Catalog.Status != core.CatalogDiscovered || result.Health.Status != core.ProviderHealthHealthy ||
		len(result.Probes) != 1 || !result.Probes[0].InferenceVerified() || result.Probes[0].Latency != time.Millisecond ||
		!reflect.DeepEqual(result.Targets, []core.Target{{Provider: "configured-zen", Model: "responses-free"}}) {
		t.Fatalf("path=%q result=%+v", probePath, result)
	}
}

func TestConnectRejectsWrongConnectionAndClassifiesProbeFailure(t *testing.T) {
	client, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Connect(context.Background(), core.ProviderConnectRequest{
		Connection:        core.ProviderConnection{ProviderID: DefaultProviderID, Kind: core.ProviderConnectionSystem, AuthKind: core.ProviderAuthAPIKey},
		PublicationPolicy: core.PublishDiscoveredTargets,
	})
	if err == nil || !strings.Contains(err.Error(), "anonymous connection") {
		t.Fatalf("error=%v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/zen/models":
			_, _ = fmt.Fprint(w, `{"data":[{"id":"chat-free"}]}`)
		case "/metadata":
			_, _ = fmt.Fprint(w, `{"opencode":{"npm":"@ai-sdk/openai-compatible","models":{"chat-free":{"id":"chat-free","status":"active","cost":{"input":0,"output":0}}}}}`)
		default:
			w.Header().Set("Retry-After", "3")
			http.Error(w, "limited", http.StatusTooManyRequests)
		}
	}))
	defer server.Close()
	client = testClient(t, server.URL+"/zen", server.URL+"/metadata", time.Now())
	result, err := client.Connect(context.Background(), core.ProviderConnectRequest{
		Connection:        core.ProviderConnection{ProviderID: DefaultProviderID, Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous},
		PublicationPolicy: core.PublishDiscoveredTargets,
	})
	if err != nil || result.Health.Status != core.ProviderHealthDegraded || result.Health.ErrorClass != core.ProviderErrorRateLimited ||
		len(result.Probes) != 1 || result.Probes[0].Status != core.CompletionFailed || len(result.Targets) != 1 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestConnectDoesNotVerifyMalformedSSE(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/zen/models":
			_, _ = fmt.Fprint(w, `{"data":[{"id":"chat-free"}]}`)
		case "/metadata":
			_, _ = fmt.Fprint(w, `{"opencode":{"npm":"@ai-sdk/openai-compatible","models":{"chat-free":{"id":"chat-free","status":"active","cost":{"input":0,"output":0}}}}}`)
		default:
			_, _ = fmt.Fprint(w, "data: {\"unexpected\":true}\n\ndata: [DONE]\n\n")
		}
	}))
	defer server.Close()
	client := testClient(t, server.URL+"/zen", server.URL+"/metadata", time.Now())
	result, err := client.Connect(context.Background(), core.ProviderConnectRequest{
		Connection:        core.ProviderConnection{ProviderID: DefaultProviderID, Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous},
		PublicationPolicy: core.PublishVerifiedTargets,
	})
	if err != nil || result.Health.Status != core.ProviderHealthUnhealthy || result.Health.ErrorClass != core.ProviderErrorUpstream ||
		len(result.Probes) != 1 || result.Probes[0].Status != core.CompletionFailed || len(result.Targets) != 0 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestProbeSSERequiresCompleteSuccessfulTerminalStream(t *testing.T) {
	tests := []struct {
		name    string
		surface core.ModelSurface
		stream  string
		valid   bool
	}{
		{name: "chat complete", surface: core.ModelSurfaceChatCompletions, stream: "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", valid: true},
		{name: "chat first delta only", surface: core.ModelSurfaceChatCompletions, stream: "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"},
		{name: "chat done without finish", surface: core.ModelSurfaceChatCompletions, stream: "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"},
		{name: "chat event after finish", surface: core.ModelSurfaceChatCompletions, stream: "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\ndata: [DONE]\n\n"},
		{name: "responses complete", surface: core.ModelSurfaceResponses, stream: "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n", valid: true},
		{name: "responses delta only", surface: core.ModelSurfaceResponses, stream: "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n"},
		{name: "responses malformed", surface: core.ModelSurfaceResponses, stream: "data: {not-json}\n\n"},
		{name: "responses truncated", surface: core.ModelSurfaceResponses, stream: "data: {\"type\":\"response.completed\""},
		{name: "responses undelimited terminal", surface: core.ModelSurfaceResponses, stream: "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}"},
		{name: "responses failed", surface: core.ModelSurfaceResponses, stream: "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\"}}\n\n"},
		{name: "responses incomplete", surface: core.ModelSurfaceResponses, stream: "data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\"}}\n\n"},
		{name: "responses invalid completion", surface: core.ModelSurfaceResponses, stream: "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"failed\"}}\n\n"},
		{name: "responses event after terminal", surface: core.ModelSurfaceResponses, stream: "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"late\"}\n\n"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateProbeSSE(test.surface, []byte(test.stream))
			if (err == nil) != test.valid {
				t.Fatalf("error=%v, valid=%v", err, test.valid)
			}
		})
	}
}

func TestConnectRejectsProviderIdentityMismatch(t *testing.T) {
	client, err := New(Config{ProviderID: "configured-zen"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Connect(context.Background(), core.ProviderConnectRequest{
		Connection:        core.ProviderConnection{ProviderID: "other-zen", Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous},
		PublicationPolicy: core.PublishDiscoveredTargets,
	})
	if err == nil || !strings.Contains(err.Error(), "does not match") || result.Connection.ProviderID != "other-zen" || result.Catalog.Status != "" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestConnectPreservesCallerCancellationWithoutUnhealthyEvidence(t *testing.T) {
	cause := errors.New("caller stopped Zen connection")
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(cause)
	client, err := New(Config{HTTPClient: httpClientFunc(func(request *http.Request) (*http.Response, error) {
		return nil, request.Context().Err()
	})})
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Connect(ctx, core.ProviderConnectRequest{
		Connection:        core.ProviderConnection{ProviderID: DefaultProviderID, Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous},
		PublicationPolicy: core.PublishDiscoveredTargets,
	})
	if !errors.Is(err, cause) || result.Catalog.Status != core.CatalogNotProbed || result.Health.Status != core.ProviderHealthUnknown || result.Health.ErrorClass != core.ProviderErrorNone {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestConnectCancellationDuringProbeIsNotProviderFailureEvidence(t *testing.T) {
	probeStarted := make(chan struct{})
	releaseProbe := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/zen/models":
			_, _ = fmt.Fprint(w, `{"data":[{"id":"chat-free"}]}`)
		case "/metadata":
			_, _ = fmt.Fprint(w, `{"opencode":{"npm":"@ai-sdk/openai-compatible","models":{"chat-free":{"id":"chat-free","status":"active","cost":{"input":0,"output":0}}}}}`)
		default:
			close(probeStarted)
			select {
			case <-r.Context().Done():
			case <-releaseProbe:
			}
		}
	}))
	defer server.Close()
	client := testClient(t, server.URL+"/zen", server.URL+"/metadata", time.Now())
	cause := errors.New("caller stopped probe")
	ctx, cancel := context.WithCancelCause(context.Background())
	done := make(chan struct {
		result core.ProviderConnectResult
		err    error
	}, 1)
	go func() {
		result, err := client.Connect(ctx, core.ProviderConnectRequest{
			Connection:        core.ProviderConnection{ProviderID: DefaultProviderID, Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous},
			PublicationPolicy: core.PublishVerifiedTargets,
		})
		done <- struct {
			result core.ProviderConnectResult
			err    error
		}{result: result, err: err}
	}()
	<-probeStarted
	cancel(cause)
	outcome := <-done
	close(releaseProbe)
	if !errors.Is(outcome.err, cause) || outcome.result.Health.Status != core.ProviderHealthUnknown || len(outcome.result.Probes) != 1 || outcome.result.Probes[0].Status != core.CompletionNotProbed || len(outcome.result.Targets) != 0 {
		t.Fatalf("result=%+v err=%v", outcome.result, outcome.err)
	}
}

func TestConfigDoesNotRetainCallerMaps(t *testing.T) {
	payload := map[string]any{"input": "hello"}
	clone := cloneMap(payload)
	clone["input"] = "changed"
	if reflect.DeepEqual(payload, clone) || payload["input"] != "hello" {
		t.Fatalf("payload=%v clone=%v", payload, clone)
	}
}
