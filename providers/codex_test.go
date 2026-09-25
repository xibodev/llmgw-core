package providers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	auth "github.com/xibodev/llm-provider-auth"
	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

func TestCodexCatalogKnownEnvelopesAndExactIdentity(t *testing.T) {
	discoveredAt := time.Date(2026, time.September, 21, 12, 0, 0, 0, time.UTC)
	data := readCodexFixture(t, "catalog-data.json")
	models, err := parseCodexCatalog(data, discoveredAt)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].ID != "codex-alpha-2026-09" || models[0].OwnedBy != "example-vendor" {
		t.Fatalf("data catalog = %+v", models)
	}
	if models[0].APIEligible == nil || !*models[0].APIEligible || models[0].APIVisibility != "list" || len(models[0].SupportedAPIs) != 1 || models[0].SupportedAPIs[0] != "/responses" {
		t.Fatalf("eligibility fields = %+v", models[0])
	}
	capabilities := models[0].Capabilities
	if capabilities == nil || capabilities.SchemaVersion != 1 ||
		capabilities.Operations.Chat != core.SupportSupported ||
		capabilities.Surfaces.Responses != core.SupportSupported ||
		capabilities.Surfaces.ChatCompletions != core.SupportUnsupported ||
		capabilities.Streaming != core.SupportSupported ||
		capabilities.Provenance.Source != core.ModelCapabilitySourceUpstreamReported ||
		capabilities.Provenance.Confidence != core.ModelCapabilityConfidenceHigh ||
		capabilities.Freshness.DiscoveredAt == nil || !capabilities.Freshness.DiscoveredAt.Equal(discoveredAt) ||
		capabilities.Freshness.ExpiresAt == nil || !capabilities.Freshness.ExpiresAt.Equal(discoveredAt.Add(time.Hour)) {
		t.Fatalf("capability evidence = %+v", capabilities)
	}

	models, err = parseCodexCatalog(readCodexFixture(t, "catalog-models.json"), discoveredAt)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "codex-beta-exact" || models[1].ID != "codex-name-identity" || models[0].APIEligible == nil || *models[0].APIEligible || models[0].Capabilities != nil || models[1].Capabilities != nil {
		t.Fatalf("models catalog = %+v", models)
	}
}

func TestCodexCatalogRejectsUnknownAmbiguousAndFabricatedIdentity(t *testing.T) {
	for name, fixture := range map[string]string{
		"unknown envelope":     `{"items":[]}`,
		"ambiguous envelope":   `{"data":[],"models":[]}`,
		"missing identity":     `{"data":[{"display_name":"Not an identity"}]}`,
		"blank exact identity": `{"data":[{"id":" "}]}`,
		"wrong eligibility":    `{"data":[{"id":"exact","supported_in_api":"yes"}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := parseCodexCatalog([]byte(fixture), time.Now()); err == nil {
				t.Fatal("malformed catalog accepted")
			}
		})
	}
}

func TestCodexCompleteUsesAuthenticatedStreamingResponses(t *testing.T) {
	fixture := readCodexFixture(t, "response-complete.sse")
	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		assertCodexRequestFixture(t, r, request, "request-chat.json")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(fixture)
	}))
	defer server.Close()

	provider := newFixtureCodexProvider(t, server, "Follow the caller's request.")
	response, err := provider.Complete(context.Background(), "codex-alpha-2026-09", map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "Use the tool"}},
		"tools": []any{map[string]any{"type": "function", "function": map[string]any{
			"name": "lookup", "description": "Lookup", "parameters": map[string]any{"type": "object"},
		}}},
		"prompt_cache_key": "fixture-cache",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	choices := response["choices"].([]any)
	message := choices[0].(map[string]any)["message"].(map[string]any)
	usage := response["usage"].(map[string]any)
	if message["content"] != nil || message["reasoning_content"] != "fixture reasoning" || len(message["tool_calls"].([]any)) != 1 || usage["total_tokens"] != 18 {
		t.Fatalf("completion response = %+v", response)
	}
}

func TestCodexCompleteResponsesPreservesNativeRequestAndOutput(t *testing.T) {
	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		assertCodexRequestFixture(t, r, request, "request-native.json")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"type":"response.completed","sequence_number":4,"response":{"id":"resp_native","object":"response","status":"completed","model":"exact-model","conversation":{"id":"conv_1"},"output":[{"id":"rs_1","type":"reasoning","encrypted_content":"opaque","summary":[{"type":"summary_text","text":"kept"}]},{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[{"type":"url_citation","url":"https://example.test"}]}]}]}}`+"\n\n")
	}))
	defer server.Close()
	var provider Provider = newFixtureCodexProvider(t, server, "Required instructions")
	responses, ok := provider.(ResponsesProvider)
	if !ok {
		t.Fatal("Codex provider does not expose the optional native Responses interface")
	}
	payload := map[string]any{
		"input":                "continue",
		"instructions":         "Caller instructions",
		"previous_response_id": "resp_previous",
		"conversation":         "conv_1",
		"include":              []any{"reasoning.encrypted_content"},
		"reasoning":            map[string]any{"effort": "high", "summary": "auto"},
		"tools":                []any{map[string]any{"type": "web_search"}},
		"prompt_cache_key":     "fixture-cache",
		"force_api_support":    true,
		"stream":               false,
	}
	response, err := responses.CompleteResponses(context.Background(), "exact-model", payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	output := response["output"].([]any)
	if len(output) != 2 || output[0].(map[string]any)["encrypted_content"] != "opaque" || response["conversation"].(map[string]any)["id"] != "conv_1" {
		t.Fatalf("native response output was not preserved: %+v", response)
	}
	if payload["stream"] != false || payload["instructions"] != "Caller instructions" {
		t.Fatalf("caller payload was mutated: %+v", payload)
	}
}

func TestCodexCompleteResponsesUsesCompletedOutputItems(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}}`,
			`data: {"type":"response.completed","response":{"id":"resp_live","output":[]}}`,
		}, "\n\n")+"\n\n")
	}))
	defer server.Close()
	provider := newFixtureCodexProvider(t, server, "Required instructions")
	response, err := provider.CompleteResponses(context.Background(), "exact-model", map[string]any{"input": "continue"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	output, ok := response["output"].([]any)
	if !ok || len(output) != 1 || output[0].(map[string]any)["id"] != "msg_1" {
		t.Fatalf("completed output items were not retained: %+v", response)
	}
}

func TestCodexCompleteResponsesUsesStreamedTextWhenItemsAreOmitted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"type":"response.output_text.delta","delta":"streamed answer"}`,
			`data: {"type":"response.completed","response":{"id":"resp_live"}}`,
		}, "\n\n")+"\n\n")
	}))
	defer server.Close()
	provider := newFixtureCodexProvider(t, server, "Required instructions")
	response, err := provider.CompleteResponses(context.Background(), "exact-model", map[string]any{"input": "continue"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	output := response["output"].([]any)
	content := output[0].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["text"] != "streamed answer" {
		t.Fatalf("streamed text was not retained: %+v", response)
	}
}

func TestCodexCompleteResponsesAcceptsMissingStreamingContentType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header()["Content-Type"] = nil
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"type":"response.output_text.delta","delta":"answer"}`,
			`data: {"type":"response.completed","response":{"id":"resp_live"}}`,
		}, "\n\n")+"\n\n")
	}))
	defer server.Close()
	provider := newFixtureCodexProvider(t, server, "Required instructions")
	response, err := provider.CompleteResponses(context.Background(), "exact-model", map[string]any{"input": "continue"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if response["id"] != "resp_live" {
		t.Fatalf("response = %+v", response)
	}
}

func TestCodexResponsesRejectsUnsupportedExecutionModes(t *testing.T) {
	provider, err := NewCodexProvider(CodexProviderConfig{
		SessionSource: NewCodexTokenSessionSource(auth.NewStaticTokenSource(&auth.Token{AccessToken: "fixture"}), ""),
		Instructions:  "Required instructions", ResponsesURL: "http://unused", ClientVersion: fixtureCodexClientVersion,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, payload := range []map[string]any{
		{"input": "hello", "store": true},
		{"input": "hello", "background": true},
	} {
		if _, err := provider.CompleteResponses(context.Background(), "exact-model", payload, nil); err == nil {
			t.Fatalf("unsupported Codex request accepted: %+v", payload)
		}
	}
}

func TestCodexHTTPErrorExposesOnlyStructuredIdentifiers(t *testing.T) {
	response := &http.Response{
		StatusCode: http.StatusBadRequest,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(`{"error":{"message":"Authorization: Bearer secret-token user@example.test","type":"invalid_request_error","code":"unsupported_parameter","param":"max_output_tokens"}}`)),
	}
	err := invocationHTTPError(response)
	want := "Codex upstream request failed (code=unsupported_parameter, type=invalid_request_error, param=max_output_tokens)"
	if err.Error() != want {
		t.Fatalf("error = %q, want %q", err, want)
	}
	if strings.Contains(err.Error(), "secret-token") || strings.Contains(err.Error(), "example.test") {
		t.Fatalf("error exposed upstream message: %v", err)
	}

	response.Body = io.NopCloser(strings.NewReader(`{"error":{"type":"invalid request: Bearer secret-token","code":"bad/value","param":"user@example.test"}}`))
	if got := invocationHTTPError(response).Error(); got != "Codex upstream request failed" {
		t.Fatalf("unsafe identifiers were exposed: %q", got)
	}
}

func TestCodexRejectsUnprovenTools(t *testing.T) {
	provider, err := NewCodexProvider(CodexProviderConfig{
		SessionSource: NewCodexTokenSessionSource(auth.NewStaticTokenSource(&auth.Token{AccessToken: "fixture"}), ""),
		Instructions:  "Required instructions", ResponsesURL: "http://unused", ClientVersion: fixtureCodexClientVersion,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, payload := range []map[string]any{
		{"messages": []any{map[string]any{"role": "user", "content": "hello"}}, "tools": []any{map[string]any{"type": "computer"}}},
		{"input": "hello", "tools": []any{map[string]any{"type": "file_search"}}},
	} {
		if payload["messages"] != nil {
			if _, err := provider.Complete(context.Background(), "exact-model", payload, nil); err == nil {
				t.Fatalf("unproven Chat tool accepted: %+v", payload)
			}
		} else if _, err := provider.CompleteResponses(context.Background(), "exact-model", payload, nil); err == nil {
			t.Fatalf("unproven Responses tool accepted: %+v", payload)
		}
	}
}

func TestCodexStreamResponsesPreservesNativeSSEEvents(t *testing.T) {
	frames := []string{
		": keepalive\r\nretry: 1000\r\n\r\n",
		": upstream comment\r\nevent: response.created\r\nid: evt-0\r\nretry: 1500\r\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\r\ndata: \"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"output\":[]}}\r\n\r\n",
		"event: response.future.delta\nid: evt-1\ndata: {\"type\":\"response.future.delta\",\"sequence_number\":1,\"custom\":{\"nested\":true}}\n\n",
		"retry: 2500\ndata: {\"type\":\"response.reasoning_summary_text.delta\",\"sequence_number\":2,\"delta\":\"thinking\"}\n\n",
		"event: response.completed\nid: evt-3\ndata: {\"type\":\"response.completed\",\"sequence_number\":3,\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[{\"type\":\"reasoning\",\"encrypted_content\":\"opaque\"}]}}\n\n",
	}
	stream := strings.Join(frames, "")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, stream)
	}))
	defer server.Close()
	var provider Provider = newFixtureCodexProvider(t, server, "Required instructions")
	responses, ok := provider.(ResponsesStreamProvider)
	if !ok {
		t.Fatal("Codex provider does not expose the optional native Responses streaming interface")
	}
	iter, err := responses.StreamResponses(context.Background(), "exact-model", map[string]any{"input": "go"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer iter.Close()
	var got strings.Builder
	for {
		frame, err := iter.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		got.Write(frame)
	}
	if got.String() != stream {
		t.Fatalf("native stream changed upstream SSE frames:\n got %q\nwant %q", got.String(), stream)
	}
}

func TestCodexStreamResponsesRequiresClosedTerminalFrame(t *testing.T) {
	stream := `data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}`
	iter := newCodexResponsesStreamIter(io.NopCloser(strings.NewReader(stream)))
	defer iter.Close()

	frame, err := iter.Next()
	var invocation *InvocationError
	if len(frame) != 0 || !errors.As(err, &invocation) || invocation.Cause == nil || !strings.Contains(invocation.Cause.Error(), "incomplete event frame") {
		t.Fatalf("frame=%q err=%v", frame, err)
	}
}

func TestCodexStreamExposesDeltasToolsReasoningAndUsage(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"type":"response.reasoning_summary_text.delta","delta":"thinking"}`,
		`data: {"type":"response.output_text.delta","delta":"answer"}`,
		`data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup"}}`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{}"}`,
		`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}`,
	}, "\n\n") + "\n\n"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, stream)
	}))
	defer server.Close()
	provider := newFixtureCodexProvider(t, server, "Required instructions")
	iter, err := provider.Stream(context.Background(), "exact-model", map[string]any{"messages": []any{map[string]any{"role": "user", "content": "go"}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer iter.Close()
	var chunks string
	for {
		chunk, err := iter.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		chunks += string(chunk)
	}
	for _, expected := range []string{`"reasoning_content":"thinking"`, `"content":"answer"`, `"name":"lookup"`, `"arguments":"{}"`, `"total_tokens":5`} {
		if !strings.Contains(chunks, expected) {
			t.Fatalf("stream missing %s: %s", expected, chunks)
		}
	}
	if !strings.Contains(chunks, "data: {") || !strings.HasSuffix(chunks, "data: [DONE]\n\n") {
		t.Fatalf("stream is not valid terminal SSE: %q", chunks)
	}
}

func TestCodexRejectsUnsupportedFieldsAndRequiresInstructions(t *testing.T) {
	source := NewCodexTokenSessionSource(auth.NewStaticTokenSource(&auth.Token{AccessToken: "fixture"}), "")
	if _, err := NewCodexProvider(CodexProviderConfig{SessionSource: source, ClientVersion: fixtureCodexClientVersion}); err == nil {
		t.Fatal("empty instructions accepted")
	}
	provider, err := NewCodexProvider(CodexProviderConfig{SessionSource: source, Instructions: "required", ResponsesURL: "http://unused", ClientVersion: fixtureCodexClientVersion})
	if err != nil {
		t.Fatal(err)
	}
	for _, payload := range []map[string]any{
		{"messages": []any{map[string]any{"role": "user", "content": "hello"}}, "temperature": 0.7},
		{"messages": []any{map[string]any{"role": "user", "content": "hello"}}, "tool_choice": "required"},
		{"messages": []any{map[string]any{"role": "user", "content": "hello"}}, "response_format": map[string]any{"type": "json_object"}},
	} {
		_, err = provider.Complete(context.Background(), "exact-model", payload, nil)
		var invocation *InvocationError
		if !errors.As(err, &invocation) || !strings.Contains(err.Error(), "unsupported field") {
			t.Fatalf("unsupported Chat field error = %T %v", err, err)
		}
	}
	for _, field := range []string{"temperature", "top_p", "max_output_tokens"} {
		_, err = provider.CompleteResponses(context.Background(), "exact-model", map[string]any{"input": "hello", field: 1}, nil)
		var invocation *InvocationError
		if !errors.As(err, &invocation) || !strings.Contains(err.Error(), "unsupported field") {
			t.Fatalf("unsupported Responses field %q error = %T %v", field, err, err)
		}
	}
}

func TestCodexRequiresClientVersionAndSendsItToTheCatalog(t *testing.T) {
	source := NewCodexTokenSessionSource(auth.NewStaticTokenSource(&auth.Token{AccessToken: "fixture"}), "")
	for _, version := range []string{"", " \t"} {
		_, err := NewCodexProvider(CodexProviderConfig{SessionSource: source, Instructions: "required", ClientVersion: version})
		if err == nil || err.Error() != "Codex client version is required" {
			t.Fatalf("client version %q: err = %v", version, err)
		}
	}

	queries := make(chan url.Values, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		queries <- r.URL.Query()
		_, _ = io.WriteString(w, `{"models":[]}`)
	}))
	defer server.Close()
	provider := newFixtureCodexProvider(t, server, "required")
	if _, err := provider.ListModels(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	if got := (<-queries)["client_version"]; len(got) != 1 || got[0] != fixtureCodexClientVersion {
		t.Fatalf("catalog client_version = %q, want %q", got, fixtureCodexClientVersion)
	}
}

func TestCodexTerminalStatusMustMatchEventAndOutputMustExist(t *testing.T) {
	for name, stream := range map[string]string{
		"contradictory status": `data: {"type":"response.completed","response":{"id":"resp_1","status":"failed","output":[]}}` + "\n\n",
		"missing output":       `data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}` + "\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := readCodexFinalResponse(strings.NewReader(stream)); err == nil {
				t.Fatal("invalid terminal response accepted")
			}
		})
	}
}

func TestCodexDecodeErrorDoesNotExposeUpstreamIdentifiers(t *testing.T) {
	stream := `data: {"type":"Bearer secret-token user@example.test"}` + "\n\n"
	iter := newCodexStreamIter(io.NopCloser(strings.NewReader(stream)), "exact")
	_, err := iter.Next()
	if err == nil || strings.Contains(err.Error(), "secret-token") || strings.Contains(err.Error(), "example.test") {
		t.Fatalf("public error exposed upstream identifier: %v", err)
	}
}

func TestCodexSSERejectsMalformedTruncatedAndMissingTerminal(t *testing.T) {
	for name, stream := range map[string]string{
		"malformed":        "data: {not-json}\n\n",
		"truncated":        `data: {"type":"response.completed"`,
		"missing terminal": `data: {"type":"response.output_text.delta","delta":"partial"}\n\n`,
		"done only":        "data: [DONE]\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := readCodexFinalResponse(strings.NewReader(stream)); err == nil {
				t.Fatal("invalid stream accepted")
			}
		})
	}
}

func TestCodexRejectsMissingTerminalChatEventAndResponseMaterialLoss(t *testing.T) {
	if _, err := readCodexFinalResponse(strings.NewReader("data: {\"type\":\"response.future.delta\"}\n\n")); err == nil || !strings.Contains(err.Error(), "without a terminal response event") {
		t.Fatalf("buffered missing terminal error = %v", err)
	}

	unknown := newCodexStreamIter(io.NopCloser(strings.NewReader("data: {\"type\":\"response.future.delta\"}\n\n")), "exact")
	_, err := unknown.Next()
	var invocation *InvocationError
	if !errors.As(err, &invocation) || invocation.Cause == nil || !strings.Contains(invocation.Cause.Error(), "unsupported Codex streaming event") {
		t.Fatalf("unknown event error = %v", err)
	}

	response := map[string]any{
		"status": "completed",
		"output": []any{map[string]any{"type": "computer_call", "id": "unsupported"}},
	}
	_, err = codexResponseToChat("exact", response)
	var loss *translate.MaterialLossError
	if !errors.As(err, &loss) {
		t.Fatalf("response material loss error = %T %v", err, err)
	}
}

func TestCodexCancellationAndRetryAfter(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			w.Header().Set("Retry-After", "7")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		close(started)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer server.Close()
	defer close(release)
	provider := newFixtureCodexProvider(t, server, "required")
	_, err := provider.ListModels(context.Background(), nil)
	var catalog *CatalogError
	if !errors.As(err, &catalog) || catalog.Status != 429 || catalog.RetryAfter != 7*time.Second {
		t.Fatalf("catalog error = %#v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := provider.Complete(ctx, "exact-model", map[string]any{"messages": []any{map[string]any{"role": "user", "content": "wait"}}}, nil)
		done <- err
	}()
	<-started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation error = %v", err)
	} else if InvocationRetryable(err) || InvocationFailoverEligible(err) || InvocationCircuitFailure(err) {
		t.Fatal("Codex cancellation must not retry, fail over, or trip a circuit")
	}
}

const fixtureCodexClientVersion = "fixture-client/1.0"

func newFixtureCodexProvider(t *testing.T, server *httptest.Server, instructions string) *CodexProvider {
	t.Helper()
	provider, err := NewCodexProvider(CodexProviderConfig{
		SessionSource: NewCodexTokenSessionSource(auth.NewStaticTokenSource(&auth.Token{AccessToken: "caller-token", TokenType: "Bearer"}), "account-fixture"),
		Instructions:  instructions, ResponsesURL: server.URL + "/responses", ModelsURL: server.URL + "/models", Client: server.Client(),
		ClientVersion: fixtureCodexClientVersion,
	})
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func readCodexFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "codex", name))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func readCodexJSONFixture(t *testing.T, name string) map[string]any {
	t.Helper()
	var fixture map[string]any
	if err := json.Unmarshal(readCodexFixture(t, name), &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func assertCodexRequestFixture(t *testing.T, request *http.Request, body map[string]any, name string) {
	t.Helper()
	fixture := readCodexJSONFixture(t, name)
	wantHeaders := fixture["headers"].(map[string]any)
	wantHeaders["X-Stainless-Os"] = codexSDKOS()
	wantHeaders["X-Stainless-Arch"] = codexSDKArch()
	wantHeaders["X-Stainless-Runtime-Version"] = runtime.Version()
	gotHeaders := map[string]any{}
	for name, values := range request.Header {
		value := strings.Join(values, ", ")
		if name == "Authorization" && value != "" {
			value = "Bearer <redacted>"
		}
		gotHeaders[name] = value
	}
	got := map[string]any{
		"method":  request.Method,
		"path":    request.URL.Path,
		"headers": gotHeaders,
		"body":    body,
	}
	if !reflect.DeepEqual(got, fixture) {
		t.Fatalf("upstream request drifted:\n got: %#v\nwant: %#v", got, fixture)
	}
}
