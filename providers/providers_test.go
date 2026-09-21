package providers_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

func TestRegistryIntegrityAndManifest(t *testing.T) {
	entries := providers.ProviderRegistry()
	if len(entries) != 24 {
		t.Fatalf("expected 24 registry entries, got %d", len(entries))
	}

	seen := make(map[string]bool)
	for _, entry := range entries {
		if seen[entry.ID] {
			t.Fatalf("duplicate provider id: %s", entry.ID)
		}
		seen[entry.ID] = true
	}

	// Verify lookup by id and alias
	copilot, ok := providers.RegistryProvider("github_copilot")
	if !ok || copilot.Label != "GitHub Copilot" {
		t.Fatalf("failed to resolve github_copilot: %+v", copilot)
	}

	byAlias, ok := providers.RegistryProvider("copilot")
	if !ok || byAlias.ID != "github_copilot" {
		t.Fatalf("failed to resolve by alias 'copilot': %+v", byAlias)
	}

	canonical := providers.CanonicalRegistryID("zen")
	if canonical != "opencode_zen" {
		t.Fatalf("expected canonical id 'opencode_zen', got %s", canonical)
	}
}

func TestByteStreamIterFramesSplitAndCoalescedSSERecords(t *testing.T) {
	reader := &chunkReader{chunks: [][]byte{
		[]byte("data: fir"),
		[]byte("st\n\ndata: second\n\ndata: third"),
		[]byte("\n\n"),
	}}
	iter := providers.NewByteStreamIter(reader)
	defer iter.Close()

	want := []string{"data: first\n\n", "data: second\n\n", "data: third\n\n"}
	for i, expected := range want {
		frame, err := iter.Next()
		if err != nil || string(frame) != expected {
			t.Fatalf("frame %d=%q err=%v, want %q", i, frame, err, expected)
		}
	}
	if frame, err := iter.Next(); len(frame) != 0 || err != io.EOF {
		t.Fatalf("terminal frame=%q err=%v", frame, err)
	}
}

func TestByteStreamIterRejectsTruncatedSSERecord(t *testing.T) {
	iter := providers.NewByteStreamIter(&chunkReader{chunks: [][]byte{[]byte("data: truncated")}})
	defer iter.Close()

	if frame, err := iter.Next(); len(frame) != 0 || err == nil {
		t.Fatalf("frame=%q err=%v", frame, err)
	}
}

func TestByteStreamIterRejectsOversizedSSERecord(t *testing.T) {
	iter := providers.NewByteStreamIter(io.NopCloser(strings.NewReader("data: " + strings.Repeat("x", 1<<20) + "\n\n")))
	defer iter.Close()

	if frame, err := iter.Next(); len(frame) != 0 || err == nil {
		t.Fatalf("frame length=%d err=%v", len(frame), err)
	}
}

type chunkReader struct {
	chunks [][]byte
}

func (r *chunkReader) Read(p []byte) (int, error) {
	if len(r.chunks) == 0 {
		return 0, io.EOF
	}
	chunk := r.chunks[0]
	r.chunks = r.chunks[1:]
	return copy(p, chunk), nil
}

func (*chunkReader) Close() error { return nil }

func TestAnonymousProfiles(t *testing.T) {
	profiles := providers.AnonymousProviderProfiles()
	if len(profiles) != 5 {
		t.Fatalf("expected 5 anonymous provider profiles, got %d", len(profiles))
	}

	profileMap := make(map[string]providers.AnonymousProviderProfile)
	for _, p := range profiles {
		profileMap[p.RegistryID] = p
	}

	expected := []string{"kilo_code", "llm7", "opencode_zen", "ovh_ai_endpoints", "pollinations"}
	for _, id := range expected {
		if _, ok := profileMap[id]; !ok {
			t.Fatalf("missing anonymous profile %s", id)
		}
	}

	// Test probe model selection
	models := []core.ModelInfo{
		{ID: "other-model"},
		{ID: "openai-fast"},
	}
	probe := providers.AnonymousVerificationModel("pollinations", models)
	if probe != "openai-fast" {
		t.Fatalf("expected openai-fast for pollinations probe, got %s", probe)
	}
}

func TestErrorClassification(t *testing.T) {
	if !providers.IsThrottle(errors.New("rate limit exceeded")) {
		t.Fatal("expected throttle detection from rate limit string")
	}
	if !providers.IsThrottle(&providers.InvocationError{Status: 429}) {
		t.Fatal("expected throttle detection from 429 status")
	}

	for _, status := range []int{500, 502, 503, 504} {
		retryableErr := &providers.InvocationError{Status: status}
		if !providers.InvocationRetryable(retryableErr) {
			t.Errorf("expected %d to be retryable", status)
		}
		if !providers.InvocationFailoverEligible(retryableErr) {
			t.Errorf("expected %d to be failover eligible", status)
		}
		if !providers.InvocationCircuitFailure(retryableErr) {
			t.Errorf("expected retryable %d to count as a circuit failure", status)
		}
	}

	unrecoverableErr := &providers.InvocationError{Status: 401}
	if providers.InvocationRetryable(unrecoverableErr) {
		t.Fatal("401 should not be retryable")
	}
	if providers.InvocationCircuitFailure(unrecoverableErr) {
		t.Fatal("401 should not count as a circuit failure")
	}

	statusZero := &providers.InvocationError{Retryable: true}
	if !providers.InvocationRetryable(statusZero) || providers.InvocationFailoverEligible(statusZero) {
		t.Fatal("status-zero retry and failover flags must be honored independently")
	}
	statusZero.FailoverEligible = true
	if !providers.InvocationFailoverEligible(statusZero) {
		t.Fatal("explicit status-zero failover flag was ignored")
	}
	canceled := &providers.InvocationError{
		Retryable: true, FailoverEligible: true, CircuitFailure: true, Cause: context.Canceled,
	}
	if providers.InvocationRetryable(canceled) || providers.InvocationFailoverEligible(canceled) || providers.InvocationCircuitFailure(canceled) {
		t.Fatal("caller cancellation must not retry, fail over, or trip a circuit")
	}
	deadline := &providers.InvocationError{
		Retryable: true, FailoverEligible: true, CircuitFailure: true, Cause: context.DeadlineExceeded,
	}
	if providers.InvocationRetryable(deadline) || providers.InvocationFailoverEligible(deadline) || providers.InvocationCircuitFailure(deadline) {
		t.Fatal("caller deadline must not retry, fail over, or trip a circuit")
	}
}

func TestMockAutoConnectAnonymousProviders(t *testing.T) {
	// Mock server that answers /models and /chat/completions
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{
					{
						"id":     "ling-3.0-flash-fin-free",
						"object": "model",
					},
				},
			})
		case "/chat/completions":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "chatcmpl-mock",
				"object": "chat.completion",
				"model":  "ling-3.0-flash-fin-free",
				"choices": []any{
					map[string]any{
						"index": 0,
						"message": map[string]any{
							"role":    "assistant",
							"content": "ok",
						},
						"finish_reason": "stop",
					},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	// Discover models directly with mock server
	profile := providers.AnonymousProviderProfile{
		RegistryID:         "opencode_zen",
		ProviderID:         "mock-zen",
		RuntimeType:        "openai_compatible",
		BaseURL:            server.URL,
		VerificationModels: []string{"ling-3.0-flash-fin-free"},
	}

	models, err := providers.DiscoverAnonymousModels(context.Background(), profile, server.Client())
	if err != nil {
		t.Fatalf("unexpected discovery error: %v", err)
	}
	if len(models) != 1 || models[0].ID != "ling-3.0-flash-fin-free" {
		t.Fatalf("unexpected models: %+v", models)
	}

	// Verify probe completion via OpenAIProvider
	p := providers.NewOpenAIProvider("mock-zen", server.URL, "none", server.Client())
	resp, err := p.Complete(context.Background(), "ling-3.0-flash-fin-free", map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected probe error: %v", err)
	}
	if resp["id"] != "chatcmpl-mock" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}
