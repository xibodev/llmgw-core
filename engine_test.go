package core_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

type mockProvider struct {
	id string
}

func (m *mockProvider) Complete(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (map[string]any, error) {
	return map[string]any{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "Hello from mock provider " + m.id,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     10,
			"completion_tokens": 8,
			"total_tokens":      18,
		},
	}, nil
}

func (m *mockProvider) Stream(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (any, error) {
	return nil, nil
}

func (m *mockProvider) ListModels(ctx context.Context, cred *core.Credential) ([]core.ModelInfo, error) {
	return []core.ModelInfo{
		{ID: "mock-model", Object: "model", OwnedBy: m.id},
	}, nil
}

func TestEngineChatCompletionsAndAnthropicMessages(t *testing.T) {
	var capturedRecord *core.UsageRecord
	cfg := core.Config{
		Routes: map[string]core.RouteConfig{
			"smart": {
				Targets: []core.Target{
					{Provider: "mock1", Model: "model-alpha"},
				},
			},
		},
		UsageHook: core.UsageHookFunc(func(ctx context.Context, record core.UsageRecord) {
			capturedRecord = &record
		}),
	}

	engine := core.NewEngine(cfg)
	engine.RegisterProvider("mock1", &mockProvider{id: "mock1"})

	// 1. Test /v1/models
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("GET /v1/models status=%d", w.Code)
	}

	// 2. Test /v1/chat/completions (OpenAI format)
	chatReqBody, _ := json.Marshal(map[string]any{
		"model": "smart",
		"messages": []map[string]string{
			{"role": "user", "content": "hi"},
		},
	})
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(chatReqBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("POST /v1/chat/completions status=%d body=%s", w.Code, w.Body.String())
	}
	if capturedRecord == nil || capturedRecord.Provider != "mock1" {
		t.Fatalf("expected usage record captured, got: %+v", capturedRecord)
	}

	// 3. Test /v1/messages (Anthropic Messages API format)
	anthropicReqBody, _ := json.Marshal(map[string]any{
		"model":      "mock1/model-alpha",
		"max_tokens": 1024,
		"messages": []map[string]any{
			{"role": "user", "content": "hello claude"},
		},
	})
	req = httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(anthropicReqBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("POST /v1/messages status=%d body=%s", w.Code, w.Body.String())
	}

	var anthropicResp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &anthropicResp); err != nil {
		t.Fatalf("failed unmarshaling anthropic response: %v", err)
	}
	if anthropicResp["role"] != "assistant" {
		t.Fatalf("expected assistant role in anthropic response, got: %+v", anthropicResp)
	}
}
