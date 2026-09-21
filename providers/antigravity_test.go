package providers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

func TestExperimentalAntigravityListModelsDiscoversProjectAndExactCatalog(t *testing.T) {
	var paths []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if r.Header.Get("Authorization") != "Bearer access-token" {
			t.Fatalf("unexpected authorization header")
		}
		switch r.URL.Path {
		case "/v1internal:loadCodeAssist":
			_ = json.NewEncoder(w).Encode(map[string]any{"cloudaicompanionProject": "discovered-project"})
		case "/v1internal:fetchAvailableModels":
			var request map[string]any
			_ = json.NewDecoder(r.Body).Decode(&request)
			if request["project"] != "discovered-project" {
				t.Fatalf("project = %#v", request["project"])
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"models": map[string]any{
				"model-z": map[string]any{"displayName": "Model Z"},
				"model-a": map[string]any{"displayName": "Model A"},
			}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	provider := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return "access-token", "", nil
	}, server.Client(), server.URL)
	models, err := provider.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := []string{models[0].ID, models[1].ID}; !reflect.DeepEqual(got, []string{"model-a", "model-z"}) {
		t.Fatalf("models = %#v", got)
	}
	if !reflect.DeepEqual(paths, []string{"/v1internal:loadCodeAssist", "/v1internal:fetchAvailableModels"}) {
		t.Fatalf("paths = %#v", paths)
	}
	capabilities := models[0].Capabilities
	if capabilities == nil || capabilities.Operations.Chat != core.SupportSupported ||
		capabilities.Tools != core.SupportSupported || capabilities.Reasoning != core.SupportSupported ||
		capabilities.Streaming != core.SupportUnknown {
		t.Fatalf("capabilities = %#v", capabilities)
	}
}

func TestExperimentalAntigravityCompleteMapsOpenAIRequestAndBufferedSSE(t *testing.T) {
	var inner map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1internal:streamGenerateContent" || r.URL.Query().Get("alt") != "sse" {
			http.NotFound(w, r)
			return
		}
		var envelope map[string]any
		if err := json.NewDecoder(r.Body).Decode(&envelope); err != nil {
			t.Fatal(err)
		}
		inner = envelope["request"].(map[string]any)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"think\",\"thought\":true},{\"text\":\"ans\"}]}}]}}\n\n"))
		_, _ = w.Write([]byte("data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ing\",\"thought\":true},{\"text\":\"wer\"},{\"functionCall\":{\"name\":\"lookup\",\"args\":{\"q\":\"x\"}},\"thought_signature\":\"signature\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":4,\"totalTokenCount\":7}}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	provider := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return "access-token", "project-id", nil
	}, server.Client(), server.URL)
	response, err := provider.Complete(context.Background(), "upstream-model", map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "be concise"},
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "question"}}},
			map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{
				"id": "call-old", "type": "function", "function": map[string]any{"name": "lookup", "arguments": `{"q":"old"}`, "thought_signature": "old-signature"},
			}}},
			map[string]any{"role": "tool", "tool_call_id": "call-old", "content": "old result"},
		},
		"tools": []any{map[string]any{"type": "function", "function": map[string]any{
			"name": "lookup", "description": "look up", "parameters": map[string]any{"type": "object"},
		}}},
		"max_tokens":  128,
		"temperature": 0.0,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}

	config := inner["generationConfig"].(map[string]any)
	if config["maxOutputTokens"] != float64(128) || config["temperature"] != float64(0) {
		t.Fatalf("generation config = %#v", config)
	}
	system := inner["systemInstruction"].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if system["text"] != "be concise" {
		t.Fatalf("system instruction = %#v", system)
	}
	contents := inner["contents"].([]any)
	storedCall := contents[1].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if storedCall["thoughtSignature"] != "old-signature" {
		t.Fatalf("stored tool call = %#v", storedCall)
	}
	toolResult := contents[2].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if toolResult["name"] != "lookup" {
		t.Fatalf("tool result = %#v", toolResult)
	}
	declaration := inner["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	if declaration["name"] != "lookup" {
		t.Fatalf("tool declaration = %#v", declaration)
	}

	choice := response["choices"].([]any)[0].(map[string]any)
	message := choice["message"].(map[string]any)
	if message["content"] != "answer" || message["reasoning_content"] != "thinking" || choice["finish_reason"] != "tool_calls" {
		t.Fatalf("choice = %#v", choice)
	}
	call := message["tool_calls"].([]any)[0].(map[string]any)
	function := call["function"].(map[string]any)
	if function["name"] != "lookup" || function["arguments"] != `{"q":"x"}` || function["thought_signature"] != "signature" {
		t.Fatalf("tool call = %#v", call)
	}
	usage := response["usage"].(map[string]any)
	if usage["total_tokens"] != 7 {
		t.Fatalf("usage = %#v", usage)
	}
}

func TestExperimentalAntigravityPreservesHTTPFailureWithoutResponseBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Retry-After", "17")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte("secret response body"))
	}))
	defer server.Close()

	provider := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return "access-token", "project-id", nil
	}, server.Client(), server.URL)
	_, err := provider.ListModels(context.Background(), nil)
	var operationError *core.ProviderOperationError
	if !errors.As(err, &operationError) {
		t.Fatalf("error = %T %v", err, err)
	}
	if operationError.Failure.StatusCode != http.StatusTooManyRequests || operationError.Failure.RetryAfter != "17" {
		t.Fatalf("failure = %#v", operationError.Failure)
	}
	if strings.Contains(err.Error(), "secret") {
		t.Fatalf("error exposed response body: %v", err)
	}
}

func TestExperimentalAntigravityStreamIsExplicitlyUnsupported(t *testing.T) {
	provider := NewExperimentalAntigravityProvider(nil, nil, "")
	_, err := provider.Stream(context.Background(), "model", nil, nil)
	if !errors.Is(err, ErrExperimentalAntigravityStreamingUnsupported) {
		t.Fatalf("error = %v", err)
	}
}
