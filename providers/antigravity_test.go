package providers

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
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
				"model-z": map[string]any{"displayName": "Model Z", "supportsThinking": false},
				"model-a": map[string]any{
					"displayName": "Model A", "supportsImages": true, "supportsThinking": true,
					"supportsVideo": true, "supportedMimeTypes": map[string]any{"image/png": true, "video/mp4": true},
				},
			}, "imageGenerationModelIds": []string{"model-z"}})
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
		capabilities.Tools != core.SupportUnknown || capabilities.Reasoning != core.SupportSupported ||
		capabilities.Inputs.Image != core.SupportUnknown || capabilities.Operations.Image != core.SupportUnsupported ||
		capabilities.Operations.AudioIn != core.SupportUnknown || !reflect.DeepEqual(models[0].SupportedAPIs, []string{"/v1/chat/completions"}) ||
		capabilities.Streaming != core.SupportUnknown ||
		capabilities.Provenance.Source != core.ModelCapabilitySourceInferred ||
		capabilities.Provenance.Confidence != core.ModelCapabilityConfidenceMedium {
		t.Fatalf("capabilities = %#v", capabilities)
	}
	if models[1].Capabilities.Reasoning != core.SupportUnsupported ||
		models[1].Capabilities.Operations.Image != core.SupportSupported ||
		models[1].Capabilities.Inputs.Image != core.SupportUnknown ||
		!reflect.DeepEqual(models[1].SupportedAPIs, []string{"/v1/chat/completions", "/v1/images/generations"}) ||
		models[1].Capabilities.Provenance.Source != core.ModelCapabilitySourceInferred ||
		models[1].Capabilities.Provenance.Confidence != core.ModelCapabilityConfidenceMedium {
		t.Fatalf("model-z capabilities = %#v", models[1].Capabilities)
	}
}

func TestAntigravityCatalogLeavesAbsentCapabilityMetadataUnknown(t *testing.T) {
	capabilities := antigravityModelCapabilities(antigravityCatalogModel{}, core.SupportUnknown, core.SupportUnknown)
	if capabilities.Reasoning != core.SupportUnknown || capabilities.Tools != core.SupportUnknown ||
		capabilities.Inputs.Image != core.SupportUnknown || capabilities.Operations.Image != core.SupportUnknown ||
		capabilities.Operations.AudioIn != core.SupportUnknown || capabilities.Operations.Video != core.SupportUnknown {
		t.Fatalf("capabilities = %#v", capabilities)
	}
}

func TestAntigravityCatalogMetadataFixture(t *testing.T) {
	raw, err := os.ReadFile("testdata/antigravity_catalog_metadata.json")
	if err != nil {
		t.Fatal(err)
	}
	var catalog antigravityCatalogResponse
	if err := json.Unmarshal(raw, &catalog); err != nil {
		t.Fatal(err)
	}
	if len(catalog.Models) != 5 || catalog.ImageGenerationModelIDs == nil || !slices.Contains(*catalog.ImageGenerationModelIDs, "gemini-3.1-flash-image") ||
		catalog.AudioTranscriptionModelIDs == nil ||
		!slices.Contains(catalog.TabModelIDs, "chat_20706") ||
		!slices.Contains(catalog.TieredModelIDs.Flash, "gemini-3.8-flash-tiered") {
		t.Fatalf("catalog metadata = %#v", catalog)
	}
	claude := catalog.Models["claude-sonnet-4-6"]
	if claude.SupportsImages == nil || !*claude.SupportsImages || claude.SupportsThinking == nil || !*claude.SupportsThinking ||
		!claude.SupportedMimeTypes["image/png"] {
		t.Fatalf("claude metadata = %#v", claude)
	}
	image := catalog.Models["gemini-3.1-flash-image"]
	if image.SupportsImages != nil || image.SupportsThinking != nil || len(image.SupportedMimeTypes) != 0 {
		t.Fatalf("absent image-model metadata became known = %#v", image)
	}

	imageCapabilities := antigravityModelCapabilities(image,
		rosterSupport(catalog.ImageGenerationModelIDs, "gemini-3.1-flash-image"),
		rosterSupport(catalog.AudioTranscriptionModelIDs, "gemini-3.1-flash-image"))
	if imageCapabilities.Operations.Image != core.SupportSupported || imageCapabilities.Operations.AudioIn != core.SupportUnsupported {
		t.Fatalf("image capabilities = %#v", imageCapabilities)
	}
	chatCapabilities := antigravityModelCapabilities(claude,
		rosterSupport(catalog.ImageGenerationModelIDs, "claude-sonnet-4-6"),
		rosterSupport(catalog.AudioTranscriptionModelIDs, "claude-sonnet-4-6"))
	if chatCapabilities.Operations.Image != core.SupportUnsupported || chatCapabilities.Operations.AudioIn != core.SupportUnsupported ||
		chatCapabilities.Inputs.Image != core.SupportUnknown || chatCapabilities.Operations.Video != core.SupportUnknown {
		t.Fatalf("chat capabilities = %#v", chatCapabilities)
	}
}

func TestAntigravityCatalogAbsentRostersRemainUnknown(t *testing.T) {
	raw, err := os.ReadFile("testdata/antigravity_catalog_absent_rosters.json")
	if err != nil {
		t.Fatal(err)
	}
	var catalog antigravityCatalogResponse
	if err := json.Unmarshal(raw, &catalog); err != nil {
		t.Fatal(err)
	}
	if catalog.ImageGenerationModelIDs != nil || catalog.AudioTranscriptionModelIDs != nil {
		t.Fatalf("absent rosters became present = %#v", catalog)
	}
	metadata := catalog.Models["model"]
	capabilities := antigravityModelCapabilities(metadata,
		rosterSupport(catalog.ImageGenerationModelIDs, "model"),
		rosterSupport(catalog.AudioTranscriptionModelIDs, "model"))
	if capabilities.Operations.Image != core.SupportUnknown || capabilities.Operations.AudioIn != core.SupportUnknown ||
		capabilities.Inputs.Image != core.SupportUnknown || capabilities.Operations.Video != core.SupportUnknown {
		t.Fatalf("capabilities = %#v", capabilities)
	}
}

func TestExperimentalAntigravityCompleteRecoversOneUnauthorizedResponse(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := requests.Add(1)
		if call == 1 {
			if r.Header.Get("Authorization") != "Bearer old-token" {
				t.Errorf("first authorization = %q", r.Header.Get("Authorization"))
			}
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if r.Header.Get("Authorization") != "Bearer new-token" {
			t.Errorf("retry authorization = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"recovered\"}]}}]}}\n\n"))
	}))
	defer server.Close()

	var token atomic.Value
	token.Store("old-token")
	provider := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return token.Load().(string), "project-id", nil
	}, server.Client(), server.URL)
	var recoveries atomic.Int32
	provider.SetUnauthorizedHandler(func(_ context.Context, rejected string) error {
		if rejected != "old-token" {
			t.Fatalf("rejected token = %q", rejected)
		}
		recoveries.Add(1)
		token.Store("new-token")
		return nil
	})
	response, err := provider.Complete(context.Background(), "model", map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hello"}},
	}, nil)
	if err != nil || response == nil || requests.Load() != 2 || recoveries.Load() != 1 {
		t.Fatalf("response=%v err=%v requests=%d recoveries=%d", response, err, requests.Load(), recoveries.Load())
	}
}

func TestExperimentalAntigravityCatalogRecoversUnauthorizedOnlyOnce(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	provider := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return "rejected-token", "project-id", nil
	}, server.Client(), server.URL)
	var recoveries atomic.Int32
	provider.SetUnauthorizedHandler(func(context.Context, string) error {
		recoveries.Add(1)
		return nil
	})
	_, err := provider.ListModels(context.Background(), nil)
	var operationError *core.ProviderOperationError
	if !errors.As(err, &operationError) || operationError.Failure.StatusCode != http.StatusUnauthorized ||
		requests.Load() != 2 || recoveries.Load() != 1 {
		t.Fatalf("err=%v requests=%d recoveries=%d", err, requests.Load(), recoveries.Load())
	}
}

func TestExperimentalAntigravityReportsDiscoveredProject(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1internal:loadCodeAssist":
			_, _ = w.Write([]byte(`{"cloudaicompanionProject":"discovered-project"}`))
		case "/v1internal:fetchAvailableModels":
			_, _ = w.Write([]byte(`{"models":{}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	provider := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return "access-token", "", nil
	}, server.Client(), server.URL)
	var observedToken, observedProject string
	provider.SetProjectObserver(func(_ context.Context, token, project string) error {
		observedToken, observedProject = token, project
		return nil
	})
	if _, err := provider.ListModels(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	if observedToken != "access-token" || observedProject != "discovered-project" {
		t.Fatalf("observation = %q, %q", observedToken, observedProject)
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

func TestExperimentalAntigravityGenerateImagesUsesExplicitRosterCapability(t *testing.T) {
	png := []byte("fixture-png")
	fixture, err := os.ReadFile("testdata/antigravity_image_generation.sse")
	if err != nil {
		t.Fatal(err)
	}
	var generatedRequest map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1internal:fetchAvailableModels":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"models": map[string]any{
					"image-model": map[string]any{"displayName": "Image"},
					"chat-model":  map[string]any{"displayName": "Chat", "supportsImages": true},
				},
				"imageGenerationModelIds": []string{"image-model", "not-in-root-roster"},
			})
		case "/v1internal:streamGenerateContent":
			var envelope map[string]any
			if err := json.NewDecoder(r.Body).Decode(&envelope); err != nil {
				t.Fatal(err)
			}
			generatedRequest = envelope["request"].(map[string]any)
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write(fixture)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	provider := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return "access-token", "project", nil
	}, server.Client(), server.URL)
	models, err := provider.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if models[0].ID != "chat-model" || models[0].Capabilities.Operations.Image != core.SupportUnsupported || !reflect.DeepEqual(models[0].SupportedAPIs, []string{"/v1/chat/completions"}) ||
		models[1].ID != "image-model" || models[1].Capabilities.Operations.Image != core.SupportSupported ||
		!reflect.DeepEqual(models[1].SupportedAPIs, []string{"/v1/chat/completions", "/v1/images/generations"}) ||
		models[1].Capabilities.Provenance.Source != core.ModelCapabilitySourceInferred {
		t.Fatalf("models = %#v", models)
	}

	result, err := provider.GenerateImages(context.Background(), core.GenerateImagesRequest{
		Model: "image-model", Prompt: "draw a small test image", Count: 1,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 || !reflect.DeepEqual(result.Images[0].Data, png) || result.Images[0].MimeType != "image/png" ||
		result.Usage["totalTokenCount"] != float64(7) {
		t.Fatalf("result = %#v", result)
	}
	config := generatedRequest["generationConfig"].(map[string]any)
	if !reflect.DeepEqual(config["responseModalities"], []any{"TEXT", "IMAGE"}) {
		t.Fatalf("generationConfig = %#v", config)
	}
	contents := generatedRequest["contents"].([]any)
	part := contents[0].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if part["text"] != "draw a small test image" {
		t.Fatalf("contents = %#v", contents)
	}
}

func TestExperimentalAntigravityGenerateImagesRejectsUnsupportedCount(t *testing.T) {
	provider := NewExperimentalAntigravityProvider(nil, nil, "")
	for _, count := range []int{-1, 2} {
		_, err := provider.GenerateImages(context.Background(), core.GenerateImagesRequest{Model: "image-model", Prompt: "draw", Count: count}, nil)
		var operationError *core.ProviderOperationError
		if !errors.As(err, &operationError) || operationError.Failure.StatusCode != http.StatusBadRequest {
			t.Fatalf("count=%d error = %T %v", count, err, err)
		}
	}
}

func TestExperimentalAntigravityGenerateImagesRejectsModelsOutsideExplicitIntersection(t *testing.T) {
	var generations atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1internal:fetchAvailableModels" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"models":                  map[string]any{"chat-model": map[string]any{}},
				"imageGenerationModelIds": []string{"not-in-root-roster"},
			})
			return
		}
		generations.Add(1)
		http.NotFound(w, r)
	}))
	defer server.Close()
	provider := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return "access-token", "project", nil
	}, server.Client(), server.URL)

	_, err := provider.GenerateImages(context.Background(), core.GenerateImagesRequest{Model: "chat-model", Prompt: "draw", Count: 1}, nil)
	var operationError *core.ProviderOperationError
	if !errors.As(err, &operationError) || operationError.Failure.StatusCode != http.StatusBadRequest || generations.Load() != 0 {
		t.Fatalf("err=%v generations=%d", err, generations.Load())
	}
	_, err = provider.GenerateImages(context.Background(), core.GenerateImagesRequest{Model: "not-in-root-roster", Prompt: "draw", Count: 1}, nil)
	if !errors.As(err, &operationError) || operationError.Failure.StatusCode != http.StatusBadRequest || generations.Load() != 0 {
		t.Fatalf("err=%v generations=%d", err, generations.Load())
	}
}

func TestParseAntigravityImageSSERejectsInvalidAndOversizedInlineData(t *testing.T) {
	tests := []struct {
		name string
		data string
	}{
		{name: "invalid base64", data: "%%%"},
		{name: "oversized", data: strings.Repeat("A", base64.StdEncoding.EncodedLen(antigravityMaxImageBytes+1))},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			event, err := json.Marshal(map[string]any{"response": map[string]any{"candidates": []any{map[string]any{"content": map[string]any{"parts": []any{map[string]any{"inline_data": map[string]any{"mime_type": "image/png", "data": test.data}}}}}}}})
			if err != nil {
				t.Fatal(err)
			}
			_, err = parseAntigravityImageSSE(strings.NewReader("data: "+string(event)+"\n\n"), 1)
			var operationError *core.ProviderOperationError
			if !errors.As(err, &operationError) {
				t.Fatalf("error = %T %v", err, err)
			}
		})
	}
}

func TestParseAntigravityImageSSERejectsActiveContentMimeType(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte(`<svg xmlns="http://www.w3.org/2000/svg"/>`))
	event, err := json.Marshal(map[string]any{"response": map[string]any{"candidates": []any{map[string]any{"content": map[string]any{"parts": []any{map[string]any{"inlineData": map[string]any{"mimeType": "image/svg+xml", "data": encoded}}}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = parseAntigravityImageSSE(strings.NewReader("data: "+string(event)+"\n\n"), 1)
	var operationError *core.ProviderOperationError
	if !errors.As(err, &operationError) {
		t.Fatalf("error = %T %v", err, err)
	}
}

func TestExperimentalAntigravityRejectsOversizedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, strings.Repeat(" ", antigravityMaxBodyBytes+1))
	}))
	defer server.Close()
	provider := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return "access-token", "project", nil
	}, server.Client(), server.URL)
	_, err := provider.ListModels(context.Background(), nil)
	var operationError *core.ProviderOperationError
	if !errors.As(err, &operationError) || operationError.Failure.Err == nil || !strings.Contains(operationError.Failure.Err.Error(), "size limit") {
		t.Fatalf("error = %v", err)
	}
}

func TestExperimentalAntigravityGenerateImagesRecoversUnauthorizedAndObservesProject(t *testing.T) {
	var token atomic.Value
	token.Store("old-token")
	var generations, recoveries, observations atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1internal:loadCodeAssist":
			_ = json.NewEncoder(w).Encode(map[string]any{"cloudaicompanionProject": "project"})
		case "/v1internal:fetchAvailableModels":
			_ = json.NewEncoder(w).Encode(map[string]any{"models": map[string]any{"image-model": map[string]any{}}, "imageGenerationModelIds": []string{"image-model"}})
		case "/v1internal:streamGenerateContent":
			generations.Add(1)
			if r.Header.Get("Authorization") == "Bearer old-token" {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			encoded := base64.StdEncoding.EncodeToString([]byte("image"))
			_, _ = fmt.Fprintf(w, "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":%q}}]}}]}}\n\n", encoded)
		}
	}))
	defer server.Close()
	provider := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return token.Load().(string), "", nil
	}, server.Client(), server.URL)
	provider.SetProjectObserver(func(context.Context, string, string) error {
		observations.Add(1)
		return nil
	})
	provider.SetUnauthorizedHandler(func(context.Context, string) error {
		recoveries.Add(1)
		token.Store("new-token")
		return nil
	})
	result, err := provider.GenerateImages(context.Background(), core.GenerateImagesRequest{Model: "image-model", Prompt: "draw"}, nil)
	if err != nil || len(result.Images) != 1 || generations.Load() != 2 || recoveries.Load() != 1 || observations.Load() != 2 {
		t.Fatalf("result=%#v err=%v generations=%d recoveries=%d observations=%d", result, err, generations.Load(), recoveries.Load(), observations.Load())
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
