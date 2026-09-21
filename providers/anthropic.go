package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// AnthropicProvider is a client for the Anthropic Messages API.
type AnthropicProvider struct {
	ID         string
	BaseURL    string
	DefaultKey string
	Client     *http.Client
	Version    string
}

func NewAnthropicProvider(id, baseURL, defaultKey string, client *http.Client) *AnthropicProvider {
	if client == nil {
		client = &http.Client{Timeout: 90 * time.Second}
	}
	if baseURL == "" {
		baseURL = "https://api.anthropic.com"
	}
	baseURL = strings.TrimRight(baseURL, "/")
	return &AnthropicProvider{
		ID:         id,
		BaseURL:    baseURL,
		DefaultKey: defaultKey,
		Client:     client,
		Version:    "2023-06-01",
	}
}

func (p *AnthropicProvider) Complete(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (map[string]any, error) {
	key := p.DefaultKey
	if cred != nil && cred.APIKey != "" {
		key = cred.APIKey
	}

	// Payload is in OpenAI format; translate to Anthropic format if necessary
	var anthropicBody map[string]any
	if _, hasMessages := payload["messages"]; hasMessages {
		// Convert OpenAI chat completion payload to Anthropic messages
		msgs, _ := payload["messages"].([]map[string]any)
		system, convMsgs := translate.OpenAIMessagesToAnthropic(msgs)
		anthropicBody = map[string]any{
			"model":      model,
			"messages":   convMsgs,
			"max_tokens": 4096,
		}
		if system != "" {
			anthropicBody["system"] = system
		}
		if tools, ok := payload["tools"].([]any); ok && len(tools) > 0 {
			anthropicBody["tools"] = translate.OpenAIToolsToAnthropic(tools)
		}
	} else {
		anthropicBody = payload
		anthropicBody["model"] = model
	}
	anthropicBody["stream"] = false

	jsonBytes, err := json.Marshal(anthropicBody)
	if err != nil {
		return nil, fmt.Errorf("marshal anthropic request: %w", err)
	}

	url := p.BaseURL + "/v1/messages"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("create anthropic request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", key)
	req.Header.Set("anthropic-version", p.Version)

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, invocationError(ctx, "anthropic request failed", 0, err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, invocationError(ctx, "read anthropic response failed", 0, err)
	}

	if resp.StatusCode >= 400 {
		return nil, invocationError(ctx, fmt.Sprintf("anthropic HTTP %d: %s", resp.StatusCode, string(bodyBytes)), resp.StatusCode, nil)
	}

	var anthropicResp map[string]any
	if err := json.Unmarshal(bodyBytes, &anthropicResp); err != nil {
		return nil, fmt.Errorf("unmarshal anthropic response: %w", err)
	}

	// Translate Anthropic response back to OpenAI format
	return translate.AnthropicResponseToOpenAI(anthropicResp, model), nil
}

func (p *AnthropicProvider) Stream(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (StreamIter, error) {
	key := p.DefaultKey
	if cred != nil && cred.APIKey != "" {
		key = cred.APIKey
	}

	var anthropicBody map[string]any
	if msgs, ok := payload["messages"].([]map[string]any); ok {
		system, convMsgs := translate.OpenAIMessagesToAnthropic(msgs)
		anthropicBody = map[string]any{
			"model":      model,
			"messages":   convMsgs,
			"max_tokens": 4096,
		}
		if system != "" {
			anthropicBody["system"] = system
		}
		if tools, ok := payload["tools"].([]any); ok && len(tools) > 0 {
			anthropicBody["tools"] = translate.OpenAIToolsToAnthropic(tools)
		}
	} else {
		anthropicBody = payload
		anthropicBody["model"] = model
	}
	anthropicBody["stream"] = true

	jsonBytes, err := json.Marshal(anthropicBody)
	if err != nil {
		return nil, fmt.Errorf("marshal anthropic request: %w", err)
	}

	url := p.BaseURL + "/v1/messages"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("create anthropic request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("x-api-key", key)
	req.Header.Set("anthropic-version", p.Version)

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, invocationError(ctx, "anthropic stream request failed", 0, err)
	}

	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return nil, invocationError(ctx, fmt.Sprintf("anthropic HTTP %d: %s", resp.StatusCode, string(b)), resp.StatusCode, nil)
	}

	return newByteStreamIter(ctx, resp.Body), nil
}

func (p *AnthropicProvider) ListModels(ctx context.Context, cred *core.Credential) ([]core.ModelInfo, error) {
	// Standard curated models when listing Anthropic models
	return []core.ModelInfo{
		{ID: "claude-3-7-sonnet", Object: "model", OwnedBy: p.ID, Description: "Claude 3.7 Sonnet"},
		{ID: "claude-3-5-sonnet", Object: "model", OwnedBy: p.ID, Description: "Claude 3.5 Sonnet"},
		{ID: "claude-3-5-haiku", Object: "model", OwnedBy: p.ID, Description: "Claude 3.5 Haiku"},
		{ID: "claude-3-opus", Object: "model", OwnedBy: p.ID, Description: "Claude 3 Opus"},
	}, nil
}
