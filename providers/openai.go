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

	core "github.com/xibodev/llmgw-core"
)

// OpenAIProvider is a universal client for OpenAI and OpenAI-compatible APIs.
type OpenAIProvider struct {
	ID         string
	BaseURL    string
	DefaultKey string
	Client     *http.Client
	Headers    map[string]string
}

func NewOpenAIProvider(id, baseURL, defaultKey string, client *http.Client) *OpenAIProvider {
	if client == nil {
		client = &http.Client{Timeout: 90 * time.Second}
	}
	baseURL = strings.TrimRight(baseURL, "/")
	return &OpenAIProvider{
		ID:         id,
		BaseURL:    baseURL,
		DefaultKey: defaultKey,
		Client:     client,
		Headers:    make(map[string]string),
	}
}

func (p *OpenAIProvider) resolveAuth(cred *core.Credential) (string, map[string]string) {
	key := p.DefaultKey
	extraHeaders := make(map[string]string)

	for k, v := range p.Headers {
		extraHeaders[k] = v
	}

	if cred != nil {
		if cred.APIKey != "" {
			key = cred.APIKey
		} else if cred.Token != "" {
			key = cred.Token
		}
		for k, v := range cred.Headers {
			extraHeaders[k] = v
		}
	}

	return key, extraHeaders
}

func (p *OpenAIProvider) Complete(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (map[string]any, error) {
	reqBody := make(map[string]any, len(payload)+2)
	for k, v := range payload {
		reqBody[k] = v
	}
	reqBody["model"] = model
	reqBody["stream"] = false

	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := p.BaseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	key, extraHeaders := p.resolveAuth(cred)
	if key != "" && !strings.EqualFold(key, "none") && !strings.EqualFold(key, "free") {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	for k, v := range extraHeaders {
		req.Header.Set(k, v)
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("upstream request failed: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read upstream response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("upstream HTTP %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result map[string]any
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return nil, fmt.Errorf("unmarshal upstream response: %w", err)
	}

	return result, nil
}

func (p *OpenAIProvider) Stream(ctx context.Context, model string, payload map[string]any, cred *core.Credential) (StreamIter, error) {
	reqBody := make(map[string]any, len(payload)+2)
	for k, v := range payload {
		reqBody[k] = v
	}
	reqBody["model"] = model
	reqBody["stream"] = true

	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := p.BaseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	key, extraHeaders := p.resolveAuth(cred)
	if key != "" && !strings.EqualFold(key, "none") && !strings.EqualFold(key, "free") {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	for k, v := range extraHeaders {
		req.Header.Set(k, v)
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("upstream request failed: %w", err)
	}

	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		errBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("upstream HTTP %d: %s", resp.StatusCode, string(errBytes))
	}

	return NewByteStreamIter(resp.Body), nil
}

func (p *OpenAIProvider) ListModels(ctx context.Context, cred *core.Credential) ([]core.ModelInfo, error) {
	url := p.BaseURL + "/models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	key, extraHeaders := p.resolveAuth(cred)
	if key != "" && !strings.EqualFold(key, "none") && !strings.EqualFold(key, "free") {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	for k, v := range extraHeaders {
		req.Header.Set(k, v)
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("query models failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("upstream HTTP %d: %s", resp.StatusCode, string(b))
	}

	var envelope struct {
		Data []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return nil, fmt.Errorf("decode models: %w", err)
	}

	models := make([]core.ModelInfo, 0, len(envelope.Data))
	for _, m := range envelope.Data {
		models = append(models, core.ModelInfo{
			ID:      m.ID,
			Object:  m.Object,
			Created: m.Created,
			OwnedBy: p.ID,
		})
	}

	return models, nil
}
