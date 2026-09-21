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

// NewAnonymousOpenAICompatibleAdapter adapts a reviewed, keyless
// OpenAI-compatible endpoint to the shared provider connector.
func NewAnonymousOpenAICompatibleAdapter(baseURL string, client *http.Client) core.ProviderAdapter {
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	baseURL = strings.TrimRight(baseURL, "/")
	doJSON := func(ctx context.Context, method, path string, payload any, out any) error {
		var body io.Reader
		if payload != nil {
			encoded, err := json.Marshal(payload)
			if err != nil {
				return fmt.Errorf("marshal provider request: %w", err)
			}
			body = bytes.NewReader(encoded)
		}
		req, err := http.NewRequestWithContext(ctx, method, baseURL+path, body)
		if err != nil {
			return fmt.Errorf("create provider request: %w", err)
		}
		if payload != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, err := client.Do(req)
		if err != nil {
			return core.NewProviderOperationError("provider transport", 0, "", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return core.NewProviderOperationError("provider HTTP request", resp.StatusCode, resp.Header.Get("Retry-After"), nil)
		}
		if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(out); err != nil {
			return core.NewProviderOperationError("provider response decode", resp.StatusCode, "", err)
		}
		return nil
	}

	return core.ProviderAdapter{
		ValidateAuthentication: func(_ context.Context, connection core.ProviderConnection) error {
			if connection.Kind != core.ProviderConnectionAnonymous || connection.AuthKind != core.ProviderAuthAnonymous {
				return fmt.Errorf("anonymous OpenAI-compatible adapter requires anonymous connection")
			}
			return nil
		},
		DiscoverModels: func(ctx context.Context, connection core.ProviderConnection) ([]core.ModelInfo, error) {
			var envelope struct {
				Data []core.ModelInfo `json:"data"`
			}
			if err := doJSON(ctx, http.MethodGet, "/models", nil, &envelope); err != nil {
				return nil, err
			}
			return envelope.Data, nil
		},
		Complete: func(ctx context.Context, _ core.ProviderConnection, target core.Target, payload map[string]any) (map[string]any, error) {
			request := make(map[string]any, len(payload)+2)
			for key, value := range payload {
				request[key] = value
			}
			request["model"] = target.Model
			request["stream"] = false
			var response map[string]any
			if err := doJSON(ctx, http.MethodPost, "/chat/completions", request, &response); err != nil {
				return nil, err
			}
			return response, nil
		},
	}
}
