package providers

import (
	"context"
	"fmt"
	"net/http"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// AutoConnectResult represents the outcome of self-adjudicating an anonymous provider.
type AutoConnectResult struct {
	RegistryID string   `json:"registry_id"`
	ProviderID string   `json:"provider_id"`
	Status     string   `json:"status"` // "verified", "connected", "failed"
	Models     []string `json:"models,omitempty"`
	ProbeModel string   `json:"probe_model,omitempty"`
	LatencyMs  int64    `json:"latency_ms,omitempty"`
	Error      string   `json:"error,omitempty"`
}

// AutoConnectAnonymousProviders discovers models for all reviewed anonymous providers,
// self-adjudicates each with a minimal completion probe, and returns the live results.
func AutoConnectAnonymousProviders(ctx context.Context, httpClient *http.Client) []AutoConnectResult {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 30 * time.Second}
	}

	profiles := AnonymousProviderProfiles()
	results := make([]AutoConnectResult, 0, len(profiles))

	for _, profile := range profiles {
		if ctx.Err() != nil {
			break
		}

		res := AutoConnectResult{
			RegistryID: profile.RegistryID,
			ProviderID: profile.ProviderID,
		}

		// 1. Discover models
		models, err := DiscoverAnonymousModels(ctx, profile, httpClient)
		if err != nil {
			res.Status = "failed"
			res.Error = fmt.Sprintf("discovery failed: %v", err)
			results = append(results, res)
			continue
		}

		modelIDs := make([]string, 0, len(models))
		for _, m := range models {
			modelIDs = append(modelIDs, m.ID)
		}
		res.Models = modelIDs

		// 2. Select probe model
		probeModel := AnonymousVerificationModel(profile.RegistryID, models)
		if probeModel == "" {
			res.Status = "connected"
			res.Error = "no suitable verification model identified"
			results = append(results, res)
			continue
		}
		res.ProbeModel = probeModel

		// 3. Self-adjudicate with probe completion
		client := NewOpenAIProvider(profile.ProviderID, profile.BaseURL, "none", httpClient)
		start := time.Now()
		_, probeErr := client.Complete(ctx, probeModel, map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": "Reply with: ok"},
			},
			"max_tokens": 16,
		}, &core.Credential{APIKey: "none"})

		res.LatencyMs = time.Since(start).Milliseconds()

		if probeErr != nil {
			res.Status = "connected" // models discovered, but probe timed out or failed
			res.Error = fmt.Sprintf("verification probe failed: %v", probeErr)
		} else {
			res.Status = "verified"
		}

		results = append(results, res)
	}

	return results
}
