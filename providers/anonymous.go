package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

type AnonymousProviderProfile struct {
	RegistryID         string   `json:"registry_id"`
	ProviderID         string   `json:"provider_id"`
	RuntimeType        string   `json:"runtime_type"`
	BaseURL            string   `json:"base_url"`
	VerificationModels []string `json:"verification_models"`
}

var anonymousVerificationModels = map[string][]string{
	"opencode_zen":     {"ling-3.0-flash-fin-free", "muse-spark-1.2-contributor-free", "nemotron-3.5-lightning-free"},
	"kilo_code":        {"kilo-auto/free", "liquid/lfm-2.5-2.6b:free", "cohere/north-mini-code:free"},
	"llm7":             {"codestral-latest", "mistral-Nemo-Instruct-2407", "minimax-m2.7"},
	"ovh_ai_endpoints": {"Qwen3.8-27B", "Mistral-Nemo-Instruct-2407", "gpt-oss-20b"},
	"pollinations":     {"openai-fast"},
}

// AnonymousProviderProfiles returns all curated providers configured for anonymous automation.
func AnonymousProviderProfiles() []AnonymousProviderProfile {
	profiles := []AnonymousProviderProfile{}
	for _, entry := range ProviderRegistry() {
		if !entry.AnonymousAutomation {
			continue
		}
		profiles = append(profiles, AnonymousProviderProfile{
			RegistryID: entry.ID, ProviderID: entry.DefaultProviderID,
			RuntimeType: entry.RuntimeType, BaseURL: entry.DefaultBaseURL,
			VerificationModels: append([]string(nil), anonymousVerificationModels[entry.ID]...),
		})
	}
	sort.Slice(profiles, func(i, j int) bool { return profiles[i].RegistryID < profiles[j].RegistryID })
	return profiles
}

// AnonymousVerificationModel returns the best probe model for self-adjudication.
func AnonymousVerificationModel(registryID string, models []core.ModelInfo) string {
	free := map[string]string{}
	for _, m := range models {
		free[strings.ToLower(m.ID)] = m.ID
	}
	for _, preferred := range anonymousVerificationModels[registryID] {
		if model := free[strings.ToLower(preferred)]; model != "" {
			return model
		}
	}
	ids := make([]string, 0, len(free))
	for _, id := range free {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if len(ids) > 0 {
		return ids[0]
	}
	return ""
}

// DiscoverAnonymousModels queries an anonymous provider's catalog endpoint and returns
// normalized, filtered free models.
func DiscoverAnonymousModels(ctx context.Context, profile AnonymousProviderProfile, client *http.Client) ([]core.ModelInfo, error) {
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	reqURL := strings.TrimRight(profile.BaseURL, "/") + "/models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create discovery request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("query discovery endpoint %s: %w", reqURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("discovery status %d: %s", resp.StatusCode, string(b))
	}

	bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, 10<<20))
	if err != nil {
		return nil, fmt.Errorf("read discovery response: %w", err)
	}

	if profile.RegistryID == "pollinations" {
		return parsePollinationsCatalog(bodyBytes, profile.ProviderID)
	}

	var envelope struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(bodyBytes, &envelope); err == nil && len(envelope.Data) > 0 {
		return filterAnonymousCatalogRows(envelope.Data, profile.RegistryID, profile.ProviderID), nil
	}

	var listEnvelope []map[string]any
	if err := json.Unmarshal(bodyBytes, &listEnvelope); err == nil && len(listEnvelope) > 0 {
		return filterAnonymousCatalogRows(listEnvelope, profile.RegistryID, profile.ProviderID), nil
	}

	return nil, fmt.Errorf("unrecognized model envelope format from %s", profile.BaseURL)
}

func parsePollinationsCatalog(data []byte, providerID string) ([]core.ModelInfo, error) {
	var items []map[string]any
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, fmt.Errorf("unmarshal pollinations catalog: %w", err)
	}
	rows := []core.ModelInfo{}
	for _, item := range items {
		name, ok := item["name"].(string)
		if !ok || name == "" {
			continue
		}
		tier, _ := item["tier"].(string)
		if !strings.EqualFold(tier, "anonymous") {
			continue
		}
		desc, _ := item["description"].(string)
		rows = append(rows, core.ModelInfo{
			ID:          name,
			Object:      "model",
			OwnedBy:     providerID,
			Description: desc,
		})
	}
	return rows, nil
}

func filterAnonymousCatalogRows(items []map[string]any, registryID, providerID string) []core.ModelInfo {
	out := []core.ModelInfo{}
	for _, raw := range items {
		id, _ := raw["id"].(string)
		if id == "" {
			id, _ = raw["name"].(string)
		}
		if id == "" {
			continue
		}

		include := false
		switch registryID {
		case "kilo_code":
			include, _ = raw["isFree"].(bool)
			if pricing, ok := raw["pricing"].(map[string]any); ok {
				include = include && isZeroPrice(pricing["prompt"]) && isZeroPrice(pricing["completion"])
			}
		case "llm7":
			tier, _ := raw["tier"].(string)
			modelType, _ := raw["model_type"].(string)
			usageBasedOnly, _ := raw["usage_based_only"].(bool)
			include = strings.EqualFold(tier, "turbo") && strings.EqualFold(modelType, "chat") && !usageBasedOnly
		case "ovh_ai_endpoints":
			if pricing, ok := raw["pricing"].(map[string]any); ok {
				include = isZeroPrice(pricing["prompt"]) && isZeroPrice(pricing["completion"])
			}
		case "opencode_zen":
			include = true
		default:
			include = true
		}

		if include {
			out = append(out, core.ModelInfo{
				ID:      id,
				Object:  "model",
				OwnedBy: providerID,
			})
		}
	}
	return out
}

func isZeroPrice(val any) bool {
	switch v := val.(type) {
	case string:
		num, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		return err == nil && num == 0 && !math.IsNaN(num) && !math.IsInf(num, 0)
	case float64:
		return v == 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
	default:
		return false
	}
}
