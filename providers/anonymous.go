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

// AnonymousProviderProfile describes one provider curated for anonymous
// automation. Whether a product enrolls it remains the product's policy.
type AnonymousProviderProfile struct {
	RegistryID         string   `json:"registry_id"`
	ProviderID         string   `json:"provider_id"`
	RuntimeType        string   `json:"runtime_type"`
	BaseURL            string   `json:"base_url"`
	VerificationModels []string `json:"verification_models"`
}

// anonymousVerificationDefaults returns the reviewed probe models for one
// anonymous provider, most preferred first. It is a function rather than a
// map variable so the package keeps no mutable state.
func anonymousVerificationDefaults(registryID string) []string {
	switch registryID {
	case "kilo_code":
		return []string{"kilo-auto/free", "liquid/lfm-2.5-2.6b:free", "cohere/north-mini-code:free"}
	case "llm7":
		return []string{"codestral-latest", "mistral-Nemo-Instruct-2407", "minimax-m2.7"}
	case "ovh_ai_endpoints":
		return []string{"Qwen3.8-27B", "Mistral-Nemo-Instruct-2407", "gpt-oss-20b"}
	case "pollinations":
		return []string{"openai-fast"}
	}
	return nil
}

// AnonymousProviderProfiles returns the default registry's anonymous profiles.
func AnonymousProviderProfiles() []AnonymousProviderProfile {
	return DefaultRegistry().AnonymousProfiles()
}

// AnonymousProfiles returns a profile for every entry curated for anonymous
// automation, sorted by registry id.
func (r *Registry) AnonymousProfiles() []AnonymousProviderProfile {
	profiles := []AnonymousProviderProfile{}
	for _, entry := range r.entries {
		if !entry.AnonymousAutomation {
			continue
		}
		profiles = append(profiles, AnonymousProviderProfile{
			RegistryID: entry.ID, ProviderID: entry.DefaultProviderID,
			RuntimeType: entry.RuntimeType, BaseURL: entry.DefaultBaseURL,
			VerificationModels: anonymousVerificationDefaults(entry.ID),
		})
	}
	sort.Slice(profiles, func(i, j int) bool { return profiles[i].RegistryID < profiles[j].RegistryID })
	return profiles
}

// SelectVerificationModel returns the model to probe an anonymous provider
// with: the first reviewed default among the free model ids, otherwise the
// lexically first free id, otherwise "". Matching ignores case.
func SelectVerificationModel(registryID string, freeModelIDs []string) string {
	free := make(map[string]string, len(freeModelIDs))
	for _, id := range freeModelIDs {
		free[strings.ToLower(id)] = id
	}
	for _, preferred := range anonymousVerificationDefaults(registryID) {
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

// AnonymousVerificationModel returns the probe model among discovered free
// models; see SelectVerificationModel.
func AnonymousVerificationModel(registryID string, models []core.ModelInfo) string {
	ids := make([]string, 0, len(models))
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	return SelectVerificationModel(registryID, ids)
}

// AnonymousAdmission is the verdict on one raw catalog row of an anonymous
// provider.
type AnonymousAdmission struct {
	// Free reports a free model that is usable without credentials.
	Free bool
	// ContextWindow is the declared context length, or 0.
	ContextWindow int
	// Reasoning and ToolCalls report capabilities the row declares.
	Reasoning bool
	ToolCalls bool
}

// AdmitAnonymousModel applies the reviewed free-model rules to one raw row of
// an anonymous provider's catalog. Admission fails closed: an unknown
// provider is never admitted, and neither is OpenCode Zen, whose admission
// needs its verified metadata rather than a raw row (see package zen).
func AdmitAnonymousModel(registryID string, row map[string]any) AnonymousAdmission {
	switch registryID {
	case "kilo_code":
		isFree, _ := row["isFree"].(bool)
		pricing, _ := row["pricing"].(map[string]any)
		architecture, _ := row["architecture"].(map[string]any)
		return AnonymousAdmission{
			Free: isFree && zeroPrice(pricing["prompt"]) && zeroPrice(pricing["completion"]) &&
				listContains(architecture["output_modalities"], "text"),
			ContextWindow: rowInt(row["context_length"]),
		}
	case "llm7":
		tier, _ := row["tier"].(string)
		modelType, _ := row["model_type"].(string)
		usageBasedOnly, usageKnown := row["usage_based_only"].(bool)
		return AnonymousAdmission{
			Free: strings.EqualFold(tier, "turbo") && strings.EqualFold(modelType, "chat") &&
				listContains(row["schema_endpoints"], "openai") && usageKnown && !usageBasedOnly,
			ContextWindow: rowInt(row["context_length"]),
		}
	case "ovh_ai_endpoints":
		pricing, _ := row["pricing"].(map[string]any)
		return AnonymousAdmission{
			Free: rowInt(row["context_length"]) > 0 && rowInt(row["max_completion_tokens"]) > 0 &&
				zeroPrice(pricing["prompt"]) && zeroPrice(pricing["completion"]),
			ContextWindow: rowInt(row["context_length"]),
		}
	case "pollinations":
		tier, _ := row["tier"].(string)
		reasoning, _ := row["reasoning"].(bool)
		tools, _ := row["tools"].(bool)
		return AnonymousAdmission{
			Free:      strings.EqualFold(tier, "anonymous") && listContains(row["output_modalities"], "text"),
			Reasoning: reasoning, ToolCalls: tools,
		}
	}
	return AnonymousAdmission{}
}

// DiscoverAnonymousModels queries an anonymous provider's catalog and returns
// only the models AdmitAnonymousModel admits.
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
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return nil, fmt.Errorf("discovery status %d: %s", resp.StatusCode, string(b))
	}

	bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, 10<<20))
	if err != nil {
		return nil, fmt.Errorf("read discovery response: %w", err)
	}

	var rows []map[string]any
	var envelope struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(bodyBytes, &envelope); err == nil && len(envelope.Data) > 0 {
		rows = envelope.Data
	} else if err := json.Unmarshal(bodyBytes, &rows); err != nil || len(rows) == 0 {
		return nil, fmt.Errorf("unrecognized model envelope format from %s", profile.BaseURL)
	}
	return admittedAnonymousModels(rows, profile), nil
}

func admittedAnonymousModels(rows []map[string]any, profile AnonymousProviderProfile) []core.ModelInfo {
	out := []core.ModelInfo{}
	for _, row := range rows {
		id, _ := row["id"].(string)
		if id == "" {
			id, _ = row["name"].(string)
		}
		if strings.TrimSpace(id) == "" || !AdmitAnonymousModel(profile.RegistryID, row).Free {
			continue
		}
		description, _ := row["description"].(string)
		out = append(out, core.ModelInfo{ID: id, Object: "model", OwnedBy: profile.ProviderID, Description: description})
	}
	return out
}

func zeroPrice(value any) bool {
	switch typed := value.(type) {
	case string:
		number, err := strconv.ParseFloat(strings.TrimSpace(typed), 64)
		return err == nil && number == 0 && !math.IsNaN(number) && !math.IsInf(number, 0)
	case float64:
		return typed == 0 && !math.IsNaN(typed) && !math.IsInf(typed, 0)
	default:
		return false
	}
}

func listContains(value any, wanted string) bool {
	items, _ := value.([]any)
	for _, item := range items {
		if text, ok := item.(string); ok && strings.EqualFold(text, wanted) {
			return true
		}
	}
	return false
}

func rowInt(value any) int {
	switch typed := value.(type) {
	case float64:
		return int(typed)
	case int:
		return typed
	case int64:
		return int(typed)
	case json.Number:
		if number, err := typed.Int64(); err == nil {
			return int(number)
		}
		if number, err := typed.Float64(); err == nil {
			return int(number)
		}
	}
	return 0
}
