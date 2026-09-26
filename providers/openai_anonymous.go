package providers

import (
	"context"
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// openAIAnonymousEntry reports a registry entry curated for anonymous
// automation whose catalog the OpenAI-compatible transport reads: without
// a key, it lists only the models AdmitAnonymousModel admits. OpenCode Zen
// is the other such entry, and Zen serves it.
func openAIAnonymousEntry(registryID string) bool {
	switch registryID {
	case "kilo_code", "llm7", "ovh_ai_endpoints", "pollinations":
		return true
	}
	return false
}

// admitAnonymousRows keeps the rows anonymous access admits, as the gateway
// normalizes an anonymous catalog: each is free, serves Chat Completions
// only, and declares chat and the context window its raw row states. A
// row is judged by the raw row its id, or else its name, keys; the last
// raw row of an id wins, as in the gateway.
func admitAnonymousRows(registryID string, models []core.ModelInfo, items []any) []core.ModelInfo {
	raw := make(map[string]map[string]any, len(items))
	for _, item := range items {
		row, _ := item.(map[string]any)
		id, _ := row["id"].(string)
		if id == "" {
			id, _ = row["name"].(string)
		}
		raw[id] = row
	}
	admitted := make([]core.ModelInfo, 0, len(models))
	for _, model := range models {
		admission := AdmitAnonymousModel(registryID, raw[model.ID])
		if !admission.Free {
			continue
		}
		capabilities := maps.Clone(model.LegacyCapabilities)
		if capabilities == nil {
			capabilities = map[string]any{}
		}
		if admission.ContextWindow > 0 {
			capabilities["context_window"] = admission.ContextWindow
		}
		capabilities["chat"] = true
		model.Free, model.SupportedAPIs, model.LegacyCapabilities = true, []string{"/chat/completions"}, capabilities
		admitted = append(admitted, model)
	}
	return admitted
}

// decodePollinationsCatalog reads the Pollinations catalog as the gateway
// does: a bare JSON array, never an envelope, within the catalog bound,
// whose every row names its model. Only models that output text are
// listed, each serving Chat Completions, and anonymous access lists only
// those AdmitAnonymousModel admits.
func decodePollinationsCatalog(ctx context.Context, response *http.Response, anonymous bool) ([]core.ModelInfo, error) {
	defer response.Body.Close()
	status := response.StatusCode
	raw, err := io.ReadAll(io.LimitReader(response.Body, catalogMaxResponseBytes+1))
	if err != nil {
		return nil, catalogFailure(ctx, CatalogCodeTransportError, "Provider catalog response could not be read.", status, 0, err)
	}
	if len(raw) > catalogMaxResponseBytes {
		return nil, catalogFailure(ctx, CatalogCodeNotDiscoverable, "Provider catalog response exceeded the size limit.", status, 0, nil)
	}
	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil || items == nil {
		return nil, catalogFailure(ctx, CatalogCodeInvalidJSON, "Provider catalog response was not valid JSON.", status, 0, err)
	}
	models := []core.ModelInfo{}
	for _, item := range items {
		id, ok := item["name"].(string)
		if !ok || strings.TrimSpace(id) == "" {
			return nil, catalogFailure(ctx, CatalogCodeInvalidShape, "Provider catalog response contained an invalid model row.", status, 0, nil)
		}
		admission := AdmitAnonymousModel("pollinations", item)
		if (anonymous && !admission.Free) || !listContains(item["output_modalities"], "text") {
			continue
		}
		capabilities := map[string]any{"chat": true}
		if admission.Reasoning {
			capabilities["reasoning"] = true
		}
		if admission.ToolCalls {
			capabilities["tool_calls"] = true
		}
		// The gateway labels a Pollinations row with its id.
		models = append(models, core.ModelInfo{
			ID: id, Object: "model", DisplayName: id, Free: admission.Free,
			LegacyCapabilities: capabilities, SupportedAPIs: []string{"/chat/completions"},
		})
	}
	return models, nil
}
