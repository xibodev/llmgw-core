package providers

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// ListModels returns the catalog the credential can use, read as the
// gateway reads an OpenAI-compatible catalog: every row of /models, with
// its vendor, display name, endpoints and the capabilities it reports.
// Each row's typed capabilities are what core.InferCapabilities derives
// from it, as the gateway derives them when it stores the row.
func (p *OpenAICompatible) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	access, err := p.access(credential)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/models", nil)
	if err != nil {
		return nil, core.NewConfigurationError("the "+p.label+" catalog request could not be created", err)
	}
	request.Header = p.header(access, "", false)
	response, err := p.catalogClient.Do(request)
	if err != nil {
		return nil, catalogFailure(ctx, CatalogCodeTransportError, "Provider catalog request could not reach the upstream service.", 0, 0, err)
	}
	discoveredAt := p.now()
	if status := response.StatusCode; status >= http.StatusBadRequest {
		response.Body.Close()
		// The gateway reads this catalog's refusals, 401 and 403 among
		// them, as http_error.
		retryAfter := retryAfterDelay(response.Header.Get("Retry-After"), discoveredAt)
		return nil, catalogFailure(ctx, CatalogCodeHTTPError, fmt.Sprintf("Provider catalog returned HTTP %d.", status), status, retryAfter, nil)
	}
	models, err := p.decodeCatalog(ctx, response, discoveredAt)
	if err != nil {
		return nil, err
	}
	for index := range models {
		models[index].Capabilities = core.InferCapabilities(models[index], discoveredAt, time.Time{})
	}
	return models, nil
}

// decodeCatalog reads a data envelope whose every row names its model by
// id or name, checking every row before any is used.
func (p *OpenAICompatible) decodeCatalog(ctx context.Context, response *http.Response, now time.Time) ([]core.ModelInfo, error) {
	body, err := decodeCatalogResponse(ctx, response, now, "data", "id", "name")
	if err != nil {
		return nil, err
	}
	items, _ := body["data"].([]any)
	models := make([]core.ModelInfo, 0, len(items))
	for _, item := range items {
		row, _ := item.(map[string]any)
		models = append(models, openAICatalogModel(row))
	}
	return models, nil
}

// openAICatalogModel reads one checked row as the gateway does. The id
// wins unless blank; the vendor, or else the owner, makes the model; the
// display name, or else the name, describes it unless it repeats the id.
// supported_endpoints is kept as listed, without values that are not
// strings, and a capabilities block is distilled by openAICapabilities.
func openAICatalogModel(row map[string]any) core.ModelInfo {
	id, _ := row["id"].(string)
	if strings.TrimSpace(id) == "" {
		id, _ = row["name"].(string)
	}
	vendor, _ := row["vendor"].(string)
	if vendor == "" {
		vendor, _ = row["owned_by"].(string)
	}
	model := core.ModelInfo{ID: id, Object: "model", OwnedBy: vendor, Vendor: vendor}
	if display, ok := row["display_name"].(string); ok && display != "" && display != id {
		model.DisplayName = display
	} else if name, ok := row["name"].(string); ok && name != id {
		model.DisplayName = name
	}
	if created, ok := row["created"].(float64); ok {
		model.Created = int64(created)
	}
	if capabilities := openAICapabilities(row["capabilities"]); len(capabilities) > 0 {
		model.LegacyCapabilities = capabilities
	}
	for _, raw := range anySlice(row["supported_endpoints"]) {
		if endpoint, ok := raw.(string); ok {
			model.SupportedAPIs = append(model.SupportedAPIs, endpoint)
		}
	}
	return model
}

// openAICapabilities distills a capabilities block, as GitHub Copilot's
// catalog reports one, into the gateway's untyped capabilities: the
// reasoning efforts, the headline feature flags, the context window and
// the output limit, and the family. It is empty for a block that reports
// none of them.
func openAICapabilities(raw any) map[string]any {
	block, ok := raw.(map[string]any)
	if !ok {
		return nil
	}
	distilled := map[string]any{}
	if supports, ok := block["supports"].(map[string]any); ok {
		if efforts, ok := supports["reasoning_effort"].([]any); ok && len(efforts) > 0 {
			levels := []string{}
			for _, effort := range efforts {
				if level, ok := effort.(string); ok {
					levels = append(levels, level)
				}
			}
			distilled["reasoning_effort"] = levels
		}
		for _, flag := range []string{"streaming", "tool_calls", "vision", "structured_outputs", "parallel_tool_calls"} {
			if value, ok := supports[flag].(bool); ok {
				distilled[flag] = value
			}
		}
	}
	if limits, ok := block["limits"].(map[string]any); ok {
		if window := rowInt(limits["max_context_window_tokens"]); window > 0 {
			distilled["context_window"] = window
		} else if window := rowInt(limits["max_prompt_tokens"]); window > 0 {
			distilled["context_window"] = window
		}
		if output := rowInt(limits["max_output_tokens"]); output > 0 {
			distilled["max_output_tokens"] = output
		}
	}
	if family, ok := block["family"].(string); ok && family != "" {
		distilled["family"] = family
	}
	return distilled
}
