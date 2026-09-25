package providers

import (
	"context"
	"net/http"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// ListModels returns the models the daemon holds, from /api/tags, as the
// gateway lists them. Every row is checked before any is used. A row is
// named by its name, or by its model when the name is blank, and made by
// its details.family, or by "ollama". Ollama reports no surfaces or
// capabilities for a row.
func (p *Ollama) ListModels(ctx context.Context, _ *core.Credential) ([]core.ModelInfo, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, p.tagsURL, nil)
	if err != nil {
		return nil, core.NewConfigurationError("the Ollama catalog request could not be created", err)
	}
	response, err := p.catalogClient.Do(request)
	if err != nil {
		return nil, catalogFailure(ctx, CatalogCodeTransportError, "Provider catalog request could not reach the upstream service.", 0, 0, err)
	}
	body, err := decodeCatalogResponse(ctx, response, p.now(), "models", "name", "model")
	if err != nil {
		return nil, err
	}
	rows := body["models"].([]any)
	models := make([]core.ModelInfo, 0, len(rows))
	for _, item := range rows {
		row := item.(map[string]any)
		id, _ := row["name"].(string)
		if strings.TrimSpace(id) == "" {
			id, _ = row["model"].(string)
		}
		vendor := "ollama"
		if details, ok := row["details"].(map[string]any); ok {
			if family, ok := details["family"].(string); ok && family != "" {
				vendor = family
			}
		}
		models = append(models, core.ModelInfo{ID: id, Object: "model", OwnedBy: vendor, Vendor: vendor})
	}
	return models, nil
}
