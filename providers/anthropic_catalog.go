package providers

import (
	"context"
	"net/http"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// anthropicCatalogFreshness is how long a listed model's capabilities stay
// fresh: the gateway's Anthropic catalog TTL.
const anthropicCatalogFreshness = time.Hour

// ListModels lists the models /v1/models returns for the credential, as the
// gateway lists them: the one page the endpoint answers, every row a
// nonblank string id. Each model serves Messages, with the display name
// Anthropic gives it and the capabilities the registry declares for
// Anthropic, fresh for an hour. A catalog failure's *CatalogError has the
// gateway's code.
func (p *Anthropic) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	auth, err := anthropicCredential(credential)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/v1/models", nil)
	if err != nil {
		return nil, core.NewConfigurationError("the Anthropic catalog request could not be created", err)
	}
	if err := anthropicAuthorize(ctx, request, auth); err != nil {
		return nil, err
	}
	response, err := p.catalogClient.Do(request)
	if err != nil {
		return nil, catalogFailure(ctx, CatalogCodeTransportError, "Provider catalog request could not reach the upstream service.", 0, 0, err)
	}
	now := p.now()
	body, err := decodeCatalogResponse(ctx, response, now, "data", "id")
	if err != nil {
		return nil, err
	}
	rows := body["data"].([]any)
	models := make([]core.ModelInfo, 0, len(rows))
	for _, item := range rows {
		row := item.(map[string]any)
		label, _ := row["display_name"].(string)
		models = append(models, core.ModelInfo{
			ID: row["id"].(string), Object: "model", OwnedBy: "anthropic", Vendor: "anthropic", DisplayName: label,
			SupportedAPIs: []string{"/v1/messages"}, Capabilities: anthropicModelCapabilities(now.UTC()),
		})
	}
	return models, nil
}

// anthropicModelCapabilities is what the registry declares for every
// Anthropic model, as the gateway records it: chat over Messages only,
// streamed, with high confidence and an hour's freshness from discovery.
// Everything else stays unknown.
func anthropicModelCapabilities(discoveredAt time.Time) *core.ModelCapabilities {
	expiresAt := discoveredAt.Add(anthropicCatalogFreshness)
	return &core.ModelCapabilities{
		SchemaVersion: core.ModelCapabilitiesSchemaVersion,
		Operations:    core.ModelOperationCapabilities{Chat: core.SupportSupported},
		Surfaces: core.ModelSurfaceCapabilities{
			ChatCompletions: core.SupportUnsupported,
			Responses:       core.SupportUnsupported,
			Messages:        core.SupportSupported,
		},
		Streaming: core.SupportSupported,
		Provenance: core.ModelCapabilityProvenance{
			Source: core.ModelCapabilitySourceRegistryStatic, Confidence: core.ModelCapabilityConfidenceHigh,
		},
		Freshness: core.ModelCapabilityFreshness{DiscoveredAt: &discoveredAt, ExpiresAt: &expiresAt},
	}
}
