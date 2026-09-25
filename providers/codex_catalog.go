package providers

import (
	"time"

	core "github.com/xibodev/llmgw-core"
)

// codexResponsesEndpoint is the only surface the Codex backend serves.
const codexResponsesEndpoint = "/responses"

// CodexVerifiedClientVersion is the Codex client version whose catalog the
// parser in this package was verified against: the client_version the
// gateway sends. It is not a default. CodexConfig.ClientVersion names the
// product making the call, so a product must still pass one explicitly,
// this value or its own.
const CodexVerifiedClientVersion = "0.155.1"

// parseCodexModels parses a Codex catalog into the models a caller can use:
// rows supported in the API and listed. It drops hidden rows, rows the API
// does not serve and rows that do not say, as the gateway always has.
//
// Current catalogs omit the legacy supported_endpoints field. Such a row
// still serves Responses, so it is given that endpoint rather than none: a
// model without a surface is not routable, which is how Codex models once
// vanished from the gateway's model list. An endpoint the catalog does list
// is kept as listed.
func parseCodexModels(raw []byte, discoveredAt time.Time) ([]core.ModelInfo, error) {
	rows, err := parseCodexCatalog(raw, discoveredAt)
	if err != nil {
		return nil, err
	}
	models := make([]core.ModelInfo, 0, len(rows))
	for _, model := range rows {
		if model.APIEligible == nil || !*model.APIEligible || model.APIVisibility != "list" {
			continue
		}
		if len(model.SupportedAPIs) == 0 {
			model.SupportedAPIs = []string{codexResponsesEndpoint}
		}
		models = append(models, model)
	}
	return models, nil
}
