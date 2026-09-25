package providers

import (
	"encoding/json"
	"errors"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers/zen"
)

// ModelTagFree tags a catalog row that anonymous access admits: a free
// model usable without a credential, as AnonymousAdmission.Free reports.
const ModelTagFree = "free"

// The endpoints a Zen catalog row lists for its native surface.
const (
	zenChatEndpoint      = "/chat/completions"
	zenResponsesEndpoint = "/responses"
)

// parseZenAnonymousModels derives the models anonymous access admits from
// the models.dev catalog and Zen's live /models: the verified catalog of
// zen.Normalize. The gateway marked each such row free and gave it the
// endpoint of its native surface, and routed it by that endpoint, so each
// row here is tagged ModelTagFree and lists that one endpoint.
func parseZenAnonymousModels(metadata, live []byte, options zen.NormalizeOptions) ([]core.ModelInfo, error) {
	evidence, err := zen.Normalize(metadata, live, options)
	if err != nil {
		var document *zen.DocumentError
		if errors.As(err, &document) {
			code := document.Code
			if document.Document == zen.DocumentMetadata {
				code = "metadata_" + code
			}
			return nil, zenCatalogFailure(code, "OpenCode Zen could not read "+zenDocumentName(document.Document), 0, 0, err)
		}
		return nil, err
	}
	for index := range evidence.Models {
		model := &evidence.Models[index]
		model.Tags = []string{ModelTagFree}
		if surface, ok := zen.NativeSurface(*model); ok {
			model.SupportedAPIs = []string{zenEndpoint(surface)}
		}
	}
	return evidence.Models, nil
}

func zenDocumentName(document string) string {
	if document == zen.DocumentMetadata {
		return "the models.dev catalog"
	}
	return "its model catalog"
}

// parseZenModels reads the /models catalog of a keyed Zen connection as the
// gateway reads it, with its generic OpenAI-compatible reader: every row
// must name a model by id or name. The id wins unless blank, and the vendor
// or owner, the display name or name, the creation time and any endpoints
// the row lists carry over. The current catalog lists no endpoints, so a
// keyed row routes by model alone (see zenRowSurfaces). Capabilities are
// what the gateway infers from the endpoints, because the catalog reports
// nothing else.
func parseZenModels(raw []byte, discoveredAt time.Time) ([]core.ModelInfo, error) {
	var decoded any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, zenCatalogFailure("invalid_json", "the OpenCode Zen catalog is not valid JSON", 0, 0, err)
	}
	body, ok := decoded.(map[string]any)
	if !ok {
		return nil, zenCatalogFailure("invalid_shape", "the OpenCode Zen catalog is not a JSON object", 0, 0, nil)
	}
	rows, ok := body["data"].([]any)
	if !ok {
		return nil, zenCatalogFailure("invalid_shape", "the OpenCode Zen catalog has no data array", 0, 0, nil)
	}
	// Every row is checked before any is kept, so a malformed page never
	// replaces a usable catalog with a partial one.
	models := make([]core.ModelInfo, 0, len(rows))
	for _, raw := range rows {
		row, ok := raw.(map[string]any)
		if !ok || !zenRowIdentified(row) {
			return nil, zenCatalogFailure("invalid_shape", "the OpenCode Zen catalog has an invalid model row", 0, 0, nil)
		}
		models = append(models, zenModel(row, discoveredAt))
	}
	return models, nil
}

// zenRowIdentified reports a row whose id or name is a nonblank string. An
// identity of another type makes the row invalid even beside a good one.
func zenRowIdentified(row map[string]any) bool {
	identified := false
	for _, key := range []string{"id", "name"} {
		value, exists := row[key]
		if !exists {
			continue
		}
		id, ok := value.(string)
		if !ok {
			return false
		}
		identified = identified || strings.TrimSpace(id) != ""
	}
	return identified
}

func zenModel(row map[string]any, discoveredAt time.Time) core.ModelInfo {
	id, _ := row["id"].(string)
	if strings.TrimSpace(id) == "" {
		id, _ = row["name"].(string)
	}
	owner, _ := row["vendor"].(string)
	if owner == "" {
		owner, _ = row["owned_by"].(string)
	}
	model := core.ModelInfo{ID: id, Object: "model", OwnedBy: owner}
	if object, _ := row["object"].(string); object != "" {
		model.Object = object
	}
	if display, _ := row["display_name"].(string); display != "" && display != id {
		model.Description = display
	} else if name, ok := row["name"].(string); ok && name != id {
		model.Description = name
	}
	if created, ok := row["created"].(float64); ok {
		model.Created = int64(created)
	}
	for _, raw := range anySlice(row["supported_endpoints"]) {
		if endpoint, ok := raw.(string); ok {
			model.SupportedAPIs = append(model.SupportedAPIs, endpoint)
		}
	}
	model.Capabilities = zenInferredCapabilities(model.SupportedAPIs, discoveredAt)
	return model
}

// zenInferredCapabilities is what the gateway infers for a row that reports
// only its endpoints: each listed surface and operation is supported, chat
// implies text input, and everything else is unknown.
func zenInferredCapabilities(endpoints []string, discoveredAt time.Time) *core.ModelCapabilities {
	capabilities := &core.ModelCapabilities{
		SchemaVersion: core.ModelCapabilitiesSchemaVersion,
		Provenance:    core.ModelCapabilityProvenance{Source: core.ModelCapabilitySourceInferred, Confidence: core.ModelCapabilityConfidenceMedium},
	}
	operations := &capabilities.Operations
	for _, raw := range endpoints {
		endpoint := strings.ToLower(strings.TrimSpace(raw))
		switch {
		case endpoint == "/chat/completions" || endpoint == "/v1/chat/completions":
			capabilities.Surfaces.ChatCompletions, operations.Chat = core.SupportSupported, core.SupportSupported
		case endpoint == "/responses" || endpoint == "/v1/responses" || endpoint == "ws:/responses":
			capabilities.Surfaces.Responses, operations.Chat = core.SupportSupported, core.SupportSupported
		case endpoint == "/messages" || endpoint == "/v1/messages":
			capabilities.Surfaces.Messages, operations.Chat = core.SupportSupported, core.SupportSupported
		case strings.Contains(endpoint, "/embeddings"):
			operations.Embeddings = core.SupportSupported
		case strings.Contains(endpoint, "/images/"):
			operations.Image = core.SupportSupported
		case strings.Contains(endpoint, "/audio/transcriptions"):
			operations.AudioIn = core.SupportSupported
		case strings.Contains(endpoint, "/audio/speech"):
			operations.AudioOut = core.SupportSupported
		case strings.Contains(endpoint, "/videos/"):
			operations.Video = core.SupportSupported
		case strings.Contains(endpoint, "count_tokens"):
			operations.TokenCount = core.SupportSupported
		}
	}
	if operations.Chat == core.SupportSupported {
		capabilities.Inputs.Text = core.SupportSupported
	}
	if !discoveredAt.IsZero() {
		discovered := discoveredAt.UTC()
		capabilities.Freshness.DiscoveredAt = &discovered
	}
	return capabilities
}

// zenRowSurfaces reads which surfaces a catalog row serves natively from
// the endpoints it lists, as the gateway routes Zen models.
func zenRowSurfaces(model core.ModelInfo) (chat, responses bool) {
	for _, endpoint := range model.SupportedAPIs {
		switch strings.ToLower(strings.TrimSpace(endpoint)) {
		case "/chat/completions", "/v1/chat/completions":
			chat = true
		case "/responses", "/v1/responses", "ws:/responses":
			responses = true
		}
	}
	return chat, responses
}

func zenEndpoint(surface core.ModelSurface) string {
	if surface == core.ModelSurfaceResponses {
		return zenResponsesEndpoint
	}
	return zenChatEndpoint
}

// zenCatalogFailure reports a catalog Zen could not list. The cause is a
// *CatalogError whose Code is the gateway's catalog error code without its
// catalog_ prefix: metadata_ codes are the models.dev catalog's, the others
// Zen's own, so a product maps them without reading messages.
func zenCatalogFailure(code, detail string, status int, retryAfter time.Duration, cause error) *core.ProviderError {
	failure := &core.ProviderError{
		Message: detail,
		Cause:   &CatalogError{Code: code, Detail: detail, Status: status, RetryAfter: retryAfter, Cause: cause},
	}
	switch strings.TrimPrefix(code, "metadata_") {
	case "http_error":
		failure.Class = core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: status}).ErrorClass
		failure.Classification = (&InvocationError{Status: status, RetryAfter: retryAfter}).ProviderErrorClassification()
	case "transport_error":
		failure.Class = core.ProviderErrorTransport
		failure.Classification = core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}
	default:
		// The upstream answered with a catalog that cannot be read.
		failure.Class = core.ProviderErrorUpstream
		failure.Classification = core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}
	}
	return failure
}
