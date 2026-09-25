package providers

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// copilotCatalogFreshness is how long catalog capabilities stay fresh. The
// gateway plans a transport with an hour's freshness from discovery.
const copilotCatalogFreshness = time.Hour

// copilotModel is one normalized catalog row: the model a caller sees, and
// what the transport needs to serve its Chat requests.
type copilotModel struct {
	info core.ModelInfo
	// reasoningEffort reports a nonempty capabilities.supports.reasoning_effort
	// list, whatever its values. The gateway sends max_tokens to such a model
	// as max_completion_tokens.
	reasoningEffort bool
}

// parseCopilotCatalog normalizes a Copilot catalog into the model set and
// capabilities the gateway derives. It is strict where the gateway is: the
// catalog must be an object with a data array, and every row an object
// whose id or name is a nonblank string. An id or a name that is present
// but not a string rejects the whole catalog, so a malformed discovery never
// replaces a usable catalog with a partial one.
//
// Every optional field may be absent, as it is on older rows. A row of an
// id seen before is dropped: the gateway lists and routes the first.
func parseCopilotCatalog(raw []byte, discoveredAt time.Time) ([]copilotModel, error) {
	var decoded any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, fmt.Errorf("the catalog is not JSON: %w", err)
	}
	envelope, ok := decoded.(map[string]any)
	if !ok {
		return nil, errors.New("the catalog is not a JSON object")
	}
	rows, ok := envelope["data"].([]any)
	if !ok {
		return nil, errors.New("the catalog has no data array")
	}
	models := make([]copilotModel, 0, len(rows))
	seen := make(map[string]bool, len(rows))
	for index, value := range rows {
		row, ok := value.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("catalog row %d is not an object", index)
		}
		id, err := copilotModelID(row)
		if err != nil {
			return nil, fmt.Errorf("catalog row %d: %w", index, err)
		}
		if seen[id] {
			continue
		}
		seen[id] = true
		models = append(models, copilotCatalogModel(id, row, discoveredAt))
	}
	return models, nil
}

// copilotModelID returns the row's id, or its name when the id is blank, as
// the gateway reads them. Neither is trimmed.
func copilotModelID(row map[string]any) (string, error) {
	named := false
	for _, field := range []string{"id", "name"} {
		value, present := row[field]
		if !present {
			continue
		}
		text, ok := value.(string)
		if !ok {
			return "", fmt.Errorf("%s is not a string", field)
		}
		named = named || strings.TrimSpace(text) != ""
	}
	if !named {
		return "", errors.New("the row has neither an id nor a name")
	}
	id, _ := row["id"].(string)
	if strings.TrimSpace(id) == "" {
		id, _ = row["name"].(string)
	}
	return id, nil
}

// copilotCatalogModel reads one validated row. The vendor owns the model,
// and the display name, or else the name, describes it unless it merely
// repeats the id. supported_endpoints is kept as listed, without values
// that are not strings; a row that omits it lists no endpoints.
func copilotCatalogModel(id string, row map[string]any, discoveredAt time.Time) copilotModel {
	info := core.ModelInfo{ID: id, Object: "model"}
	if info.OwnedBy, _ = row["vendor"].(string); info.OwnedBy == "" {
		info.OwnedBy, _ = row["owned_by"].(string)
	}
	if label, ok := row["display_name"].(string); ok && label != "" && label != id {
		info.Description = label
	} else if name, ok := row["name"].(string); ok && name != id {
		info.Description = name
	}
	if endpoints, ok := row["supported_endpoints"].([]any); ok {
		for _, endpoint := range endpoints {
			if text, ok := endpoint.(string); ok {
				info.SupportedAPIs = append(info.SupportedAPIs, text)
			}
		}
	}
	capabilities, _ := row["capabilities"].(map[string]any)
	model := copilotModel{info: info}
	model.info.Capabilities, model.reasoningEffort = copilotModelCapabilities(capabilities, info.SupportedAPIs, discoveredAt)
	return model
}

// copilotModelCapabilities derives a row's capabilities as the gateway does:
// the supports flags and limits Copilot reports, and a surface and the chat
// operation for each endpoint the row lists. What the row does not report
// stays unknown, and only an explicit false is unsupported, so a row without
// endpoints serves no known surface. Copilot's own fields are inferred into
// the schema, so the confidence is medium, as the gateway records it.
func copilotModelCapabilities(capabilities map[string]any, endpoints []string, discoveredAt time.Time) (*core.ModelCapabilities, bool) {
	derived := &core.ModelCapabilities{
		SchemaVersion: core.ModelCapabilitiesSchemaVersion,
		Provenance: core.ModelCapabilityProvenance{
			Source: core.ModelCapabilitySourceInferred, Confidence: core.ModelCapabilityConfidenceMedium,
		},
	}
	supports, _ := capabilities["supports"].(map[string]any)
	levels, _ := supports["reasoning_effort"].([]any)
	for _, level := range levels {
		if _, ok := level.(string); ok {
			derived.Reasoning = core.SupportSupported
		}
	}
	derived.Inputs.Image = copilotFlag(supports["vision"])
	derived.Tools = copilotFlag(supports["tool_calls"])
	derived.StructuredOutput = copilotFlag(supports["structured_outputs"])
	derived.Streaming = copilotFlag(supports["streaming"])

	limits, _ := capabilities["limits"].(map[string]any)
	if tokens := copilotLimit(limits["max_context_window_tokens"]); tokens != nil {
		derived.Limits.ContextTokens = tokens
	} else {
		derived.Limits.ContextTokens = copilotLimit(limits["max_prompt_tokens"])
	}
	derived.Limits.MaxOutputTokens = copilotLimit(limits["max_output_tokens"])

	for _, endpoint := range endpoints {
		copilotEndpointCapabilities(derived, strings.ToLower(strings.TrimSpace(endpoint)))
	}
	if derived.Operations.Chat == core.SupportSupported {
		derived.Inputs.Text = core.SupportSupported
	}
	if !discoveredAt.IsZero() {
		discovered := discoveredAt.UTC()
		expires := discovered.Add(copilotCatalogFreshness)
		derived.Freshness = core.ModelCapabilityFreshness{DiscoveredAt: &discovered, ExpiresAt: &expires}
	}
	return derived, len(levels) > 0
}

// copilotEndpointCapabilities records what one listed endpoint serves. The
// three chat surfaces are matched exactly; other operations by path.
func copilotEndpointCapabilities(derived *core.ModelCapabilities, endpoint string) {
	switch {
	case endpoint == "/chat/completions" || endpoint == "/v1/chat/completions":
		derived.Surfaces.ChatCompletions = core.SupportSupported
		derived.Operations.Chat = core.SupportSupported
	case copilotResponsesEndpoint(endpoint):
		derived.Surfaces.Responses = core.SupportSupported
		derived.Operations.Chat = core.SupportSupported
	case endpoint == "/messages" || endpoint == "/v1/messages":
		derived.Surfaces.Messages = core.SupportSupported
		derived.Operations.Chat = core.SupportSupported
	case strings.Contains(endpoint, "/embeddings"):
		derived.Operations.Embeddings = core.SupportSupported
	case strings.Contains(endpoint, "/images/"):
		derived.Operations.Image = core.SupportSupported
	case strings.Contains(endpoint, "/audio/transcriptions"):
		derived.Operations.AudioIn = core.SupportSupported
	case strings.Contains(endpoint, "/audio/speech"):
		derived.Operations.AudioOut = core.SupportSupported
	case strings.Contains(endpoint, "/videos/"):
		derived.Operations.Video = core.SupportSupported
	case strings.Contains(endpoint, "count_tokens"):
		derived.Operations.TokenCount = core.SupportSupported
	}
}

// copilotResponsesEndpoint reports a normalized endpoint that serves
// Responses. The gateway accepts the websocket form too.
func copilotResponsesEndpoint(endpoint string) bool {
	return endpoint == "/responses" || endpoint == "/v1/responses" || endpoint == "ws:/responses"
}

// copilotFlag reads a supports flag: absent or not a boolean is unknown.
func copilotFlag(value any) core.Support {
	enabled, ok := value.(bool)
	switch {
	case !ok:
		return core.SupportUnknown
	case enabled:
		return core.SupportSupported
	}
	return core.SupportUnsupported
}

// copilotLimit reads a positive token limit, truncating a fraction as the
// gateway does.
func copilotLimit(value any) *int64 {
	number, _ := value.(float64)
	if limit := int64(number); limit > 0 {
		return &limit
	}
	return nil
}
