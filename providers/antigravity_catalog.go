package providers

import (
	"encoding/json"
	"sort"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// The gateway surfaces an Antigravity catalog row names: every model chats,
// and a model in the image roster also generates images.
const (
	antigravityChatEndpoint   = "/v1/chat/completions"
	antigravityImagesEndpoint = "/v1/images/generations"
)

// antigravityCatalogResponse is a fetchAvailableModels response. The
// rosters are pointers because an absent roster says nothing about a
// model, while a roster that leaves a model out rules it out.
type antigravityCatalogResponse struct {
	Models                     map[string]antigravityCatalogModel `json:"models"`
	AudioTranscriptionModelIDs *[]string                          `json:"audioTranscriptionModelIds"`
	ImageGenerationModelIDs    *[]string                          `json:"imageGenerationModelIds"`
	TabModelIDs                []string                           `json:"tabModelIds"`
	TieredModelIDs             struct {
		Flash     []string `json:"flash"`
		FlashLite []string `json:"flashLite"`
		Pro       []string `json:"pro"`
	} `json:"tieredModelIds"`
}

type antigravityCatalogModel struct {
	DisplayName string `json:"displayName"`
	// MIME types and video support are retained as bounded upstream observations,
	// but are not routing facts in the version 1 capability schema.
	SupportedMimeTypes map[string]bool `json:"supportedMimeTypes"`
	SupportsImages     *bool           `json:"supportsImages"`
	SupportsThinking   *bool           `json:"supportsThinking"`
	SupportsVideo      *bool           `json:"supportsVideo"`
}

// parseAntigravityModels normalizes a fetchAvailableModels response into
// the rows and capabilities the gateway lists: every model in the root
// roster, sorted by ID, serves Chat Completions, and one the image roster
// names also serves image generation. No row streams, because Antigravity
// buffers Cloud Code Assist's stream; the gateway marks every row so, and
// routes no streaming request to one.
//
// What the catalog leaves out stays unknown: a model without
// supportsThinking has unknown reasoning, and a roster the response omits
// makes its operation unknown for every model. The limits, quotas and
// placeholders that current catalogs also carry are not capability facts
// the gateway derives, so they are ignored.
func parseAntigravityModels(raw []byte) ([]core.ModelInfo, error) {
	catalog, err := decodeAntigravityCatalog(raw)
	if err != nil {
		return nil, err
	}
	models := catalog.models()
	for _, model := range models {
		model.Capabilities.Streaming = core.SupportUnsupported
	}
	return models, nil
}

func decodeAntigravityCatalog(raw []byte) (antigravityCatalogResponse, error) {
	var catalog antigravityCatalogResponse
	err := json.Unmarshal(raw, &catalog)
	return catalog, err
}

// models returns a row for every model in the root roster, sorted by ID,
// with streaming left unknown. ExperimentalAntigravityProvider still lists
// these rows, so a product that marks them itself sees what it always saw.
func (c antigravityCatalogResponse) models() []core.ModelInfo {
	ids := make([]string, 0, len(c.Models))
	for id := range c.Models {
		if strings.TrimSpace(id) != "" {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	models := make([]core.ModelInfo, 0, len(ids))
	for _, id := range ids {
		metadata := c.Models[id]
		imageGeneration := rosterSupport(c.ImageGenerationModelIDs, id)
		supportedAPIs := []string{antigravityChatEndpoint}
		if imageGeneration == core.SupportSupported {
			supportedAPIs = append(supportedAPIs, antigravityImagesEndpoint)
		}
		models = append(models, core.ModelInfo{
			ID:            id,
			Object:        "model",
			OwnedBy:       "google-antigravity",
			Description:   metadata.DisplayName,
			SupportedAPIs: supportedAPIs,
			Capabilities:  antigravityModelCapabilities(metadata, imageGeneration, rosterSupport(c.AudioTranscriptionModelIDs, id)),
		})
	}
	return models
}

func antigravityModelCapabilities(metadata antigravityCatalogModel, imageGeneration, audioTranscription core.Support) *core.ModelCapabilities {
	capabilities := &core.ModelCapabilities{
		SchemaVersion: core.ModelCapabilitiesSchemaVersion,
		Operations: core.ModelOperationCapabilities{
			Chat: core.SupportSupported,
		},
		Surfaces: core.ModelSurfaceCapabilities{
			ChatCompletions: core.SupportSupported,
		},
		Inputs: core.ModelInputCapabilities{
			Text: core.SupportSupported,
		},
		Provenance: core.ModelCapabilityProvenance{
			Source:     core.ModelCapabilitySourceInferred,
			Confidence: core.ModelCapabilityConfidenceMedium,
		},
	}
	if metadata.SupportsThinking != nil {
		capabilities.Reasoning = supportFromBool(*metadata.SupportsThinking)
	}
	capabilities.Operations.Image = imageGeneration
	capabilities.Operations.AudioIn = audioTranscription
	return capabilities
}

func rosterSupport(roster *[]string, model string) core.Support {
	if roster == nil {
		return core.SupportUnknown
	}
	if stringSet(*roster)[model] {
		return core.SupportSupported
	}
	return core.SupportUnsupported
}

func stringSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			set[value] = true
		}
	}
	return set
}

func supportFromBool(value bool) core.Support {
	if value {
		return core.SupportSupported
	}
	return core.SupportUnsupported
}
