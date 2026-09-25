package zen

import (
	"encoding/json"
	"sort"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// The documents the anonymous catalog is derived from.
const (
	// DocumentMetadata is the models.dev catalog.
	DocumentMetadata = "metadata"
	// DocumentCatalog is Zen's live /models catalog.
	DocumentCatalog = "catalog"
)

// DocumentError reports an upstream document the catalog cannot be derived
// from. Code is "invalid_json" for a document that does not decode and
// "invalid_shape" for one that lacks the structure the catalog needs. The
// message is the one Discover and DiscoverVerified have always returned,
// which products match on.
type DocumentError struct {
	Document string
	Code     string
	message  string
}

func (e *DocumentError) Error() string { return e.message }

func documentError(document, code, message string) error {
	return &DocumentError{Document: document, Code: code, message: message}
}

// NormalizeOptions are what Normalize needs besides the documents.
type NormalizeOptions struct {
	// ProviderID owns every model. Empty uses DefaultProviderID.
	ProviderID string
	// ObservedAt stamps the evidence and the capability freshness.
	ObservedAt time.Time
	// CapabilityTTL is how long capabilities stay fresh. Zero or less uses
	// one hour.
	CapabilityTTL time.Duration
}

// Normalize derives the anonymous catalog from its upstream documents
// without fetching them. With live nil it returns the models.dev snapshot
// Discover returns; otherwise the subset DiscoverVerified returns: snapshot
// models the live catalog lists, whose status is active or beta and whose
// every described price is exactly zero.
//
// A failure returns failed evidence and a *DocumentError. Normalize is pure,
// so fixtures can pin how every upstream shape is read.
func Normalize(metadata, live []byte, options NormalizeOptions) (core.CatalogEvidence, error) {
	if strings.TrimSpace(options.ProviderID) == "" {
		options.ProviderID = DefaultProviderID
	}
	if options.CapabilityTTL <= 0 {
		options.CapabilityTTL = time.Hour
	}
	options.ObservedAt = options.ObservedAt.UTC()
	public, provider, err := snapshot(metadata, options)
	if err != nil || live == nil {
		return public, err
	}
	return verify(public, provider, live)
}

// snapshot is the OpenCode-compatible public snapshot: models with zero
// input cost and an admitted status, on the surface their npm package names.
func snapshot(raw []byte, options NormalizeOptions) (core.CatalogEvidence, metadataProvider, error) {
	observedAt := options.ObservedAt
	failed := core.CatalogEvidence{Status: core.CatalogFailed, ObservedAt: observedAt}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return failed, metadataProvider{}, documentError(DocumentMetadata, "invalid_json", "decode models.dev catalog")
	}
	providerRaw, ok := root["opencode"]
	if !ok {
		return failed, metadataProvider{}, documentError(DocumentMetadata, "invalid_shape", "models.dev catalog has no opencode provider")
	}
	var metadata metadataProvider
	if err := json.Unmarshal(providerRaw, &metadata); err != nil {
		return failed, metadataProvider{}, documentError(DocumentMetadata, "invalid_json", "decode models.dev opencode provider")
	}
	if metadata.Models == nil {
		return failed, metadataProvider{}, documentError(DocumentMetadata, "invalid_shape", "decode models.dev opencode provider")
	}

	models := make([]core.ModelInfo, 0, len(metadata.Models))
	for key, model := range metadata.Models {
		if model.ID != key || model.ID == "" {
			continue
		}
		if !admittedStatus(model.Status) || !inputCostZero(model.Cost) {
			continue
		}
		surface, ok := nativeSurface(metadata.NPM, model)
		if !ok {
			continue
		}
		models = append(models, core.ModelInfo{
			ID: key, Object: "model", OwnedBy: options.ProviderID,
			Description: model.Name, Capabilities: capabilities(model, surface, observedAt, options.CapabilityTTL),
		})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	status := core.CatalogDiscovered
	if len(models) == 0 {
		status = core.CatalogEmpty
	}
	return core.CatalogEvidence{Status: status, Models: models, ObservedAt: observedAt}, metadata, nil
}

// verify keeps the snapshot models the live catalog lists whose metadata is
// active or beta and prices every described cost at exactly zero.
func verify(public core.CatalogEvidence, metadata metadataProvider, raw []byte) (core.CatalogEvidence, error) {
	failed := core.CatalogEvidence{Status: core.CatalogFailed, ObservedAt: public.ObservedAt}
	var catalog catalogEnvelope
	if err := json.Unmarshal(raw, &catalog); err != nil {
		return failed, documentError(DocumentCatalog, "invalid_json", "decode Zen catalog")
	}
	if catalog.Data == nil {
		return failed, documentError(DocumentCatalog, "invalid_shape", "decode Zen catalog")
	}
	live := make(map[string]bool, len(catalog.Data))
	for _, row := range catalog.Data {
		live[strings.TrimSpace(row.ID)] = true
	}
	verified := public.Models[:0]
	for _, model := range public.Models {
		metadataModel := metadata.Models[model.ID]
		if live[model.ID] && verifiedStatus(metadataModel.Status) && exactZeroCost(metadataModel.Cost) {
			verified = append(verified, model)
		}
	}
	public.Models = verified
	if len(verified) == 0 {
		public.Status = core.CatalogEmpty
	}
	return public, nil
}
