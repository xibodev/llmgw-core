package providers

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/url"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
)

// registryManifestJSON is the reviewed default manifest. It is read only
// through DefaultRegistry, which validates it once.
//
//go:embed registry_manifest.json
var registryManifestJSON []byte

const (
	ProviderAvailable  = "available"
	ProviderPlanned    = "planned"
	ProviderClientOnly = "client_only"

	ConnectionScopeSystemOrPersonal = "system_or_personal"
	ConnectionScopePersonal         = "personal"
	ConnectionScopeGatewayClient    = "gateway_client"
)

// Auth methods a registry entry may advertise.
const (
	AuthAPIKey            = "api_key"
	AuthGatewayClient     = "gateway_client"
	AuthNone              = "none"
	AuthOAuthBrowser      = "oauth_browser"
	AuthOAuthDevice       = "oauth_device"
	AuthTokenImport       = "token_import"
	AuthGCPServiceAccount = "gcp_service_account"
	AuthSetupToken        = "setup_token"
)

// Risk levels. Every level but RiskGreen requires a risk notice.
const (
	RiskGreen  = "green"
	RiskYellow = "yellow"
	RiskRed    = "red"
)

// RegistryEntry describes a curated provider integration independently from a
// configured provider instance.
//
// Core owns an entry's wire and safety facts. Priority, Icon, Domain,
// CommonModels and Extensions are presentation and curation fields that
// product overlays set; the default manifest leaves them empty.
type RegistryEntry struct {
	ID                     string   `json:"id"`
	Aliases                []string `json:"aliases,omitempty"`
	Label                  string   `json:"label"`
	Description            string   `json:"description"`
	Categories             []string `json:"categories,omitempty"`
	RuntimeType            string   `json:"runtime_type"`
	Protocol               string   `json:"protocol"`
	Availability           string   `json:"availability"`
	AuthMethods            []string `json:"auth_methods"`
	ConnectionScope        string   `json:"connection_scope"`
	DefaultProviderID      string   `json:"default_provider_id"`
	DefaultBaseURL         string   `json:"default_base_url,omitempty"`
	DefaultRegion          string   `json:"default_region,omitempty"`
	DefaultLocation        string   `json:"default_location,omitempty"`
	RequiresBaseURL        bool     `json:"requires_base_url,omitempty"`
	RequiresAPIKey         bool     `json:"requires_api_key,omitempty"`
	SupportsAPIAdaptation  bool     `json:"supports_api_adaptation,omitempty"`
	SupportsModelDiscovery bool     `json:"supports_model_discovery,omitempty"`
	AnonymousAutomation    bool     `json:"anonymous_automation,omitempty"`
	InferAudioCapabilities bool     `json:"infer_audio_capabilities,omitempty"`
	OnboardingFields       []string `json:"onboarding_fields,omitempty"`
	AuthAdapter            string   `json:"auth_adapter,omitempty"`
	QuotaAdapter           string   `json:"quota_adapter,omitempty"`
	RiskLevel              string   `json:"risk_level,omitempty"`
	RiskNotice             string   `json:"risk_notice,omitempty"`
	DocsURL                string   `json:"docs_url,omitempty"`
	ClientOnly             bool     `json:"client_only,omitempty"`

	// Priority orders entries: higher first, ties in manifest order.
	Priority float64 `json:"priority,omitempty"`
	// Icon is a presentation slug, such as a brand icon name.
	Icon string `json:"icon,omitempty"`
	// Domain is the provider's public host name, for presentation.
	Domain string `json:"domain,omitempty"`
	// CommonModels suggests model ids to offer first.
	CommonModels []string `json:"common_models,omitempty"`
	// Extensions carries product-private metadata keyed by product name.
	// Core validates only that each value is JSON.
	Extensions map[string]json.RawMessage `json:"extensions,omitempty"`
}

func cloneRegistryEntry(entry RegistryEntry) RegistryEntry {
	entry.Aliases = append([]string(nil), entry.Aliases...)
	entry.AuthMethods = append([]string(nil), entry.AuthMethods...)
	entry.Categories = append([]string(nil), entry.Categories...)
	entry.OnboardingFields = append([]string(nil), entry.OnboardingFields...)
	entry.CommonModels = append([]string(nil), entry.CommonModels...)
	if entry.Extensions != nil {
		extensions := make(map[string]json.RawMessage, len(entry.Extensions))
		for key, value := range entry.Extensions {
			extensions[key] = append(json.RawMessage(nil), value...)
		}
		entry.Extensions = extensions
	}
	return entry
}

// ValidationOptions supplies the product facts a manifest cannot know.
type ValidationOptions struct {
	// RuntimeTypes lists the runtime types the product can execute. Every
	// available entry must use one of them. Nil skips the check.
	RuntimeTypes []string
}

// Registry is an immutable, validated set of provider integrations. It is safe
// for concurrent use, and every accessor returns copies.
type Registry struct {
	entries []RegistryEntry
	ids     map[string]int
	names   map[string]int
}

// NewRegistry validates and normalizes entries into a Registry. The input is
// not modified.
func NewRegistry(entries []RegistryEntry, options ValidationOptions) (*Registry, error) {
	normalized := make([]RegistryEntry, len(entries))
	for index, entry := range entries {
		normalized[index] = cloneRegistryEntry(entry)
	}
	if err := validateRegistry(normalized, options); err != nil {
		return nil, err
	}
	sort.SliceStable(normalized, func(i, j int) bool { return normalized[i].Priority > normalized[j].Priority })
	registry := &Registry{
		entries: normalized,
		ids:     make(map[string]int, len(normalized)),
		names:   map[string]int{},
	}
	for index, entry := range normalized {
		registry.ids[entry.ID] = index
		registry.names[entry.ID] = index
		for _, alias := range entry.Aliases {
			registry.names[alias] = index
		}
	}
	return registry, nil
}

// DecodeRegistryEntries strictly decodes a manifest: unknown fields and
// trailing content are errors. Pass the result to NewRegistry to validate it.
func DecodeRegistryEntries(payload []byte) ([]RegistryEntry, error) {
	var entries []RegistryEntry
	if err := decodeStrict(payload, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

func decodeStrict(payload []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("decode JSON: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return fmt.Errorf("decode JSON: trailing content")
	}
	return nil
}

// defaultRegistry validates the embedded manifest once. The Registry it holds
// is immutable, so sharing it keeps no mutable package state.
var defaultRegistry = sync.OnceValue(func() *Registry {
	entries, err := DecodeRegistryEntries(registryManifestJSON)
	if err == nil {
		var registry *Registry
		if registry, err = NewRegistry(entries, ValidationOptions{}); err == nil {
			return registry
		}
	}
	panic("llmgw-core: embedded provider registry manifest is invalid: " + err.Error())
})

// DefaultRegistry returns the reviewed default registry. Products layer their
// own curation on it with WithOverlay.
func DefaultRegistry() *Registry { return defaultRegistry() }

// Len reports the number of entries.
func (r *Registry) Len() int { return len(r.entries) }

// Entries returns copies of every entry, highest priority first and otherwise
// in manifest order.
func (r *Registry) Entries() []RegistryEntry {
	out := make([]RegistryEntry, len(r.entries))
	for index, entry := range r.entries {
		out[index] = cloneRegistryEntry(entry)
	}
	return out
}

// Lookup resolves an entry by id or alias, ignoring case and surrounding space.
func (r *Registry) Lookup(name string) (RegistryEntry, bool) {
	index, ok := r.names[normalizeIdentifier(name)]
	if !ok {
		return RegistryEntry{}, false
	}
	return cloneRegistryEntry(r.entries[index]), true
}

// ByID resolves only canonical ids, never aliases.
func (r *Registry) ByID(id string) (RegistryEntry, bool) {
	index, ok := r.ids[normalizeIdentifier(id)]
	if !ok {
		return RegistryEntry{}, false
	}
	return cloneRegistryEntry(r.entries[index]), true
}

// CanonicalID resolves an alias to its entry id and returns unknown
// identifiers in normalized form.
func (r *Registry) CanonicalID(name string) string {
	if entry, ok := r.Lookup(name); ok {
		return entry.ID
	}
	return normalizeIdentifier(name)
}

// ProviderRegistry returns copies of the default registry's entries.
func ProviderRegistry() []RegistryEntry { return DefaultRegistry().Entries() }

// RegistryProvider resolves a default registry entry by id or alias.
func RegistryProvider(id string) (RegistryEntry, bool) { return DefaultRegistry().Lookup(id) }

// RegistryProviderByID resolves only canonical default registry ids.
func RegistryProviderByID(id string) (RegistryEntry, bool) { return DefaultRegistry().ByID(id) }

// CanonicalRegistryID resolves a default registry alias while preserving
// unknown custom identifiers in normalized form.
func CanonicalRegistryID(id string) string { return DefaultRegistry().CanonicalID(id) }

// ---- overlays ----------------------------------------------------------- //

// Overlay layers one product's curation onto a registry. Core owns each
// entry's wire and safety facts, so an overlay changes only presentation,
// priority and curation fields; a product that needs different wire behavior
// adds its own entry under its own id. The result is validated like a
// manifest.
type Overlay struct {
	// Add appends product-only entries. Their ids must not already exist in
	// the base registry, even when removed.
	Add []RegistryEntry `json:"add,omitempty"`
	// Override changes fields of existing entries.
	Override []EntryOverride `json:"override,omitempty"`
	// Remove drops base entries the product does not offer.
	Remove []string `json:"remove,omitempty"`
}

// EntryOverride changes presentation, priority and curation fields of one
// entry. A nil pointer or nil slice leaves the field unchanged, and an empty
// slice clears it. Aliases are added to the entry's own, so the aliases core
// resolves keep working; Extensions are merged by key.
type EntryOverride struct {
	ID               string                     `json:"id"`
	Label            *string                    `json:"label,omitempty"`
	Description      *string                    `json:"description,omitempty"`
	Categories       []string                   `json:"categories,omitempty"`
	Icon             *string                    `json:"icon,omitempty"`
	Domain           *string                    `json:"domain,omitempty"`
	DocsURL          *string                    `json:"docs_url,omitempty"`
	Priority         *float64                   `json:"priority,omitempty"`
	Availability     *string                    `json:"availability,omitempty"`
	Aliases          []string                   `json:"aliases,omitempty"`
	CommonModels     []string                   `json:"common_models,omitempty"`
	OnboardingFields []string                   `json:"onboarding_fields,omitempty"`
	Extensions       map[string]json.RawMessage `json:"extensions,omitempty"`
}

// DecodeOverlay strictly decodes an overlay. An override naming a field the
// overlay may not change is an unknown field, and so an error.
func DecodeOverlay(payload []byte) (Overlay, error) {
	var overlay Overlay
	if err := decodeStrict(payload, &overlay); err != nil {
		return Overlay{}, err
	}
	return overlay, nil
}

// WithOverlay returns a new Registry with the overlay applied and validated.
// The receiver is unchanged.
func (r *Registry) WithOverlay(overlay Overlay, options ValidationOptions) (*Registry, error) {
	base := r.Entries()
	index := make(map[string]int, len(base))
	for position, entry := range base {
		index[entry.ID] = position
	}

	removed := map[string]bool{}
	for _, raw := range overlay.Remove {
		id := normalizeIdentifier(raw)
		if _, ok := index[id]; !ok {
			return nil, fmt.Errorf("overlay removes unknown entry %q", id)
		}
		if removed[id] {
			return nil, fmt.Errorf("overlay removes entry %q twice", id)
		}
		removed[id] = true
	}

	overridden := map[string]bool{}
	for _, override := range overlay.Override {
		id := normalizeIdentifier(override.ID)
		position, ok := index[id]
		switch {
		case !ok:
			return nil, fmt.Errorf("overlay overrides unknown entry %q", id)
		case removed[id]:
			return nil, fmt.Errorf("overlay overrides removed entry %q", id)
		case overridden[id]:
			return nil, fmt.Errorf("overlay overrides entry %q twice", id)
		}
		overridden[id] = true
		override.apply(&base[position])
	}

	result := make([]RegistryEntry, 0, len(base)+len(overlay.Add))
	for _, entry := range base {
		if !removed[entry.ID] {
			result = append(result, entry)
		}
	}
	for _, entry := range overlay.Add {
		id := normalizeIdentifier(entry.ID)
		if _, ok := index[id]; ok {
			return nil, fmt.Errorf("overlay adds %q, which the base registry defines; override it or use a product id", id)
		}
		result = append(result, cloneRegistryEntry(entry))
	}
	return NewRegistry(result, options)
}

func (o EntryOverride) apply(entry *RegistryEntry) {
	if o.Label != nil {
		entry.Label = *o.Label
	}
	if o.Description != nil {
		entry.Description = *o.Description
	}
	if o.Categories != nil {
		entry.Categories = append([]string{}, o.Categories...)
	}
	if o.Icon != nil {
		entry.Icon = *o.Icon
	}
	if o.Domain != nil {
		entry.Domain = *o.Domain
	}
	if o.DocsURL != nil {
		entry.DocsURL = *o.DocsURL
	}
	if o.Priority != nil {
		entry.Priority = *o.Priority
	}
	if o.Availability != nil {
		entry.Availability = *o.Availability
	}
	for _, alias := range o.Aliases {
		// An alias the entry already owns is not a collision, so an overlay
		// keeps working when core later adopts the same alias.
		if !slices.ContainsFunc(entry.Aliases, func(own string) bool {
			return normalizeIdentifier(own) == normalizeIdentifier(alias)
		}) && normalizeIdentifier(alias) != entry.ID {
			entry.Aliases = append(entry.Aliases, alias)
		}
	}
	if o.CommonModels != nil {
		entry.CommonModels = append([]string{}, o.CommonModels...)
	}
	if o.OnboardingFields != nil {
		entry.OnboardingFields = append([]string{}, o.OnboardingFields...)
	}
	if len(o.Extensions) > 0 && entry.Extensions == nil {
		entry.Extensions = make(map[string]json.RawMessage, len(o.Extensions))
	}
	for key, value := range o.Extensions {
		entry.Extensions[key] = append(json.RawMessage(nil), value...)
	}
}

// ---- validation --------------------------------------------------------- //

var (
	registryIdentifierPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)
	registryDomainPattern     = regexp.MustCompile(`^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$`)
)

func normalizeIdentifier(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

// validateRegistry normalizes entries in place and rejects any manifest whose
// entries a product could not safely offer.
func validateRegistry(entries []RegistryEntry, options ValidationOptions) error {
	if len(entries) == 0 {
		return fmt.Errorf("manifest has no entries")
	}

	var runtimeTypes map[string]bool
	if options.RuntimeTypes != nil {
		runtimeTypes = make(map[string]bool, len(options.RuntimeTypes))
		for _, runtimeType := range options.RuntimeTypes {
			runtimeTypes[normalizeIdentifier(runtimeType)] = true
		}
	}
	allowedAvailability := map[string]bool{
		ProviderAvailable: true, ProviderPlanned: true, ProviderClientOnly: true,
	}
	allowedScopes := map[string]bool{
		ConnectionScopeSystemOrPersonal: true,
		ConnectionScopePersonal:         true,
		ConnectionScopeGatewayClient:    true,
	}
	allowedAuth := map[string]bool{
		AuthAPIKey: true, AuthGatewayClient: true, AuthNone: true, AuthOAuthBrowser: true,
		AuthOAuthDevice: true, AuthTokenImport: true, AuthGCPServiceAccount: true, AuthSetupToken: true,
	}
	allowedRisk := map[string]bool{RiskGreen: true, RiskYellow: true, RiskRed: true}
	claimedNames := map[string]string{}

	for index := range entries {
		entry := &entries[index]
		entry.ID = normalizeIdentifier(entry.ID)
		entry.DefaultProviderID = normalizeIdentifier(entry.DefaultProviderID)
		entry.RuntimeType = normalizeIdentifier(entry.RuntimeType)
		entry.Protocol = normalizeIdentifier(entry.Protocol)
		entry.Availability = normalizeIdentifier(entry.Availability)
		entry.ConnectionScope = normalizeIdentifier(entry.ConnectionScope)
		entry.RiskLevel = normalizeIdentifier(entry.RiskLevel)
		entry.AuthAdapter = normalizeIdentifier(entry.AuthAdapter)
		entry.QuotaAdapter = normalizeIdentifier(entry.QuotaAdapter)
		entry.Icon = normalizeIdentifier(entry.Icon)
		entry.Domain = normalizeIdentifier(entry.Domain)

		if !registryIdentifierPattern.MatchString(entry.ID) {
			return fmt.Errorf("entry %d has invalid id %q", index+1, entry.ID)
		}
		if entry.Label == "" || entry.Description == "" || entry.RuntimeType == "" ||
			entry.Protocol == "" || entry.DefaultProviderID == "" {
			return fmt.Errorf("entry %q is missing required metadata", entry.ID)
		}
		if !registryIdentifierPattern.MatchString(entry.DefaultProviderID) {
			return fmt.Errorf("entry %q has invalid default_provider_id %q", entry.ID, entry.DefaultProviderID)
		}
		if !allowedAvailability[entry.Availability] {
			return fmt.Errorf("entry %q has invalid availability %q", entry.ID, entry.Availability)
		}
		if !allowedScopes[entry.ConnectionScope] {
			return fmt.Errorf("entry %q has invalid connection scope %q", entry.ID, entry.ConnectionScope)
		}
		if !allowedRisk[entry.RiskLevel] {
			return fmt.Errorf("entry %q has invalid risk level %q", entry.ID, entry.RiskLevel)
		}
		if entry.RiskLevel != RiskGreen && strings.TrimSpace(entry.RiskNotice) == "" {
			return fmt.Errorf("entry %q requires a risk notice for risk level %q", entry.ID, entry.RiskLevel)
		}
		if runtimeTypes != nil && entry.Availability == ProviderAvailable && !runtimeTypes[entry.RuntimeType] {
			return fmt.Errorf("entry %q uses unsupported runtime type %q", entry.ID, entry.RuntimeType)
		}
		if len(entry.AuthMethods) == 0 {
			return fmt.Errorf("entry %q has no auth methods", entry.ID)
		}

		hasAPIKey := false
		hasNone := false
		requiresAuthAdapter := false
		for authIndex, method := range entry.AuthMethods {
			method = normalizeIdentifier(method)
			entry.AuthMethods[authIndex] = method
			if !allowedAuth[method] {
				return fmt.Errorf("entry %q has invalid auth method %q", entry.ID, method)
			}
			hasAPIKey = hasAPIKey || method == AuthAPIKey
			hasNone = hasNone || method == AuthNone
			requiresAuthAdapter = requiresAuthAdapter ||
				method == AuthOAuthBrowser || method == AuthOAuthDevice || method == AuthTokenImport
		}
		if entry.RequiresAPIKey && !hasAPIKey {
			return fmt.Errorf("entry %q requires an API key but does not advertise api_key auth", entry.ID)
		}
		if entry.ConnectionScope == ConnectionScopePersonal && hasNone {
			return fmt.Errorf("entry %q is personal but advertises no-auth access", entry.ID)
		}
		if entry.ClientOnly &&
			(entry.Availability != ProviderClientOnly || entry.ConnectionScope != ConnectionScopeGatewayClient) {
			return fmt.Errorf("entry %q has inconsistent client-only classification", entry.ID)
		}
		if requiresAuthAdapter && entry.Availability == ProviderAvailable {
			if !registryIdentifierPattern.MatchString(entry.AuthAdapter) {
				return fmt.Errorf("entry %q requires a valid auth_adapter", entry.ID)
			}
		}
		if entry.AuthAdapter != "" && !registryIdentifierPattern.MatchString(entry.AuthAdapter) {
			return fmt.Errorf("entry %q has invalid auth_adapter %q", entry.ID, entry.AuthAdapter)
		}
		if entry.AnonymousAutomation {
			if entry.Availability != ProviderAvailable || entry.RuntimeType != "openai_compatible" ||
				entry.Protocol != "openai" || entry.ConnectionScope != ConnectionScopeSystemOrPersonal ||
				entry.RequiresAPIKey || !entry.SupportsModelDiscovery || !hasNone ||
				entry.RiskLevel != RiskGreen || strings.TrimSpace(entry.DefaultBaseURL) == "" {
				return fmt.Errorf("entry %q has an unsafe anonymous automation contract", entry.ID)
			}
			parsed, err := url.Parse(entry.DefaultBaseURL)
			host := ""
			if err == nil {
				host = strings.ToLower(parsed.Hostname())
			}
			if err != nil || parsed.Scheme != "https" || host == "" || net.ParseIP(host) != nil ||
				host == "localhost" || !strings.Contains(host, ".") || strings.HasSuffix(host, ".local") {
				return fmt.Errorf("entry %q anonymous automation requires a public HTTPS base URL", entry.ID)
			}
		}
		if err := validateRegistryURL(entry.ID, "default_base_url", entry.DefaultBaseURL); err != nil {
			return err
		}
		if err := validateRegistryURL(entry.ID, "docs_url", entry.DocsURL); err != nil {
			return err
		}

		names := append([]string{entry.ID}, entry.Aliases...)
		for nameIndex, name := range names {
			name = normalizeIdentifier(name)
			if !registryIdentifierPattern.MatchString(name) {
				return fmt.Errorf("entry %q has invalid id/alias %q", entry.ID, name)
			}
			if owner, exists := claimedNames[name]; exists {
				return fmt.Errorf("entry %q reuses id/alias %q claimed by %q", entry.ID, name, owner)
			}
			claimedNames[name] = entry.ID
			if nameIndex > 0 {
				entry.Aliases[nameIndex-1] = name
			}
		}
		for categoryIndex, category := range entry.Categories {
			category = normalizeIdentifier(category)
			if !registryIdentifierPattern.MatchString(category) {
				return fmt.Errorf("entry %q has invalid category %q", entry.ID, category)
			}
			entry.Categories[categoryIndex] = category
		}
		for fieldIndex, field := range entry.OnboardingFields {
			field = normalizeIdentifier(field)
			if !registryIdentifierPattern.MatchString(field) {
				return fmt.Errorf("entry %q has invalid onboarding field %q", entry.ID, field)
			}
			entry.OnboardingFields[fieldIndex] = field
		}

		if math.IsNaN(entry.Priority) || math.IsInf(entry.Priority, 0) {
			return fmt.Errorf("entry %q has a non-finite priority", entry.ID)
		}
		if entry.Icon != "" && !registryIdentifierPattern.MatchString(entry.Icon) {
			return fmt.Errorf("entry %q has invalid icon %q", entry.ID, entry.Icon)
		}
		if entry.Domain != "" && !registryDomainPattern.MatchString(entry.Domain) {
			return fmt.Errorf("entry %q has invalid domain %q", entry.ID, entry.Domain)
		}
		for modelIndex, model := range entry.CommonModels {
			model = strings.TrimSpace(model)
			if model == "" {
				return fmt.Errorf("entry %q has an empty common model", entry.ID)
			}
			entry.CommonModels[modelIndex] = model
		}
		for key, value := range entry.Extensions {
			if !registryIdentifierPattern.MatchString(key) {
				return fmt.Errorf("entry %q has invalid extension key %q", entry.ID, key)
			}
			if !json.Valid(value) {
				return fmt.Errorf("entry %q extension %q is not valid JSON", entry.ID, key)
			}
		}
	}
	return nil
}

func validateRegistryURL(entryID, field, value string) error {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("entry %q has invalid %s %q", entryID, field, value)
	}
	return nil
}
