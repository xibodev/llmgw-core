package core

import (
	"time"
)

// ProviderConfig configures a single upstream provider backend.
type ProviderConfig struct {
	Type          string            `json:"type"`                     // "openai_compatible", "anthropic", "googleai", "ollama", "github_copilot", "codex", "edgetts"
	BaseURL       string            `json:"base_url,omitempty"`       // Upstream base endpoint
	APIKey        string            `json:"api_key,omitempty"`        // Static API key or token
	Timeout       time.Duration     `json:"timeout,omitempty"`        // Request timeout
	Headers       map[string]string `json:"headers,omitempty"`        // Extra HTTP headers
	AllowedModels []string          `json:"allowed_models,omitempty"` // Whitelist of models
	Disabled      bool              `json:"disabled,omitempty"`       // Disable provider
}

// RouteConfig configures an alias (failover chain) across multiple provider targets.
type RouteConfig struct {
	Targets     []Target `json:"targets"`               // Ordered targets to try
	Fallback    bool     `json:"fallback,omitempty"`    // Fall back to subsequent targets on error
	Description string   `json:"description,omitempty"` // User-facing description
}

// Config configures the core LLM Gateway engine.
type Config struct {
	Providers    map[string]ProviderConfig `json:"providers"`
	Routes       map[string]RouteConfig    `json:"routes"`
	DefaultRoute string                    `json:"default_route,omitempty"`

	// Pluggable extension hooks
	Authenticator      Authenticator
	PolicyGate         PolicyGate
	CredentialResolver CredentialResolver
	UsageHook          UsageHook
}
