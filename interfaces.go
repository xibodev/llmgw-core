package core

import (
	"context"
	"net/http"
	"time"
)

// Principal represents the authenticated caller making a request.
type Principal struct {
	ID        string         `json:"id"`
	Type      string         `json:"type"` // "user", "api_key", "anonymous", "service"
	Name      string         `json:"name,omitempty"`
	ProjectID string         `json:"project_id,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// Target represents a resolved provider and model.
type Target struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// Credential contains authentication secrets for an upstream provider.
type Credential struct {
	APIKey             string            `json:"api_key,omitempty"`
	Token              string            `json:"token,omitempty"`
	ConnectionID       string            `json:"connection_id,omitempty"`
	CredentialRevision int64             `json:"credential_revision,omitempty"`
	Headers            map[string]string `json:"headers,omitempty"`
}

// UsageRecord holds metrics and token counts from a completed LLM call.
type UsageRecord struct {
	RequestID    string        `json:"request_id"`
	PrincipalID  string        `json:"principal_id,omitempty"`
	ProjectID    string        `json:"project_id,omitempty"`
	Provider     string        `json:"provider"`
	Model        string        `json:"model"`
	InputTokens  int           `json:"input_tokens"`
	OutputTokens int           `json:"output_tokens"`
	Duration     time.Duration `json:"duration"`
	Stream       bool          `json:"stream"`
	StatusCode   int           `json:"status_code"`
	Error        string        `json:"error,omitempty"`
}

// ModelInfo describes an available upstream model or alias.
type ModelInfo struct {
	ID          string   `json:"id"`
	Object      string   `json:"object"`
	Created     int64    `json:"created"`
	OwnedBy     string   `json:"owned_by"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
}

// Authenticator authenticates an incoming HTTP request.
// If not configured, requests proceed as anonymous.
type Authenticator interface {
	Authenticate(r *http.Request) (*Principal, error)
}

// AuthenticatorFunc is a functional adapter for Authenticator.
type AuthenticatorFunc func(r *http.Request) (*Principal, error)

func (f AuthenticatorFunc) Authenticate(r *http.Request) (*Principal, error) {
	return f(r)
}

// PolicyGate controls whether a principal is authorized to access a given provider/model target.
type PolicyGate interface {
	Allows(ctx context.Context, principal *Principal, target Target) (bool, string)
}

// AllowAllPolicy is a PolicyGate that permits all requests.
type AllowAllPolicy struct{}

func (AllowAllPolicy) Allows(ctx context.Context, principal *Principal, target Target) (bool, string) {
	return true, ""
}

// CredentialResolver resolves upstream credentials for a principal and provider.
type CredentialResolver interface {
	Resolve(ctx context.Context, principal *Principal, providerID string) (*Credential, error)
}

// StaticCredentialResolver returns pre-configured credentials from a map.
type StaticCredentialResolver struct {
	Credentials map[string]Credential
}

func (s StaticCredentialResolver) Resolve(ctx context.Context, principal *Principal, providerID string) (*Credential, error) {
	if cred, ok := s.Credentials[providerID]; ok {
		return &cred, nil
	}
	return nil, nil
}

// UsageHook receives telemetry records after each completed request.
type UsageHook interface {
	RecordUsage(ctx context.Context, record UsageRecord)
}

// UsageHookFunc is a functional adapter for UsageHook.
type UsageHookFunc func(ctx context.Context, record UsageRecord)

func (f UsageHookFunc) RecordUsage(ctx context.Context, record UsageRecord) {
	f(ctx, record)
}
