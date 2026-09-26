package core

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"slices"
	"time"
)

// Principal represents the authenticated caller making a request.
//
// Deprecated: new APIs take Caller, and Principal.Caller converts. Principal
// and the Engine hooks that use it move to Caller in the next minor release.
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

// Resolution records both the resolved targets and the canonical category name,
// when the request addressed a category rather than a direct model.
type Resolution struct {
	Targets  []Target `json:"targets"`
	Category string   `json:"category,omitempty"`
}

// Credential contains authentication secrets for an upstream provider.
//
// fmt and slog print whether each secret is set, never its value. JSON
// encoding carries every field but Metadata, so slog's JSON handler reveals
// the secrets of a credential nested in another logged value.
type Credential struct {
	APIKey             string            `json:"api_key,omitempty"`
	Token              string            `json:"token,omitempty"`
	ConnectionID       string            `json:"connection_id,omitempty"`
	CredentialRevision int64             `json:"credential_revision,omitempty"`
	Headers            map[string]string `json:"headers,omitempty"`
	// AccountID identifies the upstream account the credential belongs to,
	// such as the ChatGPT account a Codex token acts for. Empty when unknown.
	AccountID string `json:"account_id,omitempty"`
	// TokenType names the kind of credential: TokenTypeAPIKey for a static
	// key, usually "Bearer" for an OAuth token, or a kind the provider
	// defines, such as a service account.
	TokenType string `json:"token_type,omitempty"`
	// Metadata carries provider-specific values, such as a project ID.
	// Providers read it and never write it, because a credential may serve
	// concurrent requests. It may hold secrets, such as a client secret, so
	// it never serializes and its values never print.
	Metadata map[string]string `json:"-"`
}

// String describes the credential without its secrets. The API key, the
// token and the headers print only whether they are set, and each Metadata
// key prints with only whether its value is set.
func (c Credential) String() string {
	return fmt.Sprintf("core.Credential{ConnectionID:%q CredentialRevision:%d AccountID:%q TokenType:%q APIKey:%s Token:%s Headers:%s Metadata:%v}",
		c.ConnectionID, c.CredentialRevision, c.AccountID, c.TokenType,
		presence(c.APIKey != ""), presence(c.Token != ""), presence(len(c.Headers) > 0), redactValues(c.Metadata))
}

// GoString keeps %#v from printing secrets.
func (c Credential) GoString() string { return c.String() }

// Format keeps every verb from printing secrets. Without it, fmt uses String
// and GoString only for %v, %s, %q, %x and %X, and prints the fields for any
// other verb, secrets included. Only %p of a value and %w escape it: fmt
// reports those as bad verbs and prints the fields without calling a method.
func (c Credential) Format(state fmt.State, verb rune) {
	switch verb {
	case 'v', 's', 'q', 'x', 'X':
		if verb == 'v' && state.Flag('#') {
			fmt.Fprint(state, c.GoString())
			return
		}
		fmt.Fprintf(state, fmt.FormatString(state, verb), c.String())
	default:
		fmt.Fprintf(state, "%%!%c(core.Credential=%s)", verb, c.String())
	}
}

// LogValue keeps structured logging from printing secrets. Metadata keys are
// sorted, so equal credentials log alike.
func (c Credential) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("connection_id", c.ConnectionID),
		slog.Int64("credential_revision", c.CredentialRevision),
		slog.String("account_id", c.AccountID),
		slog.String("token_type", c.TokenType),
		slog.Bool("has_api_key", c.APIKey != ""),
		slog.Bool("has_token", c.Token != ""),
		slog.Bool("has_headers", len(c.Headers) > 0),
		slog.Any("metadata_keys", slices.Sorted(maps.Keys(c.Metadata))),
	)
}

// presence says whether a secret is set without revealing it.
func presence(set bool) string {
	if set {
		return "redacted"
	}
	return "absent"
}

// redactValues replaces each value with its presence. fmt prints a map in key
// order, so the keys print sorted.
func redactValues(values map[string]string) map[string]string {
	redacted := make(map[string]string, len(values))
	for key, value := range values {
		redacted[key] = presence(value != "")
	}
	return redacted
}

// StreamIter yields complete SSE frames from an active provider stream.
type StreamIter interface {
	Next() ([]byte, error)
	Close() error
}

// ProviderErrorClassification is the provider-independent routing metadata for
// an operation failure. StatusCode is zero when no HTTP response was received,
// and RetryAfter is zero when the provider gave no delay. Disposition
// summarizes what the failure permits.
type ProviderErrorClassification struct {
	StatusCode       int
	Retryable        bool
	FailoverEligible bool
	CircuitFailure   bool
	RetryAfter       time.Duration
}

// ProviderErrorClassifier exposes routing metadata without requiring callers
// to depend on a provider package's concrete error type.
type ProviderErrorClassifier interface {
	ProviderErrorClassification() ProviderErrorClassification
}

func (e *ProviderOperationError) ProviderErrorClassification() ProviderErrorClassification {
	if e == nil {
		return ProviderErrorClassification{}
	}
	if errors.Is(e, context.Canceled) || errors.Is(e, context.DeadlineExceeded) {
		return ProviderErrorClassification{StatusCode: e.Failure.StatusCode}
	}
	health := ClassifyProviderFailure(e.Failure)
	return ProviderErrorClassification{
		StatusCode:       e.Failure.StatusCode,
		Retryable:        health.Retryable,
		FailoverEligible: health.Retryable,
		CircuitFailure:   health.ErrorClass == ProviderErrorTransport || health.ErrorClass == ProviderErrorUpstream,
	}
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
//
// DisplayName, Vendor, Free and LegacyCapabilities carry what the gateway's
// catalogs record about a row beyond the fields before them. Each is left
// out of JSON when empty, so a row that sets none of them encodes exactly as
// it did before they existed.
type ModelInfo struct {
	ID            string             `json:"id"`
	Object        string             `json:"object"`
	Created       int64              `json:"created"`
	OwnedBy       string             `json:"owned_by"`
	Description   string             `json:"description,omitempty"`
	Tags          []string           `json:"tags,omitempty"`
	APIEligible   *bool              `json:"api_eligible,omitempty"`
	APIVisibility string             `json:"api_visibility,omitempty"`
	SupportedAPIs []string           `json:"supported_apis,omitempty"`
	Capabilities  *ModelCapabilities `json:"capabilities,omitempty"`
	// DisplayName is the name to show for the model, such as a catalog
	// row's display_name.
	DisplayName string `json:"display_name,omitempty"`
	// Vendor is who makes the model, as the upstream catalog names it.
	Vendor string `json:"vendor,omitempty"`
	// Free marks a model the upstream serves at no cost, such as a free
	// model an anonymous catalog admits. Catalogs that predate the field
	// tag such a row "free" instead.
	Free bool `json:"free,omitempty"`
	// LegacyCapabilities is the gateway's untyped capability map, such as
	// {"chat": true, "vision": true, "context_window": 128000}, kept as the
	// catalog reported it. InferCapabilities reads it. Copies of a row share
	// the map, so copy it before changing it.
	LegacyCapabilities map[string]any `json:"legacy_capabilities,omitempty"`
}

// GenerateImagesRequest is the storage-neutral input for image synthesis.
// Count is a caller-side result bound; providers may return fewer images.
type GenerateImagesRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Count  int    `json:"count,omitempty"`
}

// GeneratedImage contains one decoded inline image returned by a provider.
type GeneratedImage struct {
	Data     []byte `json:"-"`
	MimeType string `json:"mime_type"`
}

// GenerateImagesResult contains generated bytes and provider-native usage.
type GenerateImagesResult struct {
	Images []GeneratedImage `json:"images"`
	Usage  map[string]any   `json:"usage,omitempty"`
}

// ImageGenerator is implemented by providers with a typed image-generation
// operation. It makes no assumptions about credential or image storage.
type ImageGenerator interface {
	GenerateImages(context.Context, GenerateImagesRequest, *Credential) (GenerateImagesResult, error)
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
