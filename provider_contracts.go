package core

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ProviderConnectionKind describes who owns and may use a provider connection.
type ProviderConnectionKind string

const (
	ProviderConnectionSystem               ProviderConnectionKind = "system"
	ProviderConnectionPersonal             ProviderConnectionKind = "personal"
	ProviderConnectionPersonalSubscription ProviderConnectionKind = "personal_subscription"
	ProviderConnectionAnonymous            ProviderConnectionKind = "anonymous"
)

// ProviderAuthKind describes how a provider connection authenticates.
type ProviderAuthKind string

const (
	ProviderAuthAPIKey           ProviderAuthKind = "api_key"
	ProviderAuthOAuthDevice      ProviderAuthKind = "oauth_device"
	ProviderAuthOAuthBrowser     ProviderAuthKind = "oauth_browser"
	ProviderAuthServiceAccount   ProviderAuthKind = "service_account"
	ProviderAuthWorkloadIdentity ProviderAuthKind = "workload_identity"
	ProviderAuthOfficialClient   ProviderAuthKind = "official_client"
	ProviderAuthAnonymous        ProviderAuthKind = "anonymous"
)

// ProviderConnection is the persistence-agnostic description of a configured
// provider connection. Credential is available to connectors but never JSON encoded.
type ProviderConnection struct {
	ID         string                 `json:"id,omitempty"`
	ProviderID string                 `json:"provider_id"`
	Kind       ProviderConnectionKind `json:"kind"`
	AuthKind   ProviderAuthKind       `json:"auth_kind"`
	OwnerID    string                 `json:"owner_id,omitempty"`
	Credential *Credential            `json:"-"`
}

// Validate rejects contradictory connection descriptions. Codex is always a
// human-owned subscription connection established through device OAuth.
func (c ProviderConnection) Validate() error {
	if strings.TrimSpace(c.ProviderID) == "" {
		return fmt.Errorf("provider id is required")
	}
	if c.Kind == ProviderConnectionAnonymous && c.AuthKind != ProviderAuthAnonymous {
		return fmt.Errorf("anonymous connections require anonymous auth")
	}
	if c.AuthKind == ProviderAuthAnonymous && c.Kind != ProviderConnectionAnonymous {
		return fmt.Errorf("anonymous auth requires an anonymous connection")
	}
	if isCodexProvider(c.ProviderID) {
		if c.Kind != ProviderConnectionPersonalSubscription || c.AuthKind != ProviderAuthOAuthDevice {
			return fmt.Errorf("codex requires a personal subscription OAuth device connection")
		}
		if strings.TrimSpace(c.OwnerID) == "" {
			return fmt.Errorf("codex requires a connection owner")
		}
	}
	return nil
}

func isCodexProvider(providerID string) bool {
	switch strings.ToLower(strings.TrimSpace(providerID)) {
	case "codex", "openai_codex", "chatgpt":
		return true
	default:
		return false
	}
}

type CatalogEvidenceStatus string

const (
	CatalogNotProbed  CatalogEvidenceStatus = "not_probed"
	CatalogDiscovered CatalogEvidenceStatus = "discovered"
	CatalogEmpty      CatalogEvidenceStatus = "empty"
	CatalogFailed     CatalogEvidenceStatus = "failed"
)

// CatalogEvidence records what a provider advertised. Discovery alone does not
// prove that any model can successfully perform inference.
type CatalogEvidence struct {
	Status     CatalogEvidenceStatus `json:"status"`
	Models     []ModelInfo           `json:"models,omitempty"`
	ObservedAt time.Time             `json:"observed_at,omitempty"`
}

type CompletionProbeStatus string

const (
	CompletionNotProbed CompletionProbeStatus = "not_probed"
	CompletionVerified  CompletionProbeStatus = "verified"
	CompletionFailed    CompletionProbeStatus = "failed"
)

// CompletionProbeEvidence records inference evidence for one exact target.
type CompletionProbeEvidence struct {
	Target     Target                `json:"target"`
	Status     CompletionProbeStatus `json:"status"`
	ObservedAt time.Time             `json:"observed_at,omitempty"`
	Latency    time.Duration         `json:"latency,omitempty"`
}

func (e CompletionProbeEvidence) InferenceVerified() bool {
	return e.Status == CompletionVerified && e.Target.Provider != "" && e.Target.Model != ""
}

type ProviderHealthStatus string

const (
	ProviderHealthUnknown   ProviderHealthStatus = "unknown"
	ProviderHealthHealthy   ProviderHealthStatus = "healthy"
	ProviderHealthDegraded  ProviderHealthStatus = "degraded"
	ProviderHealthUnhealthy ProviderHealthStatus = "unhealthy"
)

type ProviderErrorClass string

const (
	ProviderErrorNone        ProviderErrorClass = "none"
	ProviderErrorAuth        ProviderErrorClass = "auth"
	ProviderErrorForbidden   ProviderErrorClass = "forbidden"
	ProviderErrorRateLimited ProviderErrorClass = "rate_limited"
	ProviderErrorTransport   ProviderErrorClass = "transport"
	ProviderErrorUpstream    ProviderErrorClass = "upstream"
)

// ProviderFailure contains only values needed for classification. Err is never
// serialized because transport errors may include credentials or private hosts.
type ProviderFailure struct {
	StatusCode int       `json:"status_code,omitempty"`
	RetryAfter string    `json:"retry_after,omitempty"`
	ObservedAt time.Time `json:"observed_at,omitempty"`
	Err        error     `json:"-"`
}

// ProviderOperationError preserves classification inputs without exposing an
// upstream response body or transport error through serialized evidence.
type ProviderOperationError struct {
	Failure ProviderFailure
	Op      string
}

func (e *ProviderOperationError) Error() string {
	if e == nil {
		return "provider operation failed"
	}
	if e.Op != "" {
		return e.Op + " failed"
	}
	return "provider operation failed"
}

func (e *ProviderOperationError) Unwrap() error { return e.Failure.Err }

// NewProviderOperationError creates a classifiable provider error. Callers
// should not put response bodies, credentials, or private endpoints in op.
func NewProviderOperationError(op string, statusCode int, retryAfter string, err error) error {
	return &ProviderOperationError{
		Op:      op,
		Failure: ProviderFailure{StatusCode: statusCode, RetryAfter: retryAfter, Err: err},
	}
}

type ProviderHealthEvidence struct {
	Status     ProviderHealthStatus `json:"status"`
	ErrorClass ProviderErrorClass   `json:"error_class"`
	Permanent  bool                 `json:"permanent,omitempty"`
	Retryable  bool                 `json:"retryable,omitempty"`
	RetryAfter time.Duration        `json:"retry_after,omitempty"`
	ObservedAt time.Time            `json:"observed_at,omitempty"`
}

// ClassifyProviderFailure maps HTTP and transport outcomes into stable health semantics.
func ClassifyProviderFailure(f ProviderFailure) ProviderHealthEvidence {
	evidence := ProviderHealthEvidence{Status: ProviderHealthUnknown, ErrorClass: ProviderErrorNone, ObservedAt: f.ObservedAt}
	if f.StatusCode >= 200 && f.StatusCode < 300 && f.Err == nil {
		evidence.Status = ProviderHealthHealthy
		return evidence
	}
	if f.StatusCode == 0 && f.Err != nil {
		evidence.Status = ProviderHealthUnhealthy
		evidence.ErrorClass = ProviderErrorTransport
		evidence.Retryable = true
		return evidence
	}
	switch f.StatusCode {
	case 401:
		evidence.Status = ProviderHealthUnhealthy
		evidence.ErrorClass = ProviderErrorAuth
		evidence.Permanent = true
	case 403:
		evidence.Status = ProviderHealthUnhealthy
		evidence.ErrorClass = ProviderErrorForbidden
		evidence.Permanent = true
	case 429:
		evidence.Status = ProviderHealthDegraded
		evidence.ErrorClass = ProviderErrorRateLimited
		evidence.Retryable = true
		evidence.RetryAfter = parseRetryAfter(f.RetryAfter, f.ObservedAt)
	default:
		evidence.Status = ProviderHealthUnhealthy
		evidence.ErrorClass = ProviderErrorUpstream
		evidence.Retryable = f.StatusCode == 408 || f.StatusCode >= 500
	}
	return evidence
}

func classifyProviderError(err error, observedAt time.Time) ProviderHealthEvidence {
	failure := ProviderFailure{ObservedAt: observedAt, Err: err}
	var operationError *ProviderOperationError
	if errors.As(err, &operationError) {
		failure = operationError.Failure
		failure.ObservedAt = observedAt
	}
	return ClassifyProviderFailure(failure)
}

func parseRetryAfter(value string, observedAt time.Time) time.Duration {
	value = strings.TrimSpace(value)
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil && seconds >= 0 {
		return time.Duration(seconds) * time.Second
	}
	if when, err := http.ParseTime(value); err == nil && !observedAt.IsZero() && when.After(observedAt) {
		return when.Sub(observedAt)
	}
	return 0
}

type TargetPublicationPolicy string

const (
	PublishDiscoveredTargets TargetPublicationPolicy = "catalog_discovered"
	PublishVerifiedTargets   TargetPublicationPolicy = "inference_verified"
)

// PublishExactTargets converts catalog evidence to exact provider/model targets.
// Under verification policy, only targets with successful inference evidence publish.
func PublishExactTargets(providerID string, catalog CatalogEvidence, probes []CompletionProbeEvidence, policy TargetPublicationPolicy) ([]Target, error) {
	if strings.TrimSpace(providerID) == "" {
		return nil, fmt.Errorf("provider id is required")
	}
	if policy != PublishDiscoveredTargets && policy != PublishVerifiedTargets {
		return nil, fmt.Errorf("unsupported target publication policy %q", policy)
	}
	if catalog.Status != CatalogDiscovered {
		return nil, nil
	}

	verified := make(map[Target]bool, len(probes))
	for _, probe := range probes {
		if probe.InferenceVerified() {
			verified[probe.Target] = true
		}
	}
	seen := make(map[Target]bool, len(catalog.Models))
	targets := make([]Target, 0, len(catalog.Models))
	for _, model := range catalog.Models {
		target := Target{Provider: providerID, Model: strings.TrimSpace(model.ID)}
		if target.Model == "" || seen[target] || (policy == PublishVerifiedTargets && !verified[target]) {
			continue
		}
		seen[target] = true
		targets = append(targets, target)
	}
	return targets, nil
}

type ProviderConnectRequest struct {
	Connection        ProviderConnection      `json:"connection"`
	PublicationPolicy TargetPublicationPolicy `json:"publication_policy"`
	// AuthenticationValidated permits callers that established authentication
	// before Connect to omit an adapter authentication validator.
	AuthenticationValidated bool `json:"authentication_validated,omitempty"`
}

// ProviderConnectResult is the complete, persistence-agnostic outcome of provider onboarding.
type ProviderConnectResult struct {
	Connection ProviderConnection        `json:"connection"`
	Catalog    CatalogEvidence           `json:"catalog"`
	Probes     []CompletionProbeEvidence `json:"probes,omitempty"`
	Health     ProviderHealthEvidence    `json:"health"`
	Targets    []Target                  `json:"targets,omitempty"`
}

// ProviderConnector orchestrates provider-specific authentication, discovery,
// probing, classification, and exact target publication.
type ProviderConnector interface {
	Connect(ctx context.Context, request ProviderConnectRequest) (ProviderConnectResult, error)
}

type ProviderAuthenticationValidator func(context.Context, ProviderConnection) error
type ProviderModelDiscoverer func(context.Context, ProviderConnection) ([]ModelInfo, error)
type ProviderProbeSelector func(ProviderConnection, []ModelInfo) []Target
type ProviderCompletionRuntime func(context.Context, ProviderConnection, Target, map[string]any) (map[string]any, error)

// ProviderAdapter contains the small provider-specific operations used by the
// shared connector. Complete is both the probe and reusable runtime path.
type ProviderAdapter struct {
	ValidateAuthentication ProviderAuthenticationValidator
	DiscoverModels         ProviderModelDiscoverer
	SelectProbeTargets     ProviderProbeSelector
	Complete               ProviderCompletionRuntime
}

// ProviderOrchestrator resolves only explicitly registered adapters. This is
// particularly important for Codex: its personal subscription OAuth flow must
// never fall through to anonymous OpenAI-compatible behavior.
type ProviderOrchestrator struct {
	mu       sync.RWMutex
	adapters map[string]ProviderAdapter
	now      func() time.Time
}

func NewProviderOrchestrator() *ProviderOrchestrator {
	return &ProviderOrchestrator{adapters: make(map[string]ProviderAdapter), now: time.Now}
}

func (o *ProviderOrchestrator) Register(providerID string, adapter ProviderAdapter) error {
	providerID = strings.ToLower(strings.TrimSpace(providerID))
	if providerID == "" {
		return fmt.Errorf("provider id is required")
	}
	if adapter.DiscoverModels == nil || adapter.Complete == nil {
		return fmt.Errorf("provider adapter requires model discovery and completion")
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.adapters[providerID] = adapter
	return nil
}

func (o *ProviderOrchestrator) Connect(ctx context.Context, request ProviderConnectRequest) (ProviderConnectResult, error) {
	result := ProviderConnectResult{Connection: request.Connection}
	if err := request.Connection.Validate(); err != nil {
		return result, err
	}

	providerID := strings.ToLower(strings.TrimSpace(request.Connection.ProviderID))
	if _, err := PublishExactTargets(providerID, CatalogEvidence{}, nil, request.PublicationPolicy); err != nil {
		return result, err
	}
	o.mu.RLock()
	adapter, ok := o.adapters[providerID]
	o.mu.RUnlock()
	if !ok {
		return result, fmt.Errorf("provider adapter %q is not registered", providerID)
	}
	now := o.now
	if now == nil {
		now = time.Now
	}

	if !request.AuthenticationValidated {
		if adapter.ValidateAuthentication == nil {
			return result, fmt.Errorf("provider authentication is not validated")
		}
		if err := adapter.ValidateAuthentication(ctx, request.Connection); err != nil {
			observedAt := now()
			result.Health = classifyProviderError(err, observedAt)
			result.Catalog = CatalogEvidence{Status: CatalogNotProbed}
			return result, err
		}
	}

	models, err := adapter.DiscoverModels(ctx, request.Connection)
	observedAt := now()
	if err != nil {
		result.Catalog = CatalogEvidence{Status: CatalogFailed, ObservedAt: observedAt}
		result.Health = classifyProviderError(err, observedAt)
		return result, err
	}
	result.Catalog = CatalogEvidence{Status: CatalogDiscovered, Models: models, ObservedAt: observedAt}
	if len(models) == 0 {
		result.Catalog.Status = CatalogEmpty
		result.Health = ClassifyProviderFailure(ProviderFailure{StatusCode: http.StatusOK, ObservedAt: observedAt})
		return result, nil
	}

	targets := defaultProbeTargets(providerID, models)
	if adapter.SelectProbeTargets != nil {
		targets = adapter.SelectProbeTargets(request.Connection, models)
	}
	result.Health = ClassifyProviderFailure(ProviderFailure{StatusCode: http.StatusOK, ObservedAt: observedAt})
	for _, target := range targets {
		if target.Provider == "" {
			target.Provider = providerID
		}
		started := now()
		_, probeErr := adapter.Complete(ctx, request.Connection, target, map[string]any{
			"messages":   []any{map[string]any{"role": "user", "content": "Reply with: ok"}},
			"max_tokens": 16,
		})
		finished := now()
		probe := CompletionProbeEvidence{Target: target, Status: CompletionVerified, ObservedAt: finished, Latency: finished.Sub(started)}
		if probeErr != nil {
			probe.Status = CompletionFailed
			result.Health = classifyProviderError(probeErr, finished)
		}
		result.Probes = append(result.Probes, probe)
	}

	result.Targets, err = PublishExactTargets(providerID, result.Catalog, result.Probes, request.PublicationPolicy)
	if err != nil {
		return result, err
	}
	return result, nil
}

func defaultProbeTargets(providerID string, models []ModelInfo) []Target {
	for _, model := range models {
		if modelID := strings.TrimSpace(model.ID); modelID != "" {
			return []Target{{Provider: providerID, Model: modelID}}
		}
	}
	return nil
}
