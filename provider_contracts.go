package core

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// ProviderConnectionKind describes who owns and may use a provider connection.
type ProviderConnectionKind string

const (
	ProviderConnectionSystem               ProviderConnectionKind = "system"
	ProviderConnectionPersonalSubscription ProviderConnectionKind = "personal_subscription"
	ProviderConnectionAnonymous            ProviderConnectionKind = "anonymous"
)

// ProviderAuthKind describes how a provider connection authenticates.
type ProviderAuthKind string

const (
	ProviderAuthAPIKey      ProviderAuthKind = "api_key"
	ProviderAuthOAuthDevice ProviderAuthKind = "oauth_device"
	ProviderAuthAnonymous   ProviderAuthKind = "anonymous"
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
	if f.StatusCode >= 200 && f.StatusCode < 400 && f.Err == nil {
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

func parseRetryAfter(value string, observedAt time.Time) time.Duration {
	value = strings.TrimSpace(value)
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil && seconds >= 0 {
		return time.Duration(seconds) * time.Second
	}
	if when, err := time.Parse(time.RFC1123, value); err == nil && !observedAt.IsZero() && when.After(observedAt) {
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
