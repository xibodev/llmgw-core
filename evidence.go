package core

import (
	"context"
	"sync"
)

// Operations named in AccountEvidence.
const (
	EvidenceOperationInvoke     = "invoke"
	EvidenceOperationStream     = "stream"
	EvidenceOperationListModels = "list_models"
)

// AccountEvidence attributes one provider operation to the credential that
// served it, so a product can audit usage and detect a stale or replaced
// credential. It carries no secrets.
type AccountEvidence struct {
	Caller   Caller
	Instance string
	// CredentialKey and CredentialRevision identify the token-store record
	// that served the operation; AccountID is the account it is bound to.
	CredentialKey      string
	CredentialRevision string
	AccountID          string
	Operation          string
	Surface            ModelSurface
	Model              string
	// Outcome is the classified result, including when it was observed.
	Outcome ProviderHealthEvidence
}

// EvidenceSink receives account evidence. Record must neither fail nor delay
// the operation it describes: a sink that persists evidence buffers or drops.
type EvidenceSink interface {
	Record(ctx context.Context, evidence AccountEvidence)
}

// EvidenceSinkFunc adapts a function to EvidenceSink.
type EvidenceSinkFunc func(ctx context.Context, evidence AccountEvidence)

// Record implements EvidenceSink.
func (f EvidenceSinkFunc) Record(ctx context.Context, evidence AccountEvidence) { f(ctx, evidence) }

// MemoryEvidenceSink is the in-memory reference EvidenceSink.
type MemoryEvidenceSink struct {
	mu      sync.Mutex
	records []AccountEvidence
}

// Record implements EvidenceSink.
func (s *MemoryEvidenceSink) Record(_ context.Context, evidence AccountEvidence) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.records = append(s.records, evidence)
}

// Records returns a copy of everything recorded, oldest first.
func (s *MemoryEvidenceSink) Records() []AccountEvidence {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]AccountEvidence(nil), s.records...)
}
