package core

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"sync"
)

// ErrCatalogNotFound reports that no catalog is stored for a key.
var ErrCatalogNotFound = errors.New("core: catalog not found")

// CatalogKey identifies one stored catalog: a provider instance and the key of
// the credential that discovered it. A catalog depends on the credential, not
// on the caller: callers sharing a system credential share its catalog, while
// a personal subscription gets its own. An empty CredentialKey is the catalog
// of a provider discovered without a credential.
type CatalogKey struct {
	Instance      string
	CredentialKey string
}

// CatalogRecord is a stored catalog and its revision. The revision is opaque
// and changes on every save.
type CatalogRecord struct {
	Evidence CatalogEvidence
	Revision string
}

// CatalogStore keeps discovered catalogs where every process that shares the
// store can read them.
//
// Save must be atomic with respect to Load in any process: a reader sees the
// previous record or the new one, never a mix. Because the revision changes on
// every save, a Runtime that cached a catalog can tell another process's
// write from its own. The catalogtest package verifies these guarantees.
type CatalogStore interface {
	// Load returns the stored record, or ErrCatalogNotFound.
	Load(ctx context.Context, key CatalogKey) (CatalogRecord, error)
	// Save stores evidence and returns its new revision.
	Save(ctx context.Context, key CatalogKey, evidence CatalogEvidence) (revision string, err error)
}

// cloneCatalogEvidence copies the model list so stores and callers never
// share it.
func cloneCatalogEvidence(evidence CatalogEvidence) CatalogEvidence {
	if evidence.Models != nil {
		evidence.Models = append([]ModelInfo(nil), evidence.Models...)
	}
	return evidence
}

// MemoryCatalogStore is the in-memory reference CatalogStore.
type MemoryCatalogStore struct {
	mu       sync.Mutex
	records  map[CatalogKey]CatalogRecord
	revision uint64
}

// NewMemoryCatalogStore returns an empty store.
func NewMemoryCatalogStore() *MemoryCatalogStore {
	return &MemoryCatalogStore{records: map[CatalogKey]CatalogRecord{}}
}

func normalizeCatalogKey(key CatalogKey) CatalogKey {
	return CatalogKey{Instance: strings.TrimSpace(key.Instance), CredentialKey: strings.TrimSpace(key.CredentialKey)}
}

// Load implements CatalogStore.
func (s *MemoryCatalogStore) Load(ctx context.Context, key CatalogKey) (CatalogRecord, error) {
	if err := ctx.Err(); err != nil {
		return CatalogRecord{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	record, ok := s.records[normalizeCatalogKey(key)]
	if !ok {
		return CatalogRecord{}, ErrCatalogNotFound
	}
	record.Evidence = cloneCatalogEvidence(record.Evidence)
	return record, nil
}

// Save implements CatalogStore.
func (s *MemoryCatalogStore) Save(ctx context.Context, key CatalogKey, evidence CatalogEvidence) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.revision++
	revision := strconv.FormatUint(s.revision, 10)
	s.records[normalizeCatalogKey(key)] = CatalogRecord{Evidence: cloneCatalogEvidence(evidence), Revision: revision}
	return revision, nil
}
