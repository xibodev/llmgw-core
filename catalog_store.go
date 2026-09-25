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

// ErrCatalogConflict reports a conditional save whose expected revision is
// no longer the key's revision.
var ErrCatalogConflict = errors.New("core: catalog revision conflict")

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

// ConditionalCatalogStore is a CatalogStore that can fence writes and
// forget catalogs. A catalog.Service uses it so that a discovery which ran
// across an invalidation, in this process or another, cannot store what
// the invalidation removed.
//
// A key's revision is its record's, or the revision Delete left, or empty
// for a key never saved. Load of a deleted key returns ErrCatalogNotFound
// together with a record that carries only that revision, so a caller can
// expect it. The catalogtest package verifies these guarantees.
type ConditionalCatalogStore interface {
	CatalogStore
	// SaveIf stores evidence and returns its new revision only while the
	// key's revision is expected, atomically. Otherwise it changes nothing
	// and returns ErrCatalogConflict.
	SaveIf(ctx context.Context, key CatalogKey, evidence CatalogEvidence, expected string) (revision string, err error)
	// Delete forgets the key's record and gives the key a new revision, so
	// a SaveIf that expects any earlier state conflicts. Deleting a key
	// that holds no record succeeds.
	Delete(ctx context.Context, key CatalogKey) error
}

// cloneCatalogEvidence copies the model list so stores and callers never
// share it.
func cloneCatalogEvidence(evidence CatalogEvidence) CatalogEvidence {
	if evidence.Models != nil {
		evidence.Models = append([]ModelInfo(nil), evidence.Models...)
	}
	return evidence
}

// MemoryCatalogStore is the in-memory reference CatalogStore. It is a
// ConditionalCatalogStore too.
type MemoryCatalogStore struct {
	mu      sync.Mutex
	records map[CatalogKey]CatalogRecord
	// deleted holds the revision Delete left on each key without a record.
	deleted  map[CatalogKey]string
	revision uint64
}

var _ ConditionalCatalogStore = (*MemoryCatalogStore)(nil)

// NewMemoryCatalogStore returns an empty store.
func NewMemoryCatalogStore() *MemoryCatalogStore {
	return &MemoryCatalogStore{records: map[CatalogKey]CatalogRecord{}, deleted: map[CatalogKey]string{}}
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
	key = normalizeCatalogKey(key)
	record, ok := s.records[key]
	if !ok {
		return CatalogRecord{Revision: s.deleted[key]}, ErrCatalogNotFound
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
	return s.saveLocked(normalizeCatalogKey(key), evidence), nil
}

// SaveIf implements ConditionalCatalogStore.
func (s *MemoryCatalogStore) SaveIf(ctx context.Context, key CatalogKey, evidence CatalogEvidence, expected string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	key = normalizeCatalogKey(key)
	if s.revisionLocked(key) != expected {
		return "", ErrCatalogConflict
	}
	return s.saveLocked(key, evidence), nil
}

// Delete implements ConditionalCatalogStore.
func (s *MemoryCatalogStore) Delete(ctx context.Context, key CatalogKey) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	key = normalizeCatalogKey(key)
	delete(s.records, key)
	s.deleted[key] = s.nextRevisionLocked()
	return nil
}

func (s *MemoryCatalogStore) revisionLocked(key CatalogKey) string {
	if record, ok := s.records[key]; ok {
		return record.Revision
	}
	return s.deleted[key]
}

func (s *MemoryCatalogStore) saveLocked(key CatalogKey, evidence CatalogEvidence) string {
	revision := s.nextRevisionLocked()
	delete(s.deleted, key)
	s.records[key] = CatalogRecord{Evidence: cloneCatalogEvidence(evidence), Revision: revision}
	return revision
}

func (s *MemoryCatalogStore) nextRevisionLocked() string {
	s.revision++
	return strconv.FormatUint(s.revision, 10)
}
