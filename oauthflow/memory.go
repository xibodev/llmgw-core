package oauthflow

import (
	"context"
	"crypto/sha256"
	"errors"
	"strconv"
	"strings"
	"sync"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// defaultExpiredRetention is how long an expired flow is kept when
// MemoryFlowStoreOptions names no retention.
const defaultExpiredRetention = 15 * time.Minute

// MemoryFlowStoreOptions configure a MemoryFlowStore. The zero value is a
// store with the real clock, no cap and the default retention.
type MemoryFlowStoreOptions struct {
	// Now is the clock; nil means time.Now.
	Now func() time.Time
	// MaxFlowsPerCaller caps a caller's pending flows, those neither
	// consumed nor expired. Creating one more evicts that caller's oldest
	// pending flow, by CreatedAt and then insertion order, as the gateway
	// does; an evicted flow answers ErrFlowNotFound. Zero means no cap.
	MaxFlowsPerCaller int
	// ExpiredRetention is how long an expired flow is kept, without its
	// secrets, so its owner is told it expired rather than that it never
	// existed. Zero means 15 minutes.
	ExpiredRetention time.Duration
}

// MemoryFlowStore is the in-memory reference FlowStore, for tests and
// single-process products. It purges expired flows lazily, when flows are
// created or looked up, and needs no background goroutine.
type MemoryFlowStore struct {
	now       func() time.Time
	maxFlows  int
	retention time.Duration

	mu       sync.Mutex
	flows    map[string]*memoryEntry
	states   map[[sha256.Size]byte]string
	revision uint64
	sequence uint64
}

// memoryEntry is a stored flow and what the store needs to manage it.
type memoryEntry struct {
	flow Flow
	// sequence is the insertion order, which breaks CreatedAt ties when
	// the cap evicts.
	sequence uint64
	// stateKey is the hash of the flow's state; indexed says whether the
	// state index still maps it to the flow.
	stateKey [sha256.Size]byte
	indexed  bool
}

var _ FlowStore = (*MemoryFlowStore)(nil)

// NewMemoryFlowStore returns an empty store.
func NewMemoryFlowStore(options MemoryFlowStoreOptions) *MemoryFlowStore {
	store := &MemoryFlowStore{
		now: options.Now, maxFlows: options.MaxFlowsPerCaller, retention: options.ExpiredRetention,
		flows: map[string]*memoryEntry{}, states: map[[sha256.Size]byte]string{},
	}
	if store.now == nil {
		store.now = time.Now
	}
	if store.retention <= 0 {
		store.retention = defaultExpiredRetention
	}
	return store
}

// Create implements FlowStore. A failed Create evicts nothing.
func (s *MemoryFlowStore) Create(ctx context.Context, flow Flow) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if strings.TrimSpace(flow.ID) == "" {
		return errors.New("oauthflow: flow id is required")
	}
	if err := flow.Caller.Validate(); err != nil {
		return err
	}
	if flow.ExpiresAt.IsZero() {
		return errors.New("oauthflow: flow expiry is required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.purgeLocked()
	if _, exists := s.flows[flow.ID]; exists {
		return ErrFlowExists
	}
	entry := &memoryEntry{flow: flow.Clone()}
	if flow.Secrets.State != "" {
		entry.stateKey, entry.indexed = sha256.Sum256([]byte(flow.Secrets.State)), true
		if _, exists := s.states[entry.stateKey]; exists {
			return ErrFlowExists
		}
	}
	s.evictLocked(flow.Caller)
	if entry.indexed {
		s.states[entry.stateKey] = flow.ID
	}
	s.sequence++
	entry.sequence = s.sequence
	entry.flow.ConsumedAt = time.Time{}
	entry.flow.Revision = s.nextRevisionLocked()
	s.flows[flow.ID] = entry
	return nil
}

// Get implements FlowStore.
func (s *MemoryFlowStore) Get(ctx context.Context, caller core.Caller, id string) (Flow, error) {
	if err := ctx.Err(); err != nil {
		return Flow{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, err := s.ownedLocked(caller, id)
	if err != nil {
		return Flow{}, err
	}
	return entry.flow.Clone(), nil
}

// Consume implements FlowStore.
func (s *MemoryFlowStore) Consume(ctx context.Context, caller core.Caller, id string) (Flow, error) {
	if err := ctx.Err(); err != nil {
		return Flow{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, err := s.ownedLocked(caller, id)
	if err != nil {
		return Flow{}, err
	}
	if entry.flow.Consumed() {
		return Flow{}, ErrFlowNotFound
	}
	consumed := entry.flow.Clone()
	consumed.ConsumedAt = s.now()
	consumed.Revision = s.nextRevisionLocked()
	s.unindexLocked(entry)
	entry.flow = consumed.Clone()
	entry.flow.Secrets = Secrets{}
	return consumed, nil
}

// Update implements FlowStore.
func (s *MemoryFlowStore) Update(ctx context.Context, caller core.Caller, flow Flow) (Flow, error) {
	if err := ctx.Err(); err != nil {
		return Flow{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, err := s.ownedLocked(caller, flow.ID)
	if err != nil {
		return Flow{}, err
	}
	if entry.flow.Revision != flow.Revision {
		return Flow{}, ErrConflict
	}
	entry.flow.Progress = flow.Progress
	entry.flow.Revision = s.nextRevisionLocked()
	return entry.flow.Clone(), nil
}

// ResolveState implements FlowStore.
func (s *MemoryFlowStore) ResolveState(ctx context.Context, state string) (core.Caller, string, error) {
	if err := ctx.Err(); err != nil {
		return core.Caller{}, "", err
	}
	if state == "" {
		return core.Caller{}, "", ErrFlowNotFound
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.flows[s.states[sha256.Sum256([]byte(state))]]
	if !ok || !entry.indexed {
		return core.Caller{}, "", ErrFlowNotFound
	}
	if _, err := s.ownedLocked(entry.flow.Caller, entry.flow.ID); err != nil {
		return core.Caller{}, "", err
	}
	return entry.flow.Caller, entry.flow.ID, nil
}

// ownedLocked returns caller's stored flow. The owner check comes first, so
// another caller cannot tell an expired flow from a missing one.
func (s *MemoryFlowStore) ownedLocked(caller core.Caller, id string) (*memoryEntry, error) {
	entry, ok := s.flows[id]
	if !ok || entry.flow.Caller != caller {
		return nil, ErrFlowNotFound
	}
	now := s.now()
	if !entry.flow.Expired(now) {
		return entry, nil
	}
	if now.Sub(entry.flow.ExpiresAt) >= s.retention {
		s.deleteLocked(entry)
		return nil, ErrFlowNotFound
	}
	s.expireLocked(entry)
	return nil, ErrFlowExpired
}

// expireLocked wipes an expired flow's secrets: it can never be consumed, so
// its verifier and state must not linger. The state index keeps only the
// state's hash, so a late callback still learns that the flow expired.
func (s *MemoryFlowStore) expireLocked(entry *memoryEntry) {
	entry.flow.Secrets = Secrets{}
}

// evictLocked makes room for one more pending flow of caller under the cap.
func (s *MemoryFlowStore) evictLocked(caller core.Caller) {
	if s.maxFlows <= 0 {
		return
	}
	now := s.now()
	for {
		count := 0
		var oldest *memoryEntry
		for _, entry := range s.flows {
			if entry.flow.Caller != caller || entry.flow.Consumed() || entry.flow.Expired(now) {
				continue
			}
			count++
			if oldest == nil || entry.flow.CreatedAt.Before(oldest.flow.CreatedAt) ||
				entry.flow.CreatedAt.Equal(oldest.flow.CreatedAt) && entry.sequence < oldest.sequence {
				oldest = entry
			}
		}
		if count < s.maxFlows {
			return
		}
		s.deleteLocked(oldest)
	}
}

// purgeLocked deletes flows past their retention and wipes the secrets of
// the other expired ones.
func (s *MemoryFlowStore) purgeLocked() {
	now := s.now()
	for _, entry := range s.flows {
		switch {
		case !entry.flow.Expired(now):
		case now.Sub(entry.flow.ExpiresAt) >= s.retention:
			s.deleteLocked(entry)
		default:
			s.expireLocked(entry)
		}
	}
}

func (s *MemoryFlowStore) deleteLocked(entry *memoryEntry) {
	delete(s.flows, entry.flow.ID)
	s.unindexLocked(entry)
}

func (s *MemoryFlowStore) unindexLocked(entry *memoryEntry) {
	if entry.indexed {
		delete(s.states, entry.stateKey)
		entry.indexed = false
	}
}

func (s *MemoryFlowStore) nextRevisionLocked() string {
	s.revision++
	return strconv.FormatUint(s.revision, 10)
}
