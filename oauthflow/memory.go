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

// expiredRetention is how long an expired flow is kept so its owner is told
// it expired rather than that it never existed.
const expiredRetention = 15 * time.Minute

// MemoryFlowStore is the in-memory reference FlowStore, for tests and
// single-process products. It purges expired flows lazily, when flows are
// created or looked up, and needs no background goroutine.
type MemoryFlowStore struct {
	now func() time.Time

	mu       sync.Mutex
	flows    map[string]Flow
	states   map[[sha256.Size]byte]string
	revision uint64
}

var _ FlowStore = (*MemoryFlowStore)(nil)

// NewMemoryFlowStore returns an empty store that reads time from now; nil
// means time.Now.
func NewMemoryFlowStore(now func() time.Time) *MemoryFlowStore {
	if now == nil {
		now = time.Now
	}
	return &MemoryFlowStore{now: now, flows: map[string]Flow{}, states: map[[sha256.Size]byte]string{}}
}

// Create implements FlowStore.
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
	stateKey := sha256.Sum256([]byte(flow.Secrets.State))
	if flow.Secrets.State != "" {
		if _, exists := s.states[stateKey]; exists {
			return ErrFlowExists
		}
		s.states[stateKey] = flow.ID
	}
	flow = flow.Clone()
	flow.ConsumedAt = time.Time{}
	flow.Revision = s.nextRevisionLocked()
	s.flows[flow.ID] = flow
	return nil
}

// Get implements FlowStore.
func (s *MemoryFlowStore) Get(ctx context.Context, caller core.Caller, id string) (Flow, error) {
	if err := ctx.Err(); err != nil {
		return Flow{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	flow, err := s.ownedLocked(caller, id)
	if err != nil {
		return Flow{}, err
	}
	return flow.Clone(), nil
}

// Consume implements FlowStore.
func (s *MemoryFlowStore) Consume(ctx context.Context, caller core.Caller, id string) (Flow, error) {
	if err := ctx.Err(); err != nil {
		return Flow{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	flow, err := s.ownedLocked(caller, id)
	if err != nil {
		return Flow{}, err
	}
	if flow.Consumed() {
		return Flow{}, ErrFlowNotFound
	}
	consumed := flow.Clone()
	consumed.ConsumedAt = s.now()
	consumed.Revision = s.nextRevisionLocked()
	if flow.Secrets.State != "" {
		delete(s.states, sha256.Sum256([]byte(flow.Secrets.State)))
	}
	stored := consumed
	stored.Secrets = Secrets{}
	s.flows[id] = stored
	return consumed, nil
}

// Update implements FlowStore.
func (s *MemoryFlowStore) Update(ctx context.Context, caller core.Caller, flow Flow) (Flow, error) {
	if err := ctx.Err(); err != nil {
		return Flow{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	stored, err := s.ownedLocked(caller, flow.ID)
	if err != nil {
		return Flow{}, err
	}
	if stored.Revision != flow.Revision {
		return Flow{}, ErrConflict
	}
	stored.Progress = flow.Progress
	stored.Revision = s.nextRevisionLocked()
	s.flows[stored.ID] = stored
	return stored.Clone(), nil
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
	id, ok := s.states[sha256.Sum256([]byte(state))]
	if !ok {
		return core.Caller{}, "", ErrFlowNotFound
	}
	flow, err := s.ownedLocked(s.flows[id].Caller, id)
	if err != nil {
		return core.Caller{}, "", err
	}
	return flow.Caller, flow.ID, nil
}

// ownedLocked returns caller's stored flow. The owner check comes first, so
// another caller cannot tell an expired flow from a missing one.
func (s *MemoryFlowStore) ownedLocked(caller core.Caller, id string) (Flow, error) {
	flow, ok := s.flows[id]
	if !ok || flow.Caller != caller {
		return Flow{}, ErrFlowNotFound
	}
	now := s.now()
	if !flow.Expired(now) {
		return flow, nil
	}
	if now.Sub(flow.ExpiresAt) >= expiredRetention {
		s.deleteLocked(flow)
		return Flow{}, ErrFlowNotFound
	}
	return Flow{}, ErrFlowExpired
}

func (s *MemoryFlowStore) purgeLocked() {
	now := s.now()
	for _, flow := range s.flows {
		if now.Sub(flow.ExpiresAt) >= expiredRetention {
			s.deleteLocked(flow)
		}
	}
}

func (s *MemoryFlowStore) deleteLocked(flow Flow) {
	delete(s.flows, flow.ID)
	if flow.Secrets.State != "" {
		delete(s.states, sha256.Sum256([]byte(flow.Secrets.State)))
	}
}

func (s *MemoryFlowStore) nextRevisionLocked() string {
	s.revision++
	return strconv.FormatUint(s.revision, 10)
}
