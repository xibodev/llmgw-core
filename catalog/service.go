// Package catalog keeps discovered provider catalogs in a core.CatalogStore
// the way the gateway keeps its own: reads that serve a stored catalog while
// it is younger than a TTL and otherwise discover it, reads that never
// discover, and discovery fenced against invalidation, so a discovery that
// began before a catalog was invalidated never stores over what came after.
//
// A Service keeps no package state. Its per-key locks and generations live
// in the Service, so keep one per process for a store, as a Runtime does.
package catalog

import (
	"strings"
	"sync"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// DefaultTTL is how long a catalog is served before a read discovers it
// again, when Options.TTL is zero. It is a Runtime's default; the gateway
// sets an hour.
const DefaultTTL = 15 * time.Minute

// Options configures a Service. Every field is optional.
type Options struct {
	// Store keeps the catalogs. Nil keeps them in this Service's memory.
	// A core.ConditionalCatalogStore fences discoveries in every process
	// that shares it; any other store fences them within this Service.
	Store core.CatalogStore
	// TTL is how long a catalog is served before a read discovers it
	// again. Zero or less uses DefaultTTL.
	TTL time.Duration
	// InstanceTTL gives an instance its own TTL. A result of zero or less
	// uses TTL.
	InstanceTTL func(instance string) time.Duration
	// SchemaVersion stamps every catalog the Service stores. A stored
	// catalog stamped otherwise is neither served nor kept as stale: the
	// rules that wrote it are not the rules that would read it.
	SchemaVersion int
	// KeepStale serves the stored catalog, marked stale, beside the error
	// of a discovery that fails, as the gateway does. Without it a failed
	// discovery returns only its error.
	KeepStale bool
	// Now returns the current time. Nil uses time.Now.
	Now func() time.Time
}

// Service reads, discovers and invalidates catalogs. It is safe for
// concurrent use.
type Service struct {
	store       core.CatalogStore
	conditional core.ConditionalCatalogStore
	ttl         time.Duration
	instanceTTL func(string) time.Duration
	schema      int
	keepStale   bool
	now         func() time.Time

	mu   sync.Mutex
	keys map[core.CatalogKey]*keyState
}

// keyState is what the Service keeps for one catalog key.
type keyState struct {
	// discovery serializes the discoveries of the key.
	discovery sync.Mutex
	// write orders stores against invalidations, and guards the
	// generations: hard counts invalidations that fence a discovery in
	// flight, soft those that do not.
	write      sync.Mutex
	hard, soft uint64
}

// New returns a Service.
func New(options Options) *Service {
	s := &Service{
		store: options.Store, ttl: options.TTL, instanceTTL: options.InstanceTTL,
		schema: options.SchemaVersion, keepStale: options.KeepStale, now: options.Now,
		keys: map[core.CatalogKey]*keyState{},
	}
	if s.store == nil {
		s.store = core.NewMemoryCatalogStore()
	}
	s.conditional, _ = s.store.(core.ConditionalCatalogStore)
	if s.ttl <= 0 {
		s.ttl = DefaultTTL
	}
	if s.now == nil {
		s.now = time.Now
	}
	return s
}

// TTL returns how long a catalog of instance is served.
func (s *Service) TTL(instance string) time.Duration {
	if s.instanceTTL != nil {
		if ttl := s.instanceTTL(instance); ttl > 0 {
			return ttl
		}
	}
	return s.ttl
}

// state returns the state of key, which the store normalizes as it does.
func (s *Service) state(key core.CatalogKey) (core.CatalogKey, *keyState) {
	key = core.CatalogKey{Instance: strings.TrimSpace(key.Instance), CredentialKey: strings.TrimSpace(key.CredentialKey)}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, ok := s.keys[key]
	if !ok {
		state = &keyState{}
		s.keys[key] = state
	}
	return key, state
}
