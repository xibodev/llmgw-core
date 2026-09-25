package catalog

import (
	"context"
	"errors"

	core "github.com/xibodev/llmgw-core"
)

// Invalidation says whether invalidating a catalog fences the discoveries
// in flight.
type Invalidation int

const (
	// Hard forgets the catalog and fences every discovery in flight, as a
	// revoked, replaced or reconfigured credential needs: a slow discovery
	// never resurrects rows the old credential listed.
	Hard Invalidation = iota
	// Soft forgets the catalog without fencing the discoveries in flight,
	// as a provider's own token refresh or metadata write needs: the
	// operation that caused it keeps its result.
	Soft
)

// Invalidate forgets the catalog of key. A ConditionalCatalogStore deletes
// it; any other store keeps a record that was never probed in its place,
// which no read serves.
func (s *Service) Invalidate(ctx context.Context, key core.CatalogKey, mode Invalidation) error {
	key, state := s.state(key)
	state.write.Lock()
	defer state.write.Unlock()
	if mode == Soft {
		state.soft++
	} else {
		state.hard++
	}
	if s.conditional != nil {
		return s.conditional.Delete(ctx, key)
	}
	_, err := s.store.Save(ctx, key, core.CatalogEvidence{Status: core.CatalogNotProbed, SchemaVersion: s.schema})
	return err
}

// Refresh discovers the catalog of the request's key now, fresh or not, and
// stores it unless an invalidation fenced the discovery. A failed discovery
// leaves the stored catalog as it was.
func (s *Service) Refresh(ctx context.Context, request Request) (core.CatalogRecord, error) {
	key, state := s.state(request.Key)
	request.Key = key
	state.discovery.Lock()
	defer state.discovery.Unlock()
	_, _, fence, err := s.load(ctx, key)
	if err != nil {
		return core.CatalogRecord{}, err
	}
	return s.refresh(ctx, request, state, fence)
}

type generations struct{ hard, soft uint64 }

func (k *keyState) generations() generations {
	k.write.Lock()
	defer k.write.Unlock()
	return generations{hard: k.hard, soft: k.soft}
}

// refresh discovers and stores, expecting the stored revision to be fence.
func (s *Service) refresh(ctx context.Context, request Request, state *keyState, fence string) (core.CatalogRecord, error) {
	started := state.generations()
	models, err := request.Discover(ctx)
	if err != nil {
		return core.CatalogRecord{}, err
	}
	status := core.CatalogDiscovered
	if len(models) == 0 {
		status = core.CatalogEmpty
	}
	evidence := core.CatalogEvidence{Status: status, Models: models, ObservedAt: s.now(), SchemaVersion: s.schema}
	return s.save(ctx, request, state, started, fence, evidence)
}

// save stores evidence unless a hard invalidation since started fenced it.
// A soft invalidation, or a hard one Rebase accepts, fences nothing, so the
// save expects the latest revision instead. A save another write or an
// invalidation elsewhere beat returns the catalog stored now, if any.
func (s *Service) save(ctx context.Context, request Request, state *keyState, started generations, fence string, evidence core.CatalogEvidence) (core.CatalogRecord, error) {
	state.write.Lock()
	defer state.write.Unlock()
	latest := state.soft != started.soft
	switch {
	case state.hard == started.hard:
	case state.hard == started.hard+1 && request.Rebase != nil && request.Rebase():
		latest = true
	default:
		return core.CatalogRecord{}, ErrStateChanged
	}
	if latest {
		_, _, current, err := s.load(ctx, request.Key)
		if err != nil {
			return core.CatalogRecord{}, err
		}
		fence = current
	}
	var revision string
	var err error
	if s.conditional != nil {
		revision, err = s.conditional.SaveIf(ctx, request.Key, evidence, fence)
	} else {
		revision, err = s.store.Save(ctx, request.Key, evidence)
	}
	if errors.Is(err, core.ErrCatalogConflict) {
		current, found, _, loadErr := s.load(ctx, request.Key)
		switch {
		case loadErr != nil:
			return core.CatalogRecord{}, loadErr
		case !found:
			return core.CatalogRecord{}, ErrStateChanged
		}
		return current, nil
	}
	if err != nil {
		return core.CatalogRecord{}, err
	}
	return core.CatalogRecord{Evidence: evidence, Revision: revision}, nil
}
