package catalog

import (
	"context"
	"errors"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// Request asks for the catalog of one key.
type Request struct {
	Key core.CatalogKey
	// Discover lists the catalog upstream. A read calls it when the stored
	// catalog is missing or no longer fresh.
	Discover func(ctx context.Context) ([]core.ModelInfo, error)
	// NotBefore makes a catalog observed before it no longer fresh, as a
	// Runtime does with catalogs observed before its settings changed.
	NotBefore time.Time
	// Rebase is asked whether a discovery that exactly one hard
	// invalidation fenced may store anyway, as the gateway lets an
	// operation whose own credential write caused the invalidation keep
	// its result. Nil never rebases. It must not call the Service.
	Rebase func() bool
}

// Read is the outcome of reading one catalog.
type Read struct {
	// Record is the catalog served, or the zero record when there is none.
	Record core.CatalogRecord
	// Diagnostics describe the read without its rows.
	Diagnostics Diagnostics
	// Err is why the read could not discover or serve a catalog. With
	// Options.KeepStale, Record may hold the stale catalog beside it.
	Err error
}

// Read returns the catalog of the request's key: the stored one while it is
// fresh, which is younger than the TTL and observed no earlier than
// NotBefore, and otherwise a new discovery. The catalog returned is the one
// stored once the discovery is over, because another write or an
// invalidation may have superseded it. Discoveries of one key are
// serialized.
func (s *Service) Read(ctx context.Context, request Request) Read {
	key, state := s.state(request.Key)
	request.Key = key
	state.discovery.Lock()
	defer state.discovery.Unlock()
	record, found, fence, err := s.load(ctx, key)
	if err != nil {
		return s.result(key, core.CatalogRecord{}, false, false, err)
	}
	if found && s.fresh(key, record, request.NotBefore) {
		return s.result(key, record, true, true, nil)
	}
	_, err = s.refresh(ctx, request, state, fence)
	current, found, _, loadErr := s.load(ctx, key)
	switch {
	case loadErr != nil:
		return s.result(key, core.CatalogRecord{}, false, false, loadErr)
	case err != nil && found && s.keepStale:
		return s.result(key, current, true, true, err)
	case err != nil:
		return s.result(key, core.CatalogRecord{}, false, false, err)
	case !found:
		return s.result(key, core.CatalogRecord{}, false, false, ErrStateChanged)
	}
	return s.result(key, current, true, false, nil)
}

// Cached returns the stored catalog of key without discovering it, fresh or
// not, so a read path stays safe while an upstream is slow or down.
func (s *Service) Cached(ctx context.Context, key core.CatalogKey) Read {
	key, _ = s.state(key)
	record, found, _, err := s.load(ctx, key)
	if !found {
		record = core.CatalogRecord{}
	}
	return s.result(key, record, found, true, err)
}

// load reads key. found reports a catalog the Service serves: one this
// schema stamped, of a discovery. fence is the revision a store expects,
// the unusable record's included.
func (s *Service) load(ctx context.Context, key core.CatalogKey) (record core.CatalogRecord, found bool, fence string, err error) {
	record, err = s.store.Load(ctx, key)
	if errors.Is(err, core.ErrCatalogNotFound) {
		return core.CatalogRecord{}, false, record.Revision, nil
	}
	if err != nil {
		return core.CatalogRecord{}, false, "", err
	}
	status := record.Evidence.Status
	found = (status == core.CatalogDiscovered || status == core.CatalogEmpty) && record.Evidence.SchemaVersion == s.schema
	if !found {
		return core.CatalogRecord{}, false, record.Revision, nil
	}
	return record, true, record.Revision, nil
}

func (s *Service) fresh(key core.CatalogKey, record core.CatalogRecord, notBefore time.Time) bool {
	observed := record.Evidence.ObservedAt
	if observed.IsZero() || (!notBefore.IsZero() && observed.Before(notBefore)) {
		return false
	}
	return s.now().Sub(observed) < s.TTL(key.Instance)
}

func (s *Service) result(key core.CatalogKey, record core.CatalogRecord, found, fromCache bool, err error) Read {
	read := Read{Record: record, Err: err, Diagnostics: Diagnostics{Status: StatusSynced, FromCache: fromCache}}
	switch {
	case err != nil:
		read.Diagnostics.Status = StatusError
		read.Diagnostics.FailureCode, read.Diagnostics.Detail, read.Diagnostics.UpstreamStatus = failure(err)
	case !found:
		read.Diagnostics.Status = StatusNotSynced
	case len(record.Evidence.Models) == 0:
		read.Diagnostics.Status = StatusEmpty
	}
	observed := record.Evidence.ObservedAt
	read.Diagnostics.Stale = found && !observed.IsZero() && s.now().Sub(observed) >= s.TTL(key.Instance)
	return read
}
