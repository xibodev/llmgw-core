package catalog

import (
	"context"

	core "github.com/xibodev/llmgw-core"
)

// Find returns the row of model in record.
func Find(record core.CatalogRecord, model string) (core.ModelInfo, bool) {
	for _, row := range record.Evidence.Models {
		if row.ID == model {
			return row, true
		}
	}
	return core.ModelInfo{}, false
}

// Lookup returns the row of model in the catalog Read serves, which may
// discover it. The error is the read's: with Options.KeepStale a row may
// come from a stale catalog beside it.
func (s *Service) Lookup(ctx context.Context, request Request, model string) (core.ModelInfo, bool, error) {
	read := s.Read(ctx, request)
	row, found := Find(read.Record, model)
	return row, found, read.Err
}

// CachedLookup returns the row of model in the stored catalog of key. It
// never discovers, so dispatch and transport planning can use it without
// adding an upstream call to a request.
func (s *Service) CachedLookup(ctx context.Context, key core.CatalogKey, model string) (core.ModelInfo, bool) {
	return Find(s.Cached(ctx, key).Record, model)
}
