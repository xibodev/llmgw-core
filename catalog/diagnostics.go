package catalog

import (
	"errors"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

// ErrStateChanged reports a discovery an invalidation fenced out: the
// catalog was invalidated while it ran, so its result was not stored.
var ErrStateChanged = errors.New("catalog: the catalog changed during discovery")

// Status summarizes a read, as the gateway's catalog diagnostics do.
type Status string

const (
	// StatusSynced is a catalog with models.
	StatusSynced Status = "synced"
	// StatusNotSynced is no catalog: nothing stored, and nothing discovered.
	StatusNotSynced Status = "not_synced"
	// StatusEmpty is a catalog discovered without models.
	StatusEmpty Status = "empty"
	// StatusError is a read that failed, stale rows or not.
	StatusError Status = "error"
)

// Failure codes for errors that are not catalog failures of a provider,
// whose providers.CatalogError code a read reports as is.
const (
	// FailureStateChanged is ErrStateChanged.
	FailureStateChanged = "state_changed"
	// FailureFailed is any other error.
	FailureFailed = "failed"
)

// Diagnostics describe a read without its rows. They never quote an error:
// Detail is a *core.ProviderError's message, which is safe to show, or a
// fixed description.
type Diagnostics struct {
	Status Status `json:"status"`
	// FailureCode is a providers.CatalogError code without the gateway's
	// catalog_ prefix, or one of the Failure codes.
	FailureCode    string `json:"failure_code,omitempty"`
	Detail         string `json:"detail,omitempty"`
	UpstreamStatus int    `json:"upstream_status,omitempty"`
	// Stale reports a catalog served past its TTL.
	Stale bool `json:"stale"`
	// FromCache reports a catalog served as stored, not just discovered.
	FromCache bool `json:"from_cache"`
}

// Failed returns the read of a catalog that could not be read at all, such
// as one whose provider cannot be built or whose credential is unavailable.
func Failed(err error) Read {
	code, detail, status := failure(err)
	return Read{Err: err, Diagnostics: Diagnostics{Status: StatusError, FailureCode: code, Detail: detail, UpstreamStatus: status}}
}

// failure reduces err to what diagnostics may show.
func failure(err error) (code, detail string, status int) {
	if errors.Is(err, ErrStateChanged) {
		return FailureStateChanged, "Provider configuration changed during catalog refresh.", 0
	}
	code, detail, status = FailureFailed, "Provider catalog failed.", core.ClassifyError(err).StatusCode
	var catalogErr *providers.CatalogError
	if errors.As(err, &catalogErr) {
		code, status = catalogErr.Code, catalogErr.Status
	}
	var providerErr *core.ProviderError
	if errors.As(err, &providerErr) && providerErr.Message != "" {
		detail = providerErr.Message
	}
	return code, detail, status
}
