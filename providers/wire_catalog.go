package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// Catalog error codes. A catalog failure is a *core.ProviderError whose
// cause is a *CatalogError with one of these codes: the gateway's catalog
// failure codes without their catalog_ prefix, so a product maps a failure
// without reading its message.
const (
	// CatalogCodeHTTPError is a status other than success, 401 or 403.
	CatalogCodeHTTPError = "http_error"
	// CatalogCodeAuthenticationFailed is the upstream rejecting the
	// credential with 401 or 403.
	CatalogCodeAuthenticationFailed = "authentication_failed"
	// CatalogCodeNotDiscoverable is a catalog that cannot be listed, such
	// as one over the size limit.
	CatalogCodeNotDiscoverable = "not_discoverable"
	// CatalogCodeTransportError is a catalog request that got no complete
	// answer.
	CatalogCodeTransportError = "transport_error"
	// CatalogCodeInvalidJSON is an answer that is not JSON.
	CatalogCodeInvalidJSON = "invalid_json"
	// CatalogCodeInvalidShape is JSON that is not a catalog: not an object,
	// without its model array, or with a row that names no model.
	CatalogCodeInvalidShape = "invalid_shape"
)

// decodeCatalogResponse validates one catalog answer as the gateway does
// and closes it. A refused answer is not read. An accepted one is read
// within catalogMaxResponseBytes and must be a JSON object whose field is
// an array of objects, each naming its model by one of identities with a
// nonblank string; an identity of another type rejects the row even beside
// a good one. Every row is checked before any is used, so a malformed
// page never replaces a usable catalog with a partial one.
func decodeCatalogResponse(ctx context.Context, response *http.Response, now time.Time, field string, identities ...string) (map[string]any, error) {
	defer response.Body.Close()
	status := response.StatusCode
	if status < 200 || status >= 300 {
		code := CatalogCodeHTTPError
		detail := fmt.Sprintf("Provider catalog returned HTTP %d.", status)
		if status == http.StatusUnauthorized || status == http.StatusForbidden {
			code = CatalogCodeAuthenticationFailed
			detail = "Provider credential was rejected by the catalog API."
		}
		return nil, catalogFailure(ctx, code, detail, status, retryAfterDelay(response.Header.Get("Retry-After"), now), nil)
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, catalogMaxResponseBytes+1))
	if len(raw) > catalogMaxResponseBytes {
		return nil, catalogFailure(ctx, CatalogCodeNotDiscoverable, "Provider catalog response exceeded the size limit.", status, 0, nil)
	}
	if err != nil {
		return nil, catalogFailure(ctx, CatalogCodeTransportError, "Provider catalog response could not be read.", status, 0, err)
	}
	var decoded any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, catalogFailure(ctx, CatalogCodeInvalidJSON, "Provider catalog response was not valid JSON.", status, 0, err)
	}
	body, ok := decoded.(map[string]any)
	if !ok {
		return nil, catalogFailure(ctx, CatalogCodeInvalidShape, "Provider catalog response was not a JSON object.", status, 0, nil)
	}
	items, ok := body[field].([]any)
	if !ok {
		return nil, catalogFailure(ctx, CatalogCodeInvalidShape, "Provider catalog response did not contain a "+field+" array.", status, 0, nil)
	}
	for _, item := range items {
		if !catalogRowIdentified(item, identities) {
			return nil, catalogFailure(ctx, CatalogCodeInvalidShape, "Provider catalog response contained an invalid model row.", status, 0, nil)
		}
	}
	return body, nil
}

// catalogRowIdentified reports a row that names its model by a nonblank
// string under one of identities, and none of another type.
func catalogRowIdentified(item any, identities []string) bool {
	row, ok := item.(map[string]any)
	if !ok {
		return false
	}
	identified := false
	for _, key := range identities {
		value, exists := row[key]
		if !exists {
			continue
		}
		id, ok := value.(string)
		if !ok {
			return false
		}
		identified = identified || strings.TrimSpace(id) != ""
	}
	return identified
}

// catalogFailure reports a catalog that could not be listed, with a
// *CatalogError cause that keeps its code, detail and the answer's status.
// A refusal, http_error or authentication_failed, is classified by its
// status, so a Runtime refreshes a credential the catalog rejects with
// 401. A transport error may repeat, and once the caller gave up permits
// nothing. Any other code is an answer the upstream sent that cannot be
// used. detail is the message, so it must be safe to show.
func catalogFailure(ctx context.Context, code, detail string, status int, retryAfter time.Duration, cause error) *core.ProviderError {
	failure := &core.ProviderError{
		Message: detail,
		Cause:   &CatalogError{Code: code, Detail: detail, Status: status, RetryAfter: retryAfter, Cause: cause},
	}
	switch code {
	case CatalogCodeHTTPError, CatalogCodeAuthenticationFailed:
		failure.Class = core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: status}).ErrorClass
		failure.Classification = statusClassification(status, retryAfter)
	case CatalogCodeTransportError:
		failure.Class = core.ProviderErrorTransport
		if ctx.Err() == nil {
			failure.Classification = core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}
		}
	default:
		failure.Class = core.ProviderErrorUpstream
		failure.Classification = core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}
	}
	return failure
}
