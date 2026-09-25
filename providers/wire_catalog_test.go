package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// catalogPaddingReader pads indefinitely, so the decoder, not the fixture,
// must stop reading.
type catalogPaddingReader struct {
	read   int
	closed bool
}

func (r *catalogPaddingReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = ' '
	}
	r.read += len(p)
	return len(p), nil
}

func (r *catalogPaddingReader) Close() error {
	r.closed = true
	return nil
}

func catalogCode(t *testing.T, err error) *CatalogError {
	t.Helper()
	var catalogErr *CatalogError
	var failure *core.ProviderError
	if !errors.As(err, &failure) || !errors.As(err, &catalogErr) || failure.Message != catalogErr.Detail {
		t.Fatalf("error = %#v, want a *core.ProviderError over a *CatalogError", err)
	}
	return catalogErr
}

// Ported from the gateway's catalog_http_test.go, with the classification a
// Runtime reads.
func TestDecodeCatalogResponseBoundsReadAndPreservesStatus(t *testing.T) {
	t.Parallel()
	for _, status := range []int{200, 401, 403, 429, 503} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			body := &catalogPaddingReader{}
			response := &http.Response{StatusCode: status, Header: http.Header{"Retry-After": {"30"}}, Body: body}
			models, err := decodeCatalogResponse(context.Background(), response, time.Now(), "data", "id")
			wantCode, wantRead := CatalogCodeHTTPError, 0
			if status == 200 {
				wantCode, wantRead = CatalogCodeNotDiscoverable, catalogMaxResponseBytes+1
			} else if status == 401 || status == 403 {
				wantCode = CatalogCodeAuthenticationFailed
			}
			catalogErr := catalogCode(t, err)
			if models != nil || catalogErr.Code != wantCode || catalogErr.Status != status ||
				body.read != wantRead || !body.closed || catalogErr.Detail == "" {
				t.Fatalf("response not safely bounded: models=%v err=%v read=%d closed=%v", models, err, body.read, body.closed)
			}
			classification := core.ClassifyError(err)
			switch status {
			case 200:
				if classification != (core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}) {
					t.Fatalf("oversized catalog classification = %+v", classification)
				}
			default:
				if classification != statusClassification(status, 30*time.Second) {
					t.Fatalf("status %d classification = %+v", status, classification)
				}
			}
		})
	}
}

// Ported from the size boundary of the gateway's
// TestCatalogResponseSizeBoundaryAndCache.
func TestDecodeCatalogResponseSizeBoundary(t *testing.T) {
	t.Parallel()
	const document = `{"data":[{"id":"fixture-model"}],"private":"fixture-secret"}`
	for _, oversized := range []bool{false, true} {
		size := catalogMaxResponseBytes - len(document)
		if oversized {
			size++
		}
		body := io.NopCloser(io.MultiReader(strings.NewReader(document), io.LimitReader(&catalogPaddingReader{}, int64(size))))
		decoded, err := decodeCatalogResponse(context.Background(), &http.Response{StatusCode: 200, Body: body}, time.Now(), "data", "id")
		if !oversized {
			if err != nil || len(decoded["data"].([]any)) != 1 {
				t.Fatalf("exact-boundary catalog rejected: %v", err)
			}
			continue
		}
		catalogErr := catalogCode(t, err)
		if catalogErr.Code != CatalogCodeNotDiscoverable || catalogErr.Status != 200 ||
			catalogErr.Detail != "Provider catalog response exceeded the size limit." || strings.Contains(err.Error(), "fixture-secret") {
			t.Fatalf("oversized catalog: %#v", catalogErr)
		}
	}
}

func TestDecodeCatalogResponseValidatesEveryRow(t *testing.T) {
	t.Parallel()
	for body, want := range map[string]string{
		`{"data":[`:                               CatalogCodeInvalidJSON,
		`[{"id":"fixture-model"}]`:                CatalogCodeInvalidShape,
		`{"models":[{"id":"fixture-model"}]}`:     CatalogCodeInvalidShape,
		`{"data":{"id":"fixture-model"}}`:         CatalogCodeInvalidShape,
		`{"data":["fixture-model"]}`:              CatalogCodeInvalidShape,
		`{"data":[{"object":"model"}]}`:           CatalogCodeInvalidShape,
		`{"data":[{"id":" "}]}`:                   CatalogCodeInvalidShape,
		`{"data":[{"id":7,"name":"fixture"}]}`:    CatalogCodeInvalidShape,
		`{"data":[{"id":"a"},{"name":null}]}`:     CatalogCodeInvalidShape,
		`{"data":[{"name":"fixture-model"}]}`:     "",
		`{"data":[{"id":"","name":"fixture-a"}]}`: "",
		`{"data":[]}`:                             "",
	} {
		decoded, err := decodeCatalogResponse(context.Background(), &http.Response{
			StatusCode: 200, Body: io.NopCloser(strings.NewReader(body)),
		}, time.Now(), "data", "id", "name")
		if want == "" {
			if err != nil || decoded == nil {
				t.Errorf("%s: rejected a valid catalog: %v", body, err)
			}
			continue
		}
		if catalogErr := catalogCode(t, err); catalogErr.Code != want || decoded != nil {
			t.Errorf("%s: code = %q, want %q", body, catalogErr.Code, want)
		}
		if core.ClassifyError(err) != (core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}) {
			t.Errorf("%s: classification = %+v", body, core.ClassifyError(err))
		}
	}
}

func TestDecodeCatalogResponseThatBreaksOffIsATransportError(t *testing.T) {
	t.Parallel()
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	for _, check := range []struct {
		ctx  context.Context
		want core.ProviderErrorClassification
	}{
		{context.Background(), core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}},
		{canceled, core.ProviderErrorClassification{}},
	} {
		_, err := decodeCatalogResponse(check.ctx, &http.Response{StatusCode: 200, Body: &partialErrorReader{}}, time.Now(), "data", "id")
		if catalogErr := catalogCode(t, err); catalogErr.Code != CatalogCodeTransportError || !errors.Is(err, io.ErrUnexpectedEOF) ||
			core.ClassifyError(err) != check.want {
			t.Fatalf("code = %q, classification = %+v, want %+v", catalogErr.Code, core.ClassifyError(err), check.want)
		}
	}
}
