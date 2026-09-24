package core

import (
	"context"
	"fmt"
	"slices"
	"strings"

	translate "github.com/xibodev/llm-translate"
)

// Content types of JSON bodies and SSE streams.
const (
	ContentTypeJSON        = "application/json"
	ContentTypeEventStream = "text/event-stream"
)

// Loss is one compatibility finding from translating between surfaces or from
// adapting a request to a provider. It is llm-translate's type, so both kinds
// of finding combine in one report.
type Loss = translate.Loss

// Request is one wire-level provider operation.
//
// Body is exactly the surface's request body: JSON for the chat surfaces and
// embeddings, multipart form data for speech-to-text. ContentType names its
// encoding. The credential is explicit; a provider never looks one up.
type Request struct {
	Surface     ModelSurface
	Model       string
	Body        []byte
	ContentType string
	Credential  *Credential
}

// Validate reports a request that no provider could serve.
func (r Request) Validate() error {
	if !KnownModelSurface(r.Surface) {
		return fmt.Errorf("request has unknown surface %q", r.Surface)
	}
	if strings.TrimSpace(r.Model) == "" {
		return fmt.Errorf("request has no model")
	}
	if len(r.Body) > 0 && strings.TrimSpace(r.ContentType) == "" {
		return fmt.Errorf("request body has no content type")
	}
	return nil
}

// Response is the result of a non-streaming operation.
//
// Body is the surface's response body, JSON or binary as ContentType says.
// Losses lists every compatibility loss the operation incurred, including
// losses the loss policy allowed, so a product can log or label them.
type Response struct {
	Body        []byte
	ContentType string
	Losses      []Loss
}

// Provider performs wire-level operations on the surfaces it serves natively.
//
// A request for any other surface fails with a *SurfaceError; a translation
// adapter in front of the provider serves other surfaces. Implementations
// honour ctx cancellation and take credentials only from the request.
type Provider interface {
	// NativeSurfaces lists the surfaces model serves without translation.
	NativeSurfaces(model string) []ModelSurface
	// Invoke performs one non-streaming operation.
	Invoke(ctx context.Context, request Request) (Response, error)
	// Stream performs one streaming operation. Each frame is one complete
	// SSE record of the surface's stream.
	Stream(ctx context.Context, request Request) (StreamIter, error)
	// ListModels returns the provider's catalog for the credential.
	ListModels(ctx context.Context, credential *Credential) ([]ModelInfo, error)
}

// ServesNatively reports whether provider serves surface for model without
// translation.
func ServesNatively(provider Provider, model string, surface ModelSurface) bool {
	return slices.Contains(provider.NativeSurfaces(model), surface)
}

// LossReporter is implemented by streams that translate. Losses returns the
// request-side losses before the first frame and adds response-side losses
// as frames are translated; the report is complete once Next returns io.EOF.
type LossReporter interface {
	Losses() []Loss
}

// StreamLosses returns the losses a stream reports, or nil for a stream that
// does not translate.
func StreamLosses(stream StreamIter) []Loss {
	if reporter, ok := stream.(LossReporter); ok {
		return reporter.Losses()
	}
	return nil
}

// SurfaceError reports a request for a surface the provider does not serve
// natively. Another target may serve it, so it permits failover.
type SurfaceError struct {
	Surface ModelSurface
	Model   string
}

func (e *SurfaceError) Error() string {
	return fmt.Sprintf("model %q does not serve the %s surface natively", e.Model, e.Surface)
}

// ProviderErrorClassification permits failover without marking the provider
// unhealthy.
func (e *SurfaceError) ProviderErrorClassification() ProviderErrorClassification {
	return ProviderErrorClassification{FailoverEligible: true}
}
