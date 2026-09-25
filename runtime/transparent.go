package runtime

import (
	"context"
	"encoding/json"
	"mime"
	"slices"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/catalog"
)

// Transparent performs one non-streaming request on instance exactly as the
// caller sent it. The body reaches the provider as is: no translation, no
// retry and no failover, only the credential refresh Invoke does.
//
// It serves only what the caller's stored catalog confirms, as the
// gateway's transparent mode does: the catalog lists the model and the
// request's surface, the provider preserves that surface's wire for the
// model (see core.PreservesWire), and the row's capabilities are trusted
// and fresh (see core.TransportEvidence). It never discovers a catalog.
// Anything else, and then a body that asks to stream, fails with a
// *core.TransportRejectError before anything is sent.
func (r *Runtime[S]) Transparent(ctx context.Context, caller core.Caller, instance string, request core.Request) (core.Response, error) {
	if err := request.Validate(); err != nil {
		return core.Response{}, invalidRequest(err)
	}
	b, err := r.bind(instance)
	if err != nil {
		return core.Response{}, err
	}
	c, err := r.credential(ctx, caller, instance, b)
	if err != nil {
		r.observe(ctx, caller, instance, c, core.EvidenceOperationInvoke, request.Surface, request.Model, err)
		return core.Response{}, err
	}
	if err := r.transparent(ctx, instance, b, c, request); err != nil {
		return core.Response{}, err
	}
	request.Credential = c.value
	response, err := b.provider.Invoke(ctx, request)
	if rejectedCredential(err, c) {
		if next, ok := refreshed(ctx, b, c); ok {
			c = next
			request.Credential = c.value
			response, err = b.provider.Invoke(ctx, request)
		}
	}
	r.observe(ctx, caller, instance, c, core.EvidenceOperationInvoke, request.Surface, request.Model, err)
	return response, err
}

// transparent refuses a request no transparent plan carries, and then one
// that streams, in the gateway's order.
func (r *Runtime[S]) transparent(ctx context.Context, instance string, b binding, c credential, request core.Request) error {
	read := r.catalogs.Cached(ctx, core.CatalogKey{Instance: instance, CredentialKey: c.key})
	if read.Err != nil {
		return read.Err
	}
	refused := &core.TransportRejectError{Reason: core.TransportRejectNotCataloged, Surface: request.Surface, Model: request.Model}
	row, found := catalog.Find(read.Record, request.Model)
	if !found {
		return refused
	}
	interfaces := core.TransportInterfaces(b.provider, c.value, request.Model, row.SupportedAPIs)
	refused.NativeInterface = slices.Contains(interfaces, core.TransportInterface{Surface: request.Surface, Native: core.SupportSupported})
	plan := core.PlanTransport(core.TransportPlanRequest{
		Operation: core.ModelOperationChat, Surface: request.Surface, EvaluatedAt: r.options.Now(), ExactTarget: true,
		Capabilities: core.TransportEvidence{Row: row, RefreshedAt: read.Record.Evidence.ObservedAt}.Capabilities(),
		Interfaces:   interfaces, Requirement: core.TransportRequirementTransparent,
	})
	if plan.Disposition != core.TransportNative {
		refused.Reason = plan.Reason
		return refused
	}
	if streams(request) {
		refused.Reason = core.TransportRejectStreaming
		return refused
	}
	return nil
}

// streams reports a JSON body whose stream field is true.
func streams(request core.Request) bool {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return false
	}
	var body struct {
		Stream bool `json:"stream"`
	}
	return json.Unmarshal(request.Body, &body) == nil && body.Stream
}

// TransportMode labels the transport of a response instance served caller
// for model on surface, as core.ResponseTransportMode does over the
// caller's stored catalog. It discovers nothing.
func (r *Runtime[S]) TransportMode(ctx context.Context, caller core.Caller, instance, model string, surface core.ModelSurface) string {
	b, err := r.bind(instance)
	if err != nil {
		return core.TransportModeTranslated
	}
	key, credential, err := r.storedCredential(ctx, caller, instance)
	if err != nil {
		return core.ResponseTransportMode(b.provider, nil, model, surface, nil, r.options.Now())
	}
	var evidence *core.TransportEvidence
	read := r.catalogs.Cached(ctx, core.CatalogKey{Instance: instance, CredentialKey: key})
	if row, ok := catalog.Find(read.Record, model); ok {
		evidence = &core.TransportEvidence{Row: row, RefreshedAt: read.Record.Evidence.ObservedAt}
	}
	return core.ResponseTransportMode(b.provider, credential, model, surface, evidence, r.options.Now())
}
