package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers/zen"
)

// zenPublicBearer is the Authorization the OpenCode CLI sends without a key.
const zenPublicBearer = "Bearer public"

// zenMaxResponseBytes is the gateway's bound on a complete Zen response. A
// stream record has the shared bound, maxStreamRecordWireSize.
const zenMaxResponseBytes = 64 << 20

// zenAccess is how one operation authenticates.
type zenAccess struct {
	anonymous     bool
	authorization string
}

// zenCredential reads a credential as the gateway reads a Zen API key: a
// "Bearer " prefix is dropped, and no key, "free", "none" or "public" is
// anonymous access. A credential without an API key offers its token.
func zenCredential(credential *core.Credential) zenAccess {
	key := ""
	if credential != nil {
		key = credential.APIKey
		if strings.TrimSpace(key) == "" {
			key = credential.Token
		}
	}
	key = strings.TrimSpace(key)
	if strings.HasPrefix(strings.ToLower(key), "bearer ") {
		key = strings.TrimSpace(key[len("bearer "):])
	}
	switch strings.ToLower(key) {
	case "", "free", "none", "public":
		return zenAccess{anonymous: true, authorization: zenPublicBearer}
	}
	return zenAccess{authorization: "Bearer " + key}
}

// zenHeaders are the headers of every inference request. The gateway sends
// the invocation identity with a key too, not only anonymously.
func zenHeaders(access zenAccess, identity zen.InvocationIdentity) http.Header {
	header := http.Header{}
	header.Set("Content-Type", core.ContentTypeJSON)
	header.Set("Authorization", access.authorization)
	zen.ApplyInvocationHeaders(header, identity)
	return header
}

// zenPayload decodes the JSON object body. Numbers stay json.Number, so a
// value passed through reaches Zen as the caller wrote it.
func zenPayload(request core.Request) (map[string]any, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return nil, &core.ProviderError{Message: "an OpenCode Zen request body must be JSON", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	decoder := json.NewDecoder(bytes.NewReader(request.Body))
	decoder.UseNumber()
	var payload map[string]any
	err = decoder.Decode(&payload)
	if err == nil {
		if _, trailing := decoder.Token(); trailing != io.EOF {
			err = errors.New("the body continues after its JSON object")
		}
	}
	if err != nil || payload == nil {
		return nil, &core.ProviderError{Message: "the OpenCode Zen request body is not a JSON object", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	return payload, nil
}

// zenEncode encodes a payload as the gateway does: keys sorted, HTML left
// unescaped, and no trailing newline.
func zenEncode(payload map[string]any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(payload); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buffer.Bytes(), "\n"), nil
}

// post sends payload to path. The response is open whatever its status.
func (p *Zen) post(ctx context.Context, path string, header http.Header, payload map[string]any) (*http.Response, error) {
	body, err := zenEncode(payload)
	if err != nil {
		return nil, zenFailure(ctx, &InvocationError{Msg: "the OpenCode Zen request could not be encoded", Cause: err, class: core.ProviderErrorInvalidRequest})
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, zenFailure(ctx, &InvocationError{Msg: "the OpenCode Zen request could not be created", Cause: err, class: core.ProviderErrorConfiguration})
	}
	request.Header = header.Clone()
	response, err := p.client.Do(request)
	if err != nil {
		return nil, zenFailure(ctx, zenTransportError(ctx, "OpenCode Zen could not be reached", err))
	}
	return response, nil
}

func zenTransportError(ctx context.Context, message string, err error) *InvocationError {
	upstream := ctx.Err() == nil
	return &InvocationError{
		Msg: message, Retryable: upstream, FailoverEligible: upstream, CircuitFailure: upstream,
		Cause: err, class: core.ProviderErrorTransport,
	}
}

// statusError reports a response Zen refused. The message carries only the
// status and identifier-like diagnostics, never the body. Without a key a
// 401 means the model needs one, so it gets the gateway's guidance.
func (p *Zen) statusError(ctx context.Context, response *http.Response, raw []byte, call zenCall) error {
	status := response.StatusCode
	message := fmt.Sprintf("OpenCode Zen returned HTTP %d", status)
	if status == http.StatusUnauthorized && call.access.anonymous {
		message = zenAnonymousRefusal(call.model)
	} else if diagnostic := codexErrorDiagnostic(raw); diagnostic != "" {
		message += " (" + diagnostic + ")"
	}
	return zenFailure(ctx, &InvocationError{
		Msg: message, Status: status, RetryAfter: parseRetryAfterHeader(response.Header.Get("Retry-After"), p.now()),
		Retryable:        status == http.StatusRequestTimeout || status == http.StatusTooManyRequests || status >= 500,
		FailoverEligible: status == http.StatusTooManyRequests || status >= 500,
	})
}

// readZenBody reads a complete response within the gateway's bound.
func readZenBody(ctx context.Context, body io.Reader) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(body, zenMaxResponseBytes+1))
	if len(raw) > zenMaxResponseBytes {
		return nil, zenUpstreamError("the OpenCode Zen response exceeds the size limit", false, nil)
	}
	if err != nil {
		return nil, zenFailure(ctx, zenTransportError(ctx, "the OpenCode Zen response could not be read", err))
	}
	return raw, nil
}

// readZenResponse reads and closes a response as the gateway does: a
// failed read of a successful response is an error, while a refused
// response whose body cannot be read has none.
func readZenResponse(ctx context.Context, response *http.Response) ([]byte, error) {
	defer response.Body.Close()
	raw, err := readZenBody(ctx, response.Body)
	if err != nil && response.StatusCode >= http.StatusBadRequest {
		return nil, nil
	}
	return raw, err
}

// zenUpstreamError reports a response Zen sent that cannot be used.
// Another target may still serve the request.
func zenUpstreamError(message string, retryable bool, cause error) *core.ProviderError {
	return &core.ProviderError{
		Message: message, Class: core.ProviderErrorUpstream, Cause: cause,
		Classification: core.ProviderErrorClassification{Retryable: retryable, FailoverEligible: true, CircuitFailure: true},
	}
}

// fetch reads one catalog document. Codes are prefixed "metadata_" for the
// models.dev catalog, as zenCatalogFailure describes.
func (p *Zen) fetch(ctx context.Context, endpoint string, header http.Header, prefix string) ([]byte, error) {
	document := "the OpenCode Zen catalog"
	if prefix != "" {
		document = "the models.dev catalog"
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, core.NewConfigurationError("OpenCode Zen could not request "+document, err)
	}
	request.Header = header
	response, err := p.catalogClient.Do(request)
	if err != nil {
		return nil, zenCatalogTransportFailure(ctx, prefix, "OpenCode Zen could not reach "+document, err)
	}
	defer response.Body.Close()
	if status := response.StatusCode; status < http.StatusOK || status >= http.StatusMultipleChoices {
		retryAfter := parseRetryAfterHeader(response.Header.Get("Retry-After"), p.now())
		return nil, zenCatalogFailure(prefix+"http_error", fmt.Sprintf("%s returned HTTP %d", document, status), status, retryAfter, nil)
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, p.maxCatalogBytes+1))
	if err != nil {
		return nil, zenCatalogTransportFailure(ctx, prefix, "OpenCode Zen could not read "+document, err)
	}
	if int64(len(raw)) > p.maxCatalogBytes {
		return nil, zenCatalogFailure(prefix+"not_discoverable", document+" exceeds the size limit", 0, 0, nil)
	}
	return raw, nil
}

// zenCatalogTransportFailure permits nothing once the caller gave up.
func zenCatalogTransportFailure(ctx context.Context, prefix, detail string, err error) error {
	failure := zenCatalogFailure(prefix+"transport_error", detail, 0, 0, err)
	if ctx.Err() != nil {
		failure.Classification = core.ProviderErrorClassification{}
	}
	return failure
}

// zenVisionHeader marks a request with an image as the gateway's shared
// OpenAI-compatible transport marks it for every provider, Zen included:
// GitHub Copilot needs the header, and other providers ignore it.
func zenVisionHeader(header http.Header, images bool) {
	if images {
		header.Set("Copilot-Vision-Request", "true")
	}
}

// zenChatHasImages reports a message with an image part.
func zenChatHasImages(messages []map[string]any) bool {
	for _, message := range messages {
		for _, raw := range anySlice(message["content"]) {
			if part, ok := raw.(map[string]any); ok {
				switch part["type"] {
				case "image_url", "input_image", "image":
					return true
				}
			}
		}
	}
	return false
}

// zenResponsesHasImages reports an image anywhere in a Responses input.
func zenResponsesHasImages(value any) bool {
	switch current := value.(type) {
	case []any:
		for _, item := range current {
			if zenResponsesHasImages(item) {
				return true
			}
		}
	case map[string]any:
		switch current["type"] {
		case "input_image", "image_url":
			return true
		}
		for _, nested := range current {
			if zenResponsesHasImages(nested) {
				return true
			}
		}
	}
	return false
}
