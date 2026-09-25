package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// azureDeploymentsAPIVersion is pinned to the last api-version observed to
// serve GET /openai/deployments. Every later one 404s there: Microsoft
// moved deployment listing to the ARM control plane, which needs Entra
// credentials and a subscription scope Azure OpenAI does not take. There is
// no newer value to bump it to.
const azureDeploymentsAPIVersion = "2023-03-15-preview"

// azureDeploymentsMaxPages bounds the has_more walk. A resource's
// deployments run to tens, so a walk this long is a resource minting
// cursors, and it fails rather than truncating the catalog.
const azureDeploymentsMaxPages = 50

// ListModels lists the resource's own deployments, the only models it can
// serve, as the gateway lists them. GET /models is Azure's whole vendor
// catalogue, hundreds of rows against a resource's few deployments with no
// field that tells them apart, so it is never requested, not even when the
// deployments listing fails.
//
// The route has not been seen to page, so the walk is defensive: a page
// with has_more continues after its last_id, or its last row's id, carried
// as a query value on a URL built from the configured origin and checked
// to stay there, so no answer can send the api-key elsewhere. A nextLink is
// ignored. A page with has_more but no new cursor, or a walk past 50 pages,
// fails rather than returning part of the catalog.
//
// A deployment is listed only when it succeeded and its base model is
// chat-callable (see azureDeploymentIsChatCallable). Its id is the
// deployment name a request sends as its model, its display name the base
// model when that differs, and it declares chat over Chat Completions and
// Messages, with the capabilities the gateway infers from that.
//
// A key the resource rejects fails with CatalogCodeAuthenticationFailed
// and its status, and any other failure with CatalogCodeNotDiscoverable,
// the gateway's codes, each classified by what failed.
func (p *AzureOpenAI) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	key, err := azureCredential(credential)
	if err != nil {
		return nil, err
	}
	discoveredAt := p.now().UTC()
	pageURL := p.origin + "/openai/deployments?api-version=" + azureDeploymentsAPIVersion
	models := []core.ModelInfo{}
	seen := map[string]bool{}
	for page := 0; ; page++ {
		if page >= azureDeploymentsMaxPages {
			return nil, azureNotDiscoverable(ctx, "Azure OpenAI deployments listing did not end within the page limit; the catalog is not discoverable.", 0, nil)
		}
		decoded, err := p.deploymentsPage(ctx, key, pageURL)
		if err != nil {
			return nil, err
		}
		models = append(models, azureDeploymentRows(decoded, discoveredAt)...)
		if more, _ := decoded["has_more"].(bool); !more {
			return models, nil
		}
		cursor := azureDeploymentsCursor(decoded)
		if cursor == "" || seen[cursor] {
			return nil, azureNotDiscoverable(ctx, "Azure OpenAI deployments listing reported more results but gave no usable cursor to continue from; the catalog is not discoverable.", 0, nil)
		}
		seen[cursor] = true
		next, ok := azureNextDeploymentsPage(p.origin, azureDeploymentsAPIVersion, cursor)
		if !ok {
			return nil, azureNotDiscoverable(ctx, "Azure OpenAI deployments continuation left the configured resource; the catalog is not discoverable.", 0, nil)
		}
		pageURL = next
	}
}

// deploymentsPage fetches and decodes one page of the deployments listing.
// The gateway decodes the first JSON value into a map, so a document that
// is null or lacks data is an empty page, not a failure.
func (p *AzureOpenAI) deploymentsPage(ctx context.Context, key, pageURL string) (map[string]any, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, pageURL, nil)
	if err != nil {
		return nil, azureNotDiscoverable(ctx, "Azure OpenAI base_url could not be built into a deployments request.", 0, err)
	}
	request.Header = azureHeaders(key)
	response, err := p.catalogClient.Do(request)
	switch {
	case errors.Is(err, errAzureRedirect):
		return nil, azureNotDiscoverable(ctx, "Azure OpenAI deployments redirected off the configured resource; the catalog is not discoverable.", 0, err)
	case err != nil:
		return nil, azureCatalogTransportFailure(ctx, "Azure OpenAI deployments could not be reached; the catalog is not discoverable.", 0, err)
	}
	defer response.Body.Close()
	status := response.StatusCode
	if status >= http.StatusBadRequest {
		retryAfter := retryAfterDelay(response.Header.Get("Retry-After"), p.now())
		if status == http.StatusUnauthorized || status == http.StatusForbidden {
			// The route refused this key, so the catalog is not undiscoverable:
			// the code tells a product to attribute the failure to the key.
			return nil, catalogFailure(ctx, CatalogCodeAuthenticationFailed, "Provider credential was rejected by the catalog API.", status, retryAfter, nil)
		}
		failure := azureNotDiscoverable(ctx, fmt.Sprintf("Azure OpenAI deployments request returned HTTP %d; the catalog is not discoverable.", status), status, nil)
		failure.Class = core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: status}).ErrorClass
		failure.Classification = statusClassification(status, retryAfter)
		return nil, failure
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, catalogMaxResponseBytes+1))
	if len(raw) > catalogMaxResponseBytes {
		return nil, azureNotDiscoverable(ctx, "Provider catalog response exceeded the size limit.", status, nil)
	}
	if err != nil {
		return nil, azureCatalogTransportFailure(ctx, "Azure OpenAI deployments response could not be read; the catalog is not discoverable.", status, err)
	}
	var decoded map[string]any
	if err := json.NewDecoder(bytes.NewReader(raw)).Decode(&decoded); err != nil {
		return nil, azureNotDiscoverable(ctx, "Azure OpenAI deployments response was not valid JSON; the catalog is not discoverable.", status, err)
	}
	return decoded, nil
}

// azureNotDiscoverable reports a listing that cannot be completed with the
// gateway's code, as an answer that cannot be used unless the caller
// reclassifies it.
func azureNotDiscoverable(ctx context.Context, detail string, status int, cause error) *core.ProviderError {
	return catalogFailure(ctx, CatalogCodeNotDiscoverable, detail, status, 0, cause)
}

// azureCatalogTransportFailure keeps the gateway's code for a listing that
// got no complete answer, which may repeat unless the caller gave up.
func azureCatalogTransportFailure(ctx context.Context, detail string, status int, cause error) *core.ProviderError {
	failure := azureNotDiscoverable(ctx, detail, status, cause)
	failure.Class = core.ProviderErrorTransport
	failure.Classification = core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}
	if ctx.Err() != nil {
		failure.Classification = core.ProviderErrorClassification{}
	}
	return failure
}

// azureDeploymentsCursor reads the cursor of a page with has_more, under
// the OpenAI list convention this envelope belongs to: last_id, or else the
// id of the page's last row.
func azureDeploymentsCursor(page map[string]any) string {
	if lastID, ok := page["last_id"].(string); ok && strings.TrimSpace(lastID) != "" {
		return strings.TrimSpace(lastID)
	}
	items, _ := page["data"].([]any)
	if len(items) == 0 {
		return ""
	}
	last, _ := items[len(items)-1].(map[string]any)
	id, _ := last["id"].(string)
	return strings.TrimSpace(id)
}

// azureNextDeploymentsPage builds the URL of the next page from the
// configured origin and the pinned api-version, the cursor carried only as
// a query value, and checks it stays on the origin anyway, so the check
// holds even if the URL is ever taken from an answer instead.
func azureNextDeploymentsPage(origin, apiVersion, cursor string) (string, bool) {
	candidate := origin + "/openai/deployments?api-version=" + url.QueryEscape(apiVersion) + "&after=" + url.QueryEscape(cursor)
	return azureSameOriginPage(origin, origin, candidate)
}

// azureSameOriginPage resolves next against current and accepts it only on
// origin and without userinfo, a second credential nobody configured. Every
// URL that carries the api-key must clear it.
func azureSameOriginPage(origin, current, next string) (string, bool) {
	base, err := url.Parse(current)
	if err != nil {
		return "", false
	}
	resolved, err := base.Parse(strings.TrimSpace(next))
	if err != nil || resolved.User != nil || !strings.EqualFold(resolved.Scheme+"://"+resolved.Host, origin) {
		return "", false
	}
	return resolved.String(), true
}

// azureDeploymentRows turns one page into catalog rows, skipping a row that
// is not an object with an id, whose deployment has not succeeded, or
// whose base model is not chat-callable.
//
// A row declares chat rather than leaving it to convention, because a row
// that declares no modality reads as a chat model downstream. Its
// capabilities are what the gateway infers when it stores the row.
func azureDeploymentRows(page map[string]any, discoveredAt time.Time) []core.ModelInfo {
	items, _ := page["data"].([]any)
	models := make([]core.ModelInfo, 0, len(items))
	for _, item := range items {
		row, ok := item.(map[string]any)
		if !ok {
			continue
		}
		id, _ := row["id"].(string)
		if id == "" {
			continue
		}
		// Creating, failed and deleting deployments cannot serve a request.
		if status, ok := row["status"].(string); ok && status != "" && status != "succeeded" {
			continue
		}
		// The base model, such as gpt-4o, only describes the deployment: a
		// request names the deployment.
		base, _ := row["model"].(string)
		if !azureDeploymentIsChatCallable(base, id) {
			continue
		}
		model := core.ModelInfo{
			ID: id, Object: "model", OwnedBy: "azure-openai", Vendor: "azure-openai",
			LegacyCapabilities: map[string]any{"chat": true},
			SupportedAPIs:      []string{"/v1/chat/completions", "/v1/messages"},
		}
		if base != "" && base != id {
			model.DisplayName = base
		}
		model.Capabilities = core.InferCapabilities(model, discoveredAt, time.Time{})
		models = append(models, model)
	}
	return models
}

// azureDeploymentIsChatCallable decides whether a deployment belongs in the
// catalog, as the gateway decides it. A resource holds embedding, image,
// audio and legacy completion deployments beside chat ones, all
// "succeeded", and Azure OpenAI serves only Chat Completions, so listing
// them would advertise models it cannot serve. A deployment it does not
// recognize is left out too, since an unannotated row reads as chat; a
// product can still route to a deployment by name.
//
// The base model the deployment was created from decides, and the
// operator-chosen deployment name only when the base is absent. The
// non-chat families are checked first, because each also carries a chat
// prefix: gpt-image-1, gpt-4o-transcribe, gpt-4o-mini-tts,
// gpt-4o-realtime-preview, gpt-35-turbo-instruct. "audio" is deliberately
// absent: gpt-4o-audio-preview is served over Chat Completions.
func azureDeploymentIsChatCallable(baseModel, deploymentID string) bool {
	name := strings.ToLower(strings.TrimSpace(baseModel))
	if name == "" {
		name = strings.ToLower(strings.TrimSpace(deploymentID))
	}
	for _, nonChat := range []string{
		"embedding", "dall-e", "image", "whisper", "tts", "transcribe",
		"realtime", "sora", "-instruct", "moderation",
	} {
		if strings.Contains(name, nonChat) {
			return false
		}
	}
	for _, chat := range []string{"gpt-", "chatgpt", "o1", "o3", "o4", "codex", "model-router"} {
		if strings.HasPrefix(name, chat) {
			return true
		}
	}
	return strings.Contains(name, "chat")
}
