package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"mime"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// azureLabel is the gateway's prefix for Azure OpenAI diagnostics.
const azureLabel = "azure_openai"

// AzureOpenAIConfig configures AzureOpenAI. BaseURL is required.
type AzureOpenAIConfig struct {
	// BaseURL is the resource's endpoint: the origin the Azure portal
	// shows, optionally followed by /openai or /openai/v1. Chat Completions
	// are sent to <origin>/openai/v1/chat/completions and the deployments
	// are listed from the origin.
	BaseURL string
	// Client performs Chat requests. Nil uses a client that times out after
	// 300 seconds, the gateway's default for OpenAI-compatible providers.
	Client *http.Client
	// CatalogClient lists deployments. Nil uses a client that times out
	// after 10 seconds, the gateway's bound on catalog requests.
	CatalogClient *http.Client
	// Now stamps catalog discovery and reads Retry-After. Nil uses time.Now.
	Now func() time.Time
}

// AzureOpenAI implements core.Provider for one Azure OpenAI resource, as
// the gateway's azure_openai provider serves it. It diverges from other
// OpenAI-compatible providers where a live resource does:
//
//   - The credential is an API key, sent as the api-key header, never as a
//     bearer. A request without one is refused before anything is sent.
//   - Chat Completions is the only native surface, sent to the resource's
//     /openai/v1 route without an api-version. The model is the deployment
//     name, and the answer comes back as Azure sent it, its version-stamped
//     model and Azure-only fields included. A translation.Adapter in front
//     serves Messages and Responses over Chat.
//   - The catalog is the resource's own deployments; see ListModels.
//
// A Chat body is rebuilt as the gateway's transport rebuilds it: the model,
// the messages, the stream flag and the fields that transport forwards,
// any other field dropped and reported as an advisory loss. It is not the
// body the client sent, so Azure OpenAI does not implement
// core.WirePreserver, and a product labels its answers translated, as the
// gateway labels Azure Chat.
//
// Redirects that leave the resource's origin are not followed: Go would
// send the api-key header with them.
type AzureOpenAI struct {
	baseURL, origin       string
	client, catalogClient *http.Client
	now                   func() time.Time
}

var _ core.Provider = (*AzureOpenAI)(nil)

// NewAzureOpenAI returns an Azure OpenAI provider. A base URL that is not a
// resource endpoint is refused, as the gateway refuses it at configuration;
// see azureInferenceBaseURL.
func NewAzureOpenAI(config AzureOpenAIConfig) (*AzureOpenAI, error) {
	baseURL, err := azureInferenceBaseURL(config.BaseURL)
	if err != nil {
		return nil, core.NewConfigurationError("the Azure OpenAI base URL is not a resource endpoint: it must be an http(s) URL "+
			"without a query or fragment, whose path is empty, /openai or /openai/v1", err)
	}
	origin, err := azureResourceOrigin(baseURL)
	if err != nil {
		return nil, core.NewConfigurationError("the Azure OpenAI base URL has no origin", err)
	}
	p := &AzureOpenAI{baseURL: baseURL, origin: origin, now: config.Now}
	client, catalogClient := config.Client, config.CatalogClient
	if client == nil {
		client = &http.Client{Timeout: 300 * time.Second}
	}
	if catalogClient == nil {
		catalogClient = &http.Client{Timeout: 10 * time.Second}
	}
	p.client, p.catalogClient = azureOriginClient(client, origin), azureOriginClient(catalogClient, origin)
	if p.now == nil {
		p.now = time.Now
	}
	return p, nil
}

// errAzureRedirect refuses a redirect off the resource's origin.
var errAzureRedirect = errors.New("the Azure OpenAI resource redirected off its origin")

// azureOriginClient returns a copy of client that follows no redirect off
// origin. Following a redirect to another host, Go drops Authorization and
// cookies but forwards every other header, api-key included, so without it
// an answer could send the key, and a Chat body with it, to that host:
// what the deployments listing's same-origin check keeps its own
// continuation from doing.
func azureOriginClient(client *http.Client, origin string) *http.Client {
	guarded := *client
	next := client.CheckRedirect
	guarded.CheckRedirect = func(request *http.Request, via []*http.Request) error {
		if _, ok := azureSameOriginPage(origin, origin, request.URL.String()); !ok {
			return errAzureRedirect
		}
		if next != nil {
			return next(request, via)
		}
		if len(via) >= 10 {
			return errors.New("stopped after 10 redirects")
		}
		return nil
	}
	return &guarded
}

// NativeSurfaces reports Chat Completions for every deployment.
func (p *AzureOpenAI) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

// Invoke performs one Chat Completions request. The answer is returned as
// Azure sent it once it checks out as the gateway checks it: a JSON object
// whose first choice is an object.
func (p *AzureOpenAI) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	call, err := p.prepare(request, false)
	if err != nil {
		return core.Response{}, err
	}
	response, err := p.post(ctx, call)
	if err != nil {
		return core.Response{Losses: call.losses}, err
	}
	defer response.Body.Close()
	raw, err := readInvocationResponseBody(ctx, response, azureLabel)
	if err == nil {
		err = azureCheckCompletion(raw)
	}
	if err != nil {
		return core.Response{Losses: call.losses}, err
	}
	return core.Response{Body: raw, ContentType: core.ContentTypeJSON, Losses: call.losses}, nil
}

// Stream performs one Chat Completions request as a stream and relays
// Azure's records byte for byte, [DONE] included. The stream reports the
// request's losses through core.LossReporter.
func (p *AzureOpenAI) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	call, err := p.prepare(request, true)
	if err != nil {
		return nil, err
	}
	response, err := p.post(ctx, call)
	if err != nil {
		return nil, err
	}
	return &azureStream{sseFrameStream: newSSEFrameStream(ctx, azureLabel, response.Body), losses: call.losses}, nil
}

// azureStream relays a Chat stream and reports the request's losses.
type azureStream struct {
	*sseFrameStream
	losses []core.Loss
}

var _ core.LossReporter = (*azureStream)(nil)

// Losses implements core.LossReporter. They are known before the first
// frame.
func (s *azureStream) Losses() []core.Loss { return slices.Clone(s.losses) }

// azureCall is one Chat request Azure OpenAI has checked and shaped.
type azureCall struct {
	key    string
	body   map[string]any
	losses []core.Loss
}

// prepare refuses what Azure OpenAI cannot serve before anything is sent.
func (p *AzureOpenAI) prepare(request core.Request, stream bool) (azureCall, error) {
	if request.Surface != core.ModelSurfaceChatCompletions {
		return azureCall{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	key, err := azureCredential(request.Credential)
	if err != nil {
		return azureCall{}, err
	}
	payload, err := azurePayload(request)
	if err != nil {
		return azureCall{}, err
	}
	body, losses, err := azureChatBody(request.Model, payload, stream)
	if err != nil {
		return azureCall{}, err
	}
	return azureCall{key: key, body: body, losses: losses}, nil
}

// azureCredential returns the key a request authenticates with: the
// credential's API key, or the token of a credential of no kind. A
// credential of another kind, or no key at all, is refused before anything
// is sent; the gateway sent an empty api-key and got the resource's 401.
func azureCredential(credential *core.Credential) (string, error) {
	if credential != nil && credential.TokenType != "" && credential.TokenType != core.TokenTypeAPIKey {
		return "", core.NewConfigurationError(fmt.Sprintf("Azure OpenAI cannot authenticate with a %q credential", credential.TokenType), nil)
	}
	key := ""
	if credential != nil {
		if key = strings.TrimSpace(credential.APIKey); key == "" {
			key = strings.TrimSpace(credential.Token)
		}
	}
	if key == "" {
		return "", core.NewConfigurationError("Azure OpenAI needs a credential with an API key", nil)
	}
	return key, nil
}

// azurePayload decodes the JSON object body. Numbers stay json.Number, so
// a value passed through reaches Azure as the caller wrote it.
func azurePayload(request core.Request) (map[string]any, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return nil, &core.ProviderError{Message: "an Azure OpenAI request body must be JSON", Class: core.ProviderErrorInvalidRequest, Cause: err}
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
		return nil, &core.ProviderError{Message: "the Azure OpenAI request body is not a JSON object", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	return payload, nil
}

// azureChatForwarded reports a Chat field the gateway's OpenAI-compatible
// transport forwards, besides the model, messages and stream flag. The
// gateway's Azure provider builds every body with that transport's payload.
func azureChatForwarded(field string) bool {
	switch field {
	case "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "tools", "tool_choice",
		"reasoning_effort", "stream_options", "metadata", "parallel_tool_calls", "thinking":
		return true
	}
	return false
}

// azureChatBody rebuilds a Chat body as the gateway's Azure transport sends
// it: the deployment as the model, the messages, the operation's stream
// flag and the forwarded fields that are set. The gateway's internal
// fields, prefixed "_", never reach Azure, but llm-translate's
// _max_output_tokens, a translated Responses request's limit, is sent as
// max_completion_tokens unless the request sets a limit of its own. Any
// other field is dropped and reported as an advisory loss.
func azureChatBody(model string, payload map[string]any, stream bool) (map[string]any, []core.Loss, error) {
	messages, err := codexMessages(payload["messages"])
	if err != nil {
		return nil, nil, &core.ProviderError{Message: "the Azure OpenAI Chat messages are invalid: " + err.Error(), Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	body := map[string]any{"model": model, "messages": messages, "stream": stream}
	if limit := payload["_max_output_tokens"]; limit != nil && payload["max_tokens"] == nil && payload["max_completion_tokens"] == nil {
		body["max_completion_tokens"] = limit
	}
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(payload)) {
		value := payload[field]
		switch {
		case field == "model" || field == "messages" || field == "stream":
		case value == nil || strings.HasPrefix(field, "_"):
		case azureChatForwarded(field):
			body[field] = value
		default:
			losses = append(losses, core.Loss{
				Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory,
				Detail: "the Azure OpenAI Chat transport does not forward this field",
			})
		}
	}
	return body, losses, nil
}

// post sends a call's body to the Chat route and returns the open response
// of a request Azure accepted. A refusal is read and reported with its
// status and Retry-After, which the gateway dropped. The body is encoded as
// the gateway encodes it, with json.Marshal: keys sorted and HTML escaped.
func (p *AzureOpenAI) post(ctx context.Context, call azureCall) (*http.Response, error) {
	body, err := json.Marshal(call.body)
	if err != nil {
		return nil, &core.ProviderError{Message: "the Azure OpenAI request could not be encoded", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, core.NewConfigurationError("the Azure OpenAI base URL is not a usable request URL", err)
	}
	request.Header = azureHeaders(call.key)
	response, err := p.client.Do(request)
	switch {
	case errors.Is(err, errAzureRedirect):
		return nil, unusableResponse("Azure OpenAI redirected the request off the resource", err)
	case err != nil:
		return nil, transportFailure(ctx, "Azure OpenAI could not be reached", err)
	}
	if response.StatusCode >= http.StatusBadRequest {
		defer response.Body.Close()
		raw, _ := readInvocationResponseBody(ctx, response, azureLabel)
		return nil, httpStatusFailure(azureLabel, response, raw, p.now())
	}
	return response, nil
}

// azureHeaders are the headers of every request, the catalog's included:
// the key goes in api-key, the main way Azure diverges from other
// OpenAI-compatible providers.
func azureHeaders(key string) http.Header {
	header := http.Header{}
	header.Set("Content-Type", core.ContentTypeJSON)
	header.Set("api-key", key)
	return header
}

// azureCheckCompletion checks a Chat answer as the gateway checks it: a
// nonempty JSON object whose choices start with an object. It decodes into
// a map, so fields only Azure sends never break it.
func azureCheckCompletion(raw []byte) error {
	var completion map[string]any
	if json.Unmarshal(raw, &completion) != nil || len(completion) == 0 {
		return unusableResponse("the azure_openai response is not a JSON object", nil)
	}
	choices, _ := completion["choices"].([]any)
	if len(choices) == 0 {
		return unusableResponse("the azure_openai response has no choices", nil)
	}
	if _, ok := choices[0].(map[string]any); !ok {
		return unusableResponse("the azure_openai response has an invalid choice", nil)
	}
	return nil
}

// azureInferenceSuffix is the path Chat Completions hang off. The Azure
// portal's endpoint is the bare origin, so it is appended when missing.
const azureInferenceSuffix = "/openai/v1"

// azureInferenceBaseURL normalizes a configured endpoint into the inference
// base, as the gateway does, or refuses it.
//
// The two consumers of the endpoint must agree: the catalog lists
// deployments from the origin alone, and inference appends to the path. So
// a bare origin, the portal's value, gets the suffix, where it once listed
// the deployments and then 404ed every completion. A path is matched
// exactly, not by suffix: a proxy mount such as /proxy/openai/v1, or the
// legacy deployment-scoped route, is another endpoint rather than a missing
// suffix, and the deployments listing would not go through it. The scheme
// must be http(s), since any other fails every request with a transport
// error that names nothing an operator can trace back to the endpoint.
func azureInferenceBaseURL(raw string) (string, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return "", fmt.Errorf("azure_openai requires base_url: the resource endpoint, optionally with the %s suffix", azureInferenceSuffix)
	}
	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") ||
		parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", fmt.Errorf("azure_openai base_url must be an absolute http(s) endpoint URL with no query or "+
			"fragment, optionally with the %s suffix", azureInferenceSuffix)
	}
	path := strings.TrimRight(parsed.Path, "/")
	switch path {
	case "", "/openai":
		path = azureInferenceSuffix
	case azureInferenceSuffix:
	default:
		return "", fmt.Errorf("azure_openai base_url path %q is not a resource endpoint; expected no path, %s, or %s",
			path, "/openai", azureInferenceSuffix)
	}
	parsed.Path = path
	return parsed.String(), nil
}

// azureResourceOrigin keeps only the scheme and host of an endpoint: the
// deployments route lives under the resource's origin, whatever path the
// inference base carries.
func azureResourceOrigin(baseURL string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return "", fmt.Errorf("base_url is not an absolute URL")
	}
	return parsed.Scheme + "://" + parsed.Host, nil
}
