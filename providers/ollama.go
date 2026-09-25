package providers

import (
	"bytes"
	"context"
	"net/http"
	"net/url"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// OllamaDefaultBaseURL is the daemon root Ollama talks to when its config
// names none, as the gateway does.
const OllamaDefaultBaseURL = "http://127.0.0.1:11434"

// OllamaConfig configures Ollama. Every field is optional.
type OllamaConfig struct {
	// BaseURL is the daemon's native root, such as OllamaDefaultBaseURL,
	// the default. OllamaBaseURLIssue says what it may not be.
	BaseURL string
	// Client performs inference. Nil uses a client that times out after 30
	// seconds, the gateway's default for Ollama.
	Client *http.Client
	// CatalogClient lists models. Nil uses a client that times out after 10
	// seconds, the gateway's bound on catalog requests.
	CatalogClient *http.Client
	// Now reads a Retry-After date. Nil uses time.Now.
	Now func() time.Time
}

// Ollama implements core.Provider for an Ollama daemon over its native
// /api/chat, as the gateway's OllamaProvider does. Ollama is keyless, so
// every operation ignores its credential.
//
// Chat Completions is its only surface, and Ollama serves it natively by
// converting each request to /api/chat and each answer back, byte for byte
// as the gateway does. A request sends its messages, temperature, top_p,
// tools, and max_tokens as num_predict; every other field is dropped and
// reported in Response.Losses. Content that is not a string reaches Ollama
// printed as Go prints it, as the gateway sends it, so an image never
// reaches it as an image; that is reported as a material loss at the
// content's path. A stream is Ollama's NDJSON as Chat chunks, ending in
// [DONE]. A translation.Adapter in front serves Messages and Responses over
// Chat.
type Ollama struct {
	chatURL, tagsURL      string
	client, catalogClient *http.Client
	now                   func() time.Time
}

var _ core.Provider = (*Ollama)(nil)

// NewOllama returns an Ollama provider. A base URL that OllamaBaseURLIssue
// finds an issue with is a configuration error carrying that issue.
func NewOllama(config OllamaConfig) (*Ollama, error) {
	base := strings.TrimSpace(config.BaseURL)
	if base == "" {
		base = OllamaDefaultBaseURL
	}
	if issue := OllamaBaseURLIssue(base); issue != "" {
		return nil, core.NewConfigurationError(issue, nil)
	}
	base = strings.TrimRight(base, "/")
	p := &Ollama{
		chatURL: base + "/api/chat", tagsURL: base + "/api/tags",
		client: config.Client, catalogClient: config.CatalogClient, now: config.Now,
	}
	if p.client == nil {
		p.client = &http.Client{Timeout: 30 * time.Second}
	}
	if p.catalogClient == nil {
		p.catalogClient = &http.Client{Timeout: 10 * time.Second}
	}
	if p.now == nil {
		p.now = time.Now
	}
	return p, nil
}

// OllamaBaseURLIssue describes, in the gateway's words, what keeps base
// from being an Ollama daemon's native root, or returns "" when it is one:
// an http or https URL with a host and no path. The OpenAI-compatible /v1
// URL gets its own advice.
func OllamaBaseURLIssue(base string) string {
	base = strings.TrimSpace(base)
	parsed, err := url.Parse(base)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return "Ollama base URL must be an http(s) native daemon root such as http://127.0.0.1:11434."
	}
	path := strings.TrimRight(parsed.EscapedPath(), "/")
	if path != "" {
		if strings.EqualFold(path, "/v1") {
			return "Ollama uses its native daemon root, not the OpenAI-compatible /v1 URL; remove /v1."
		}
		return "Ollama base URL must be the native daemon root with no path."
	}
	return ""
}

// NativeSurfaces reports Chat Completions for every model.
func (p *Ollama) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

// Invoke performs one Chat Completions request.
func (p *Ollama) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	call, err := ollamaPrepare(request, false)
	if err != nil {
		return core.Response{}, err
	}
	response, err := p.post(ctx, call.body)
	if err != nil {
		return core.Response{}, err
	}
	defer response.Body.Close()
	raw, err := readInvocationResponseBody(ctx, response, "Ollama")
	if err != nil {
		return core.Response{}, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		return core.Response{}, httpStatusFailure("Ollama", response, raw, p.now())
	}
	completion, err := ollamaCompletion(request.Model, raw)
	if err != nil {
		return core.Response{}, err
	}
	return core.Response{Body: completion, ContentType: core.ContentTypeJSON, Losses: call.losses}, nil
}

// Stream performs one Chat Completions request whose frames are Ollama's
// NDJSON events as Chat chunks, ending in [DONE]. The stream reports the
// request's losses through core.LossReporter.
func (p *Ollama) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	call, err := ollamaPrepare(request, true)
	if err != nil {
		return nil, err
	}
	response, err := p.post(ctx, call.body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		defer response.Body.Close()
		raw, _ := readInvocationResponseBody(ctx, response, "Ollama")
		return nil, httpStatusFailure("Ollama", response, raw, p.now())
	}
	return newOllamaStream(ctx, response.Body, request.Model, call.losses), nil
}

// post sends a /api/chat body. The response is open whatever its status.
func (p *Ollama) post(ctx context.Context, body []byte) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.chatURL, bytes.NewReader(body))
	if err != nil {
		return nil, core.NewConfigurationError("the Ollama request could not be created", err)
	}
	request.Header.Set("Content-Type", core.ContentTypeJSON)
	response, err := p.client.Do(request)
	if err != nil {
		return nil, transportFailure(ctx, "Ollama could not be reached", err)
	}
	return response, nil
}
