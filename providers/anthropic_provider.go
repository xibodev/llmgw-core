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
	"strconv"
	"strings"
	"time"

	anthropicauth "github.com/xibodev/llm-provider-auth/anthropic"
	core "github.com/xibodev/llmgw-core"
)

const (
	// anthropicAPIVersion is the anthropic-version every request carries.
	anthropicAPIVersion  = "2023-06-01"
	anthropicDefaultBase = "https://api.anthropic.com"
	// anthropicLabel is the gateway's prefix for Anthropic diagnostics.
	anthropicLabel = "anthropic"
	// anthropicPreambleField is where the gateway puts its preamble in a
	// Messages body. llm-translate reads the same field, so a Messages
	// request carrying it reaches a translated target with the preamble too.
	anthropicPreambleField = "_llmgw_preamble"
)

// AnthropicConfig configures Anthropic. Every field is optional.
type AnthropicConfig struct {
	// BaseURL is where the API lives: requests go to <BaseURL>/v1/messages.
	// It may carry a path, such as a proxy's mount. Empty uses
	// https://api.anthropic.com.
	BaseURL string
	// Client performs Messages requests and token counts. Nil uses a
	// client that times out after 60 seconds, the gateway's default.
	Client *http.Client
	// CatalogClient lists models. Nil uses a client that times out after
	// 10 seconds, the gateway's bound on catalog requests.
	CatalogClient *http.Client
	// Preamble returns the text to put before the system prompt of a
	// Messages request, such as a gateway preamble, or "" for none. It runs
	// once per Invoke or Stream with that operation's context, so a product
	// can return "" for a request that must reach Anthropic as sent. When
	// it returns "", a preamble the body carries under _llmgw_preamble, as
	// the gateway sets it, is used instead. Nil adds none.
	Preamble func(ctx context.Context) string
	// Now stamps catalog discovery and reads Retry-After. Nil uses time.Now.
	Now func() time.Time
}

// Anthropic implements core.Provider for the Anthropic Messages API, as the
// gateway's native Anthropic provider serves it: Messages passed through
// with its model and stream flag set, token counts, and the catalog with
// the capabilities the registry declares for its rows.
//
// Messages is the only native surface, and it is preserved: the request
// reaches Anthropic as the client wrote it and the answer comes back as
// Anthropic sent it, so Anthropic implements core.WirePreserver. Serving
// Chat Completions means translating, which is left to a
// translation.Adapter in front.
//
// Each operation authenticates with the credential it is given, through
// llm-provider-auth's anthropic.HeaderSource: an API key is sent as
// x-api-key, and a setup token, of kind core.TokenTypeAnthropicSetupToken,
// as the OAuth bearer with Anthropic's beta marker. Like the gateway, the
// header source also recognizes a setup token given as an API key. A setup
// token's completion is requested as a stream, as the gateway requests it,
// and Invoke assembles the message. Without a credential nothing
// authenticates, for an Anthropic-compatible endpoint that needs none.
type Anthropic struct {
	baseURL               string
	client, catalogClient *http.Client
	preamble              func(context.Context) string
	now                   func() time.Time
}

var (
	_ core.Provider      = (*Anthropic)(nil)
	_ core.WirePreserver = (*Anthropic)(nil)
	_ core.TokenCounter  = (*Anthropic)(nil)
)

// NewAnthropic returns an Anthropic provider. A base URL that is not an
// absolute http(s) URL without a query or fragment is refused, since every
// path is appended to it.
func NewAnthropic(config AnthropicConfig) (*Anthropic, error) {
	base := strings.TrimSpace(config.BaseURL)
	if base == "" {
		base = anthropicDefaultBase
	}
	parsed, err := url.Parse(base)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" ||
		parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, core.NewConfigurationError("the Anthropic base URL must be an absolute http(s) URL with no query or fragment", err)
	}
	p := &Anthropic{
		baseURL: strings.TrimRight(base, "/"), client: config.Client, catalogClient: config.CatalogClient,
		preamble: config.Preamble, now: config.Now,
	}
	if p.client == nil {
		p.client = &http.Client{Timeout: 60 * time.Second}
	}
	if p.catalogClient == nil {
		p.catalogClient = &http.Client{Timeout: 10 * time.Second}
	}
	if p.now == nil {
		p.now = time.Now
	}
	return p, nil
}

// NativeSurfaces reports Messages for every model.
func (p *Anthropic) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceMessages}
}

// PreservesWire implements core.WirePreserver: Messages is forwarded as
// written, as the gateway declares it.
func (p *Anthropic) PreservesWire(_ string, surface core.ModelSurface) bool {
	return surface == core.ModelSurfaceMessages
}

// Invoke performs one Messages request. The answer is returned as Anthropic
// sent it once it checks out as the gateway checks it: a JSON object with a
// content array. With a setup token, it is the message assembled from the
// stream requested instead.
func (p *Anthropic) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	call, err := p.prepare(ctx, request, false)
	if err != nil {
		return core.Response{}, err
	}
	response, err := p.post(ctx, call, "/v1/messages", anthropicLabel, nil)
	if err != nil {
		return core.Response{}, err
	}
	defer response.Body.Close()
	var body []byte
	if call.assemble {
		body, err = readAnthropicStreamedMessage(ctx, response.Body)
	} else {
		body, err = readAnthropicMessage(ctx, response)
	}
	if err != nil {
		return core.Response{}, err
	}
	return core.Response{Body: body, ContentType: core.ContentTypeJSON}, nil
}

// Stream performs one Messages request as a stream and relays Anthropic's
// records byte for byte; see anthropicStream for how it ends.
func (p *Anthropic) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	call, err := p.prepare(ctx, request, true)
	if err != nil {
		return nil, err
	}
	response, err := p.post(ctx, call, "/v1/messages", anthropicLabel, nil)
	if err != nil {
		return nil, err
	}
	return &anthropicStream{ctx: ctx, body: response.Body, reader: newSSERecordReader(response.Body)}, nil
}

// anthropicCall is one operation Anthropic has checked and is about to send.
type anthropicCall struct {
	auth    anthropicauth.HeaderSource
	payload map[string]any
	// assemble reports a completion requested as a stream, which Invoke
	// reads to its end and assembles.
	assemble bool
}

// prepare refuses what Anthropic cannot serve before anything is sent, then
// shapes the body as the gateway shapes a native Messages request: the
// request's model, the operation's stream flag, and the preamble merged
// into the system prompt.
func (p *Anthropic) prepare(ctx context.Context, request core.Request, stream bool) (anthropicCall, error) {
	if request.Surface != core.ModelSurfaceMessages {
		return anthropicCall{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	auth, err := anthropicCredential(request.Credential)
	if err != nil {
		return anthropicCall{}, err
	}
	payload, err := anthropicPayload(request)
	if err != nil {
		return anthropicCall{}, err
	}
	assemble := !stream && auth.Kind() == anthropicauth.CredentialSetupToken
	payload["model"], payload["stream"] = request.Model, stream || assemble
	preamble := ""
	if p.preamble != nil {
		preamble = p.preamble(ctx)
	}
	if preamble == "" {
		preamble, _ = payload[anthropicPreambleField].(string)
	}
	// The field never reaches Anthropic, which refuses fields it does not
	// know, whether or not it holds a preamble.
	delete(payload, anthropicPreambleField)
	anthropicMergePreamble(payload, preamble)
	return anthropicCall{auth: auth, payload: payload, assemble: assemble}, nil
}

// anthropicMergePreamble puts a preamble before the system prompt exactly
// as the gateway does: ahead of a string after a blank line, as the first
// text block of a list, and in place of anything else.
func anthropicMergePreamble(payload map[string]any, preamble string) {
	if preamble == "" {
		return
	}
	switch system := payload["system"].(type) {
	case string:
		payload["system"] = preamble + "\n\n" + system
	case []any:
		payload["system"] = append([]any{map[string]any{"type": "text", "text": preamble}}, system...)
	default:
		payload["system"] = preamble
	}
}

// anthropicCredential reads a request's credential as the gateway reads an
// Anthropic connection. An API key, or the token of a credential of no
// kind, goes to anthropic.NewHeaderSource, which tells a setup token from
// a key by its prefix. A credential of kind TokenTypeAnthropicSetupToken
// must hold a valid setup token, so it is never sent as an x-api-key. No
// credential, or one without a key, sends none. A credential of another
// kind, such as a service account, is refused before anything is sent.
func anthropicCredential(credential *core.Credential) (anthropicauth.HeaderSource, error) {
	if credential == nil {
		return anthropicauth.HeaderSource{}, nil
	}
	material := ""
	switch credential.TokenType {
	case core.TokenTypeAnthropicSetupToken:
		if err := anthropicauth.ValidateSetupToken(credential.Token); err != nil {
			return anthropicauth.HeaderSource{}, core.NewConfigurationError("the Anthropic setup token is not valid", err)
		}
		material = credential.Token
	case "", core.TokenTypeAPIKey:
		if material = credential.APIKey; strings.TrimSpace(material) == "" {
			material = credential.Token
		}
	default:
		return anthropicauth.HeaderSource{}, core.NewConfigurationError(
			fmt.Sprintf("Anthropic cannot authenticate with a %q credential", credential.TokenType), nil)
	}
	if strings.TrimSpace(material) == "" {
		return anthropicauth.HeaderSource{}, nil
	}
	source, err := anthropicauth.NewHeaderSource(material)
	if err != nil {
		return anthropicauth.HeaderSource{}, core.NewConfigurationError("the Anthropic credential is not valid", err)
	}
	return source, nil
}

// anthropicPayload decodes the JSON object body. Numbers stay json.Number,
// as the gateway decodes a Messages body, so a value passed through reaches
// Anthropic as the caller wrote it.
func anthropicPayload(request core.Request) (map[string]any, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return nil, &core.ProviderError{Message: "an Anthropic request body must be JSON", Class: core.ProviderErrorInvalidRequest, Cause: err}
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
		return nil, &core.ProviderError{Message: "the Anthropic request body is not a JSON object", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	return payload, nil
}

// post sends the call's body to path and returns the open response of a
// request Anthropic accepted. A refusal is read and reported with its
// status and Retry-After. header, when set, adjusts the headers after the
// credential's are applied, as the gateway adjusts a token count's.
//
// The body is encoded as the gateway encodes it, with json.Marshal: keys
// sorted and HTML escaped.
func (p *Anthropic) post(ctx context.Context, call anthropicCall, path, label string, header func(http.Header)) (*http.Response, error) {
	body, err := json.Marshal(call.payload)
	if err != nil {
		return nil, &core.ProviderError{Message: "the Anthropic request could not be encoded", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, core.NewConfigurationError("the Anthropic request could not be created", err)
	}
	if err := anthropicAuthorize(ctx, request, call.auth); err != nil {
		return nil, err
	}
	if header != nil {
		header(request.Header)
	}
	response, err := p.client.Do(request)
	if err != nil {
		return nil, transportFailure(ctx, "Anthropic could not be reached", err)
	}
	if response.StatusCode >= http.StatusBadRequest {
		defer response.Body.Close()
		raw, _ := readInvocationResponseBody(ctx, response, label)
		return nil, httpStatusFailure(label, response, raw, p.now())
	}
	return response, nil
}

// anthropicAuthorize sets the headers every request carries, in the
// gateway's order: the content type and anthropic-version, then the
// credential's through its header source. A keyless request sends no
// credential header.
func anthropicAuthorize(ctx context.Context, request *http.Request, auth anthropicauth.HeaderSource) error {
	request.Header.Set("Content-Type", core.ContentTypeJSON)
	request.Header.Set("anthropic-version", anthropicAPIVersion)
	if auth.Kind() == "" {
		return nil
	}
	if err := auth.Apply(ctx, request); err != nil {
		return core.NewConfigurationError("the Anthropic credential could not be applied", err)
	}
	return nil
}

// readAnthropicMessage reads a Messages answer and returns it as Anthropic
// sent it once it checks out as the gateway checks it: its first JSON value
// is a nonempty object with a content array. The gateway decodes only that
// value, so anything after it is not returned either.
func readAnthropicMessage(ctx context.Context, response *http.Response) ([]byte, error) {
	raw, err := readInvocationResponseBody(ctx, response, anthropicLabel)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var message map[string]any
	if decoder.Decode(&message) != nil || len(message) == 0 {
		return nil, unusableResponse("the anthropic response is not a JSON object", nil)
	}
	if _, ok := message["content"].([]any); !ok {
		return nil, unusableResponse("the anthropic response is not a Messages response", nil)
	}
	return raw[:decoder.InputOffset()], nil
}

// anthropicStream relays a Messages stream's records byte for byte until
// Anthropic ends it. A stream that ends before message_stop fails once its
// last record is relayed, an error event included, so a product never takes
// a stream cut short for a complete answer.
type anthropicStream struct {
	ctx     context.Context
	body    io.ReadCloser
	reader  *sseRecordReader
	stopped bool
}

var _ core.StreamIter = (*anthropicStream)(nil)

func (s *anthropicStream) Next() ([]byte, error) {
	record, err := s.reader.Next()
	switch {
	case err == io.EOF && s.stopped:
		return nil, io.EOF
	case err == io.EOF:
		return nil, unusableResponse("the anthropic stream ended before message_stop", nil)
	case err != nil:
		return nil, streamFailure(s.ctx, anthropicLabel, err)
	}
	var event struct {
		Type string `json:"type"`
	}
	if json.Unmarshal([]byte(record.data), &event) == nil && event.Type == "message_stop" {
		s.stopped = true
	}
	return record.frame, nil
}

func (s *anthropicStream) Close() error { return s.body.Close() }

// readAnthropicStreamedMessage assembles the message a Messages stream
// carries, as the gateway assembles a setup token's completion: the
// message_start message, each content block as it stops with its text,
// thinking, signature and tool input deltas, and message_delta's fields and
// usage. A stream that is not JSON, reports an error, ends incomplete or
// continues after message_stop cannot be used.
func readAnthropicStreamedMessage(ctx context.Context, body io.Reader) ([]byte, error) {
	reader := newSSERecordReader(body)
	message := map[string]any{"content": []any{}}
	blocks := map[int]map[string]any{}
	inputs := map[int]*strings.Builder{}
	started, stopped := false, false
	for {
		data, err := reader.NextData()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, anthropicStreamReadFailure(ctx, err)
		}
		if stopped {
			return nil, unusableResponse("the anthropic stream continued after message_stop", nil)
		}
		var event map[string]any
		if json.Unmarshal([]byte(data), &event) != nil {
			return nil, unusableResponse("the anthropic stream sent an event that is not JSON", nil)
		}
		number, _ := event["index"].(float64)
		index := int(number)
		switch event["type"] {
		case "message_start":
			if started {
				return nil, unusableResponse("the anthropic stream started its message twice", nil)
			}
			if start, ok := event["message"].(map[string]any); ok {
				started = true
				maps.Copy(message, start)
				message["content"] = []any{}
			}
		case "content_block_start":
			if block, ok := event["content_block"].(map[string]any); ok {
				blocks[index] = maps.Clone(block)
				if block["type"] == "tool_use" {
					inputs[index] = &strings.Builder{}
				}
			}
		case "content_block_delta":
			if delta, ok := event["delta"].(map[string]any); ok && blocks[index] != nil {
				anthropicApplyDelta(blocks[index], delta, inputs[index])
			}
		case "content_block_stop":
			block := blocks[index]
			if block == nil {
				continue
			}
			if input := inputs[index]; input != nil {
				var value any = map[string]any{}
				if input.Len() > 0 && json.Unmarshal([]byte(input.String()), &value) != nil {
					return nil, unusableResponse("the anthropic stream sent tool input that is not JSON", nil)
				}
				block["input"] = value
			}
			content, _ := message["content"].([]any)
			message["content"] = append(content, block)
			delete(blocks, index)
			delete(inputs, index)
		case "message_delta":
			if delta, ok := event["delta"].(map[string]any); ok {
				maps.Copy(message, delta)
			}
			if usage, ok := event["usage"].(map[string]any); ok {
				current, _ := message["usage"].(map[string]any)
				if current == nil {
					current = map[string]any{}
				}
				maps.Copy(current, usage)
				message["usage"] = current
			}
		case "error":
			failure := "the anthropic stream reported an error"
			if diagnostic := codexErrorDiagnostic([]byte(data)); diagnostic != "" {
				failure += " (" + diagnostic + ")"
			}
			return nil, unusableResponse(failure, nil)
		case "message_stop":
			if !started || len(blocks) != 0 {
				return nil, unusableResponse("the anthropic stream ended its message incomplete", nil)
			}
			stopped = true
		}
	}
	if !started || !stopped || len(blocks) != 0 {
		return nil, unusableResponse("the anthropic stream ended before its message was complete", nil)
	}
	if _, ok := message["content"].([]any); !ok {
		return nil, unusableResponse("the anthropic stream assembled no content list", nil)
	}
	return json.Marshal(message)
}

// anthropicApplyDelta adds one content_block_delta to its block. Tool input
// arrives as JSON fragments, joined in input and parsed when the block
// stops.
func anthropicApplyDelta(block, delta map[string]any, input *strings.Builder) {
	text := func(value any) string { s, _ := value.(string); return s }
	switch delta["type"] {
	case "text_delta":
		block["text"] = text(block["text"]) + text(delta["text"])
	case "thinking_delta":
		block["thinking"] = text(block["thinking"]) + text(delta["thinking"])
	case "signature_delta":
		block["signature"] = text(block["signature"]) + text(delta["signature"])
	case "input_json_delta":
		if input != nil {
			input.WriteString(text(delta["partial_json"]))
		}
	}
}

// anthropicStreamReadFailure reports a stream Invoke could not read to its
// end. Nothing was delivered, so a stream that broke off is a transport
// failure another attempt may get past, as the gateway treats it, while an
// oversized record cannot be used, as an oversized answer cannot.
func anthropicStreamReadFailure(ctx context.Context, err error) error {
	var tooLarge *streamRecordTooLargeError
	if errors.As(err, &tooLarge) {
		return unusableResponse("the anthropic stream sent a record over the size limit", err)
	}
	return transportFailure(ctx, "the anthropic stream broke off", err)
}

// anthropicTokenCountLabel is the gateway's prefix for token count
// diagnostics.
const anthropicTokenCountLabel = "anthropic token count"

// CountTokens implements core.TokenCounter for Messages, through
// /v1/messages/count_tokens as the gateway counts: the body with its model
// set to the request's, authenticated as its Messages request would be,
// without a preamble. Of the request's headers, anthropic-version replaces
// the default and each anthropic-beta line is added, but only values the
// gateway forwards: nonblank, without control characters. Any other surface
// fails with core.ErrTokenCountUnsupported.
func (p *Anthropic) CountTokens(ctx context.Context, request core.TokenCountRequest) (core.TokenCount, error) {
	if request.Surface != core.ModelSurfaceMessages {
		return core.TokenCount{}, core.ErrTokenCountUnsupported
	}
	auth, err := anthropicCredential(request.Credential)
	if err != nil {
		return core.TokenCount{}, err
	}
	payload, err := anthropicPayload(request.Request)
	if err != nil {
		return core.TokenCount{}, err
	}
	payload["model"] = request.Model
	protocol := func(header http.Header) {
		if version := request.Header.Get("anthropic-version"); anthropicHeaderValue(version) {
			header.Set("anthropic-version", strings.TrimSpace(version))
		}
		for _, beta := range request.Header.Values("anthropic-beta") {
			if anthropicHeaderValue(beta) {
				header.Add("anthropic-beta", strings.TrimSpace(beta))
			}
		}
	}
	call := anthropicCall{auth: auth, payload: payload}
	response, err := p.post(ctx, call, "/v1/messages/count_tokens", anthropicTokenCountLabel, protocol)
	if err != nil {
		return core.TokenCount{}, err
	}
	defer response.Body.Close()
	raw, err := readInvocationResponseBody(ctx, response, anthropicTokenCountLabel)
	if err != nil {
		return core.TokenCount{}, err
	}
	var result struct {
		InputTokens json.Number `json:"input_tokens"`
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if decoder.Decode(&result) != nil || !anthropicDigits(result.InputTokens.String()) {
		return core.TokenCount{}, unusableResponse("anthropic returned an invalid token count", nil)
	}
	count, err := strconv.ParseInt(result.InputTokens.String(), 10, 64)
	if err != nil {
		return core.TokenCount{}, unusableResponse("anthropic returned an invalid token count", err)
	}
	return core.TokenCount{InputTokens: count}, nil
}

// anthropicHeaderValue reports a protocol header value the gateway
// forwards to a token count: nonblank, with no control character.
func anthropicHeaderValue(value string) bool {
	value = strings.TrimSpace(value)
	if value == "" {
		return false
	}
	for _, r := range value {
		if r < 0x20 || r == 0x7f {
			return false
		}
	}
	return true
}

// anthropicDigits reports a count the gateway accepts: decimal digits only,
// so neither a sign, a fraction nor an exponent.
func anthropicDigits(value string) bool {
	if value == "" {
		return false
	}
	for _, r := range value {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
