package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	auth "github.com/xibodev/llm-provider-auth"
	codexauth "github.com/xibodev/llm-provider-auth/codex"
	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

const (
	codexMaxCatalogBytes  = 4 << 20
	codexMaxSSEEventBytes = 4 << 20
	codexMaxErrorBytes    = 64 << 10
	codexCatalogTTL       = time.Hour
	codexOpenAIGoVersion  = "3.22.0"
)

var codexErrorIdentifier = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,128}$`)

// CodexSession is the storage-neutral authenticated state needed by the
// official Codex transport. The source remains responsible for refresh and
// persistence; the transport asks it for current state on every operation.
type CodexSession struct {
	Token     *auth.Token
	AccountID string
}

type CodexSessionSource interface {
	Session(context.Context) (CodexSession, error)
}

type CodexSessionSourceFunc func(context.Context) (CodexSession, error)

func (f CodexSessionSourceFunc) Session(ctx context.Context) (CodexSession, error) {
	return f(ctx)
}

// NewCodexTokenSessionSource adapts llm-provider-auth's TokenSource for callers
// whose optional ChatGPT account identity is fixed outside token storage.
func NewCodexTokenSessionSource(source auth.TokenSource, accountID string) CodexSessionSource {
	return CodexSessionSourceFunc(func(ctx context.Context) (CodexSession, error) {
		if source == nil {
			return CodexSession{}, errors.New("Codex token source is required")
		}
		token, err := source.Token(ctx)
		return CodexSession{Token: token, AccountID: accountID}, err
	})
}

type CodexProviderConfig struct {
	SessionSource CodexSessionSource
	Instructions  string
	// ResponsesURL and ModelsURL default to llm-provider-auth's canonical
	// Codex endpoints.
	ResponsesURL string
	ModelsURL    string
	// ClientVersion is sent to the Codex catalog as client_version. It is
	// required because it names the product making the call, so no library
	// can supply an honest default.
	ClientVersion string
	Client        *http.Client
	Now           func() time.Time
}

// CodexProvider is the authenticated official Codex Responses transport.
type CodexProvider struct {
	sessions      CodexSessionSource
	instructions  string
	responsesURL  string
	modelsURL     string
	clientVersion string
	client        *http.Client
	now           func() time.Time
}

func NewCodexProvider(config CodexProviderConfig) (*CodexProvider, error) {
	if config.SessionSource == nil {
		return nil, fmt.Errorf("Codex session source is required")
	}
	if strings.TrimSpace(config.Instructions) == "" {
		return nil, fmt.Errorf("Codex instructions are required")
	}
	if strings.TrimSpace(config.ClientVersion) == "" {
		return nil, fmt.Errorf("Codex client version is required")
	}
	responsesURL := strings.TrimRight(config.ResponsesURL, "/")
	if responsesURL == "" {
		responsesURL = strings.TrimRight(codexauth.ResponsesBaseURL, "/") + "/responses"
	}
	modelsURL := config.ModelsURL
	if modelsURL == "" {
		modelsURL = codexauth.ModelsURL
	}
	client := config.Client
	if client == nil {
		client = &http.Client{Timeout: 90 * time.Second}
	}
	now := config.Now
	if now == nil {
		now = time.Now
	}
	return &CodexProvider{
		sessions: config.SessionSource, instructions: config.Instructions,
		responsesURL: responsesURL, modelsURL: modelsURL,
		clientVersion: config.ClientVersion, client: client, now: now,
	}, nil
}

func (p *CodexProvider) Complete(ctx context.Context, model string, payload map[string]any, _ *core.Credential) (map[string]any, error) {
	request, err := p.responsesRequest(model, payload)
	if err != nil {
		return nil, err
	}
	response, err := p.doResponses(ctx, request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	final, err := readCodexFinalResponse(response.Body)
	if err != nil {
		return nil, invocationDecodeError(err)
	}
	chat, err := codexResponseToChat(model, final)
	if err != nil {
		return nil, invocationDecodeError(err)
	}
	return chat, nil
}

// CompleteResponses executes a native Responses request. Codex always streams
// upstream, so this method buffers only until the terminal response event.
func (p *CodexProvider) CompleteResponses(ctx context.Context, model string, payload map[string]any, _ *core.Credential) (map[string]any, error) {
	request, err := p.nativeResponsesRequest(model, payload)
	if err != nil {
		return nil, err
	}
	response, err := p.doResponses(ctx, request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	final, err := readCodexFinalResponse(response.Body)
	if err != nil {
		return nil, invocationDecodeError(err)
	}
	return final, nil
}

func (p *CodexProvider) Stream(ctx context.Context, model string, payload map[string]any, _ *core.Credential) (StreamIter, error) {
	request, err := p.responsesRequest(model, payload)
	if err != nil {
		return nil, err
	}
	response, err := p.doResponses(ctx, request)
	if err != nil {
		return nil, err
	}
	return newCodexStreamIter(response.Body, model), nil
}

// StreamResponses executes a native Responses request and returns its SSE
// events without translating them through Chat Completions.
func (p *CodexProvider) StreamResponses(ctx context.Context, model string, payload map[string]any, _ *core.Credential) (StreamIter, error) {
	request, err := p.nativeResponsesRequest(model, payload)
	if err != nil {
		return nil, err
	}
	response, err := p.doResponses(ctx, request)
	if err != nil {
		return nil, err
	}
	return newCodexResponsesStreamIter(response.Body), nil
}

func (p *CodexProvider) ListModels(ctx context.Context, _ *core.Credential) ([]core.ModelInfo, error) {
	endpoint, err := url.Parse(p.modelsURL)
	if err != nil {
		return nil, &CatalogError{Code: "invalid_endpoint", Detail: "Codex catalog endpoint is invalid", Cause: err}
	}
	query := endpoint.Query()
	query.Set("client_version", p.clientVersion)
	endpoint.RawQuery = query.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint.String(), nil)
	if err != nil {
		return nil, &CatalogError{Code: "request_creation_failed", Detail: "Codex catalog request could not be created", Cause: err}
	}
	if err := p.authorize(ctx, req); err != nil {
		return nil, &CatalogError{Code: "authentication_failed", Detail: "Codex catalog authentication failed", Cause: err}
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, &CatalogError{Code: "transport_error", Detail: "Codex catalog transport failed", Cause: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, catalogHTTPError(resp)
	}
	raw, err := readLimited(resp.Body, codexMaxCatalogBytes)
	if err != nil {
		return nil, &CatalogError{Code: "invalid_response", Detail: "Codex catalog response is invalid", Status: resp.StatusCode, Cause: err}
	}
	models, err := parseCodexCatalog(raw, p.now().UTC())
	if err != nil {
		return nil, &CatalogError{Code: "invalid_response", Detail: "Codex catalog response is invalid", Status: resp.StatusCode, Cause: err}
	}
	return models, nil
}

func (p *CodexProvider) responsesRequest(model string, payload map[string]any) (map[string]any, error) {
	if model == "" || strings.TrimSpace(model) != model {
		return nil, &InvocationError{Msg: "Codex model ID is required"}
	}
	rawMessages, ok := payload["messages"]
	if !ok {
		return nil, &InvocationError{Msg: "Codex messages are required"}
	}
	messages, err := codexMessages(rawMessages)
	if err != nil {
		return nil, &InvocationError{Msg: "Codex messages are invalid", Cause: err}
	}
	options := make(map[string]any, len(payload))
	for key, value := range payload {
		switch key {
		case "messages", "model", "stream", "prompt_cache_key":
		case "tools":
			options[key] = value
		default:
			if value != nil {
				return nil, &InvocationError{Msg: "Codex request contains unsupported field " + key}
			}
		}
	}
	converted := translate.ChatToResponsesWithReport(model, messages, options, true)
	if err := translate.RejectMaterialLoss(withoutThoughtSignatures(converted.Report)); err != nil {
		return nil, &InvocationError{Msg: "Codex request contains unsupported fields", Cause: err}
	}
	request := converted.Value
	if err := validateCodexTools(request["tools"]); err != nil {
		return nil, &InvocationError{Msg: "Codex request contains unsupported tools", Cause: err}
	}
	if existing, _ := request["instructions"].(string); existing != "" {
		request["instructions"] = p.instructions + "\n\n" + existing
	} else {
		request["instructions"] = p.instructions
	}
	request["model"] = model
	request["stream"] = true
	request["store"] = false
	if cacheKey, _ := payload["prompt_cache_key"].(string); cacheKey != "" {
		request["prompt_cache_key"] = cacheKey
	}
	return request, nil
}

func (p *CodexProvider) nativeResponsesRequest(model string, payload map[string]any) (map[string]any, error) {
	if model == "" || strings.TrimSpace(model) != model {
		return nil, &InvocationError{Msg: "Codex model ID is required"}
	}
	if payload == nil || payload["input"] == nil {
		return nil, &InvocationError{Msg: "Codex Responses input is required"}
	}
	if store, ok := payload["store"].(bool); ok && store {
		return nil, &InvocationError{Msg: "Codex Responses does not support store=true"}
	}
	if background, ok := payload["background"].(bool); ok && background {
		return nil, &InvocationError{Msg: "Codex Responses does not support background=true"}
	}
	request := make(map[string]any, len(payload)+3)
	allowed := map[string]bool{
		"input": true, "instructions": true, "tools": true, "prompt_cache_key": true,
		"previous_response_id": true, "conversation": true, "include": true, "reasoning": true,
		"model": true, "stream": true, "store": true, "background": true,
		"force_api_support": true,
	}
	for key, value := range payload {
		if value != nil && !allowed[key] {
			return nil, &InvocationError{Msg: "Codex Responses request contains unsupported field " + key}
		}
	}
	for _, key := range []string{
		"input", "instructions", "tools", "prompt_cache_key",
		"previous_response_id", "conversation", "include", "reasoning",
	} {
		if value := payload[key]; value != nil {
			request[key] = value
		}
	}
	if err := validateCodexTools(request["tools"]); err != nil {
		return nil, &InvocationError{Msg: "Codex Responses request contains unsupported tools", Cause: err}
	}
	if instructions, exists := request["instructions"]; exists && instructions != nil {
		text, ok := instructions.(string)
		if !ok {
			return nil, &InvocationError{Msg: "Codex Responses instructions must be a string"}
		}
		if text != "" {
			request["instructions"] = p.instructions + "\n\n" + text
		} else {
			request["instructions"] = p.instructions
		}
	} else {
		request["instructions"] = p.instructions
	}
	request["model"] = model
	request["stream"] = true
	request["store"] = false
	return request, nil
}

// withoutThoughtSignatures removes dropped Gemini thought signatures from a
// request's loss report. llm-translate counts them as material because
// Gemini needs a signature back on its next turn, but that turn is built
// from the caller's history, which keeps it, and Codex has no use for one.
// Codex served such histories before the loss was reported and still does.
func withoutThoughtSignatures(report translate.Report) translate.Report {
	losses := make([]translate.Loss, 0, len(report.Losses))
	for _, loss := range report.Losses {
		if loss.Class != translate.LossDropped || !strings.HasSuffix(loss.Path, ".thought_signature") {
			losses = append(losses, loss)
		}
	}
	return translate.Report{Losses: losses}
}

func codexMessages(value any) ([]map[string]any, error) {
	switch messages := value.(type) {
	case []map[string]any:
		return messages, nil
	case []any:
		out := make([]map[string]any, len(messages))
		for i, value := range messages {
			message, ok := value.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("message %d is not an object", i)
			}
			out[i] = message
		}
		return out, nil
	default:
		return nil, fmt.Errorf("messages must be an array")
	}
}

func validateCodexTools(value any) error {
	if value == nil {
		return nil
	}
	tools, ok := value.([]any)
	if !ok {
		return fmt.Errorf("tools must be an array")
	}
	for i, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			return fmt.Errorf("tool %d must be an object", i)
		}
		kind, _ := tool["type"].(string)
		if kind != "function" && kind != "web_search" {
			return fmt.Errorf("tool %d has unsupported type %q", i, kind)
		}
	}
	return nil
}

func (p *CodexProvider) doResponses(ctx context.Context, payload map[string]any) (*http.Response, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, &InvocationError{Msg: "Codex request encoding failed", Cause: err}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.responsesURL, bytes.NewReader(body))
	if err != nil {
		return nil, &InvocationError{Msg: "Codex request could not be created", Cause: err}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	setCodexSDKHeaders(req.Header)
	if err := p.authorize(ctx, req); err != nil {
		return nil, &InvocationError{Msg: "Codex authentication failed", Cause: err}
	}
	resp, err := p.client.Do(req)
	if err != nil {
		upstreamFailure := ctx.Err() == nil
		return nil, &InvocationError{
			Msg: "Codex transport failed", Retryable: upstreamFailure,
			FailoverEligible: upstreamFailure, CircuitFailure: upstreamFailure, Cause: err,
		}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		return nil, invocationHTTPError(resp)
	}
	mediaType := strings.TrimSpace(strings.Split(resp.Header.Get("Content-Type"), ";")[0])
	if mediaType != "" && !strings.EqualFold(mediaType, "text/event-stream") {
		resp.Body.Close()
		return nil, invocationDecodeError(fmt.Errorf("Codex response content type %q is not text/event-stream", mediaType))
	}
	return resp, nil
}

func (p *CodexProvider) authorize(ctx context.Context, req *http.Request) error {
	session, err := p.sessions.Session(ctx)
	if err != nil {
		return err
	}
	if session.Token == nil || !session.Token.Valid() {
		return errors.New("Codex session has no valid access token")
	}
	tokenType := strings.TrimSpace(session.Token.TokenType)
	if tokenType == "" {
		tokenType = "Bearer"
	}
	req.Header.Set("Authorization", tokenType+" "+session.Token.AccessToken)
	req.Header.Set("OpenAI-Beta", "responses=experimental")
	req.Header.Set("originator", "codex_cli_rs")
	if session.AccountID != "" {
		req.Header.Set("Chatgpt-Account-Id", session.AccountID)
	}
	return nil
}

func setCodexSDKHeaders(header http.Header) {
	header.Set("User-Agent", "OpenAI/Go "+codexOpenAIGoVersion)
	header.Set("X-Stainless-Lang", "go")
	header.Set("X-Stainless-Package-Version", codexOpenAIGoVersion)
	header.Set("X-Stainless-OS", codexSDKOS())
	header.Set("X-Stainless-Arch", codexSDKArch())
	header.Set("X-Stainless-Runtime", "go")
	header.Set("X-Stainless-Runtime-Version", runtime.Version())
	header.Set("X-Stainless-Retry-Count", "0")
}

func codexSDKOS() string {
	switch runtime.GOOS {
	case "ios":
		return "iOS"
	case "android":
		return "Android"
	case "darwin":
		return "MacOS"
	case "freebsd":
		return "FreeBSD"
	case "openbsd":
		return "OpenBSD"
	case "linux":
		return "Linux"
	default:
		return "Other:" + runtime.GOOS
	}
}

func codexSDKArch() string {
	switch runtime.GOARCH {
	case "386":
		return "x32"
	case "amd64":
		return "x64"
	case "arm", "arm64":
		return runtime.GOARCH
	default:
		return "other:" + runtime.GOARCH
	}
}

func parseCodexCatalog(raw []byte, discoveredAt time.Time) ([]core.ModelInfo, error) {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, err
	}
	data, hasData := envelope["data"]
	models, hasModels := envelope["models"]
	if hasData == hasModels {
		return nil, fmt.Errorf("catalog must contain exactly one of data or models")
	}
	rowsRaw := data
	if hasModels {
		rowsRaw = models
	}
	var rows []map[string]json.RawMessage
	if err := json.Unmarshal(rowsRaw, &rows); err != nil || rows == nil {
		return nil, fmt.Errorf("catalog model envelope must be an array")
	}
	result := make([]core.ModelInfo, 0, len(rows))
	seen := make(map[string]struct{}, len(rows))
	for _, row := range rows {
		id, err := exactCatalogIdentity(row)
		if err != nil {
			return nil, err
		}
		if _, duplicate := seen[id]; duplicate {
			return nil, fmt.Errorf("duplicate model identity %q", id)
		}
		seen[id] = struct{}{}
		model := core.ModelInfo{ID: id, Object: "model", OwnedBy: "openai"}
		decodeOptionalString(row, "object", &model.Object)
		decodeOptionalString(row, "owned_by", &model.OwnedBy)
		decodeOptionalString(row, "description", &model.Description)
		if value, ok := row["supported_in_api"]; ok {
			var eligible bool
			if err := json.Unmarshal(value, &eligible); err != nil {
				return nil, fmt.Errorf("model %q supported_in_api must be boolean", id)
			}
			model.APIEligible = &eligible
		}
		if value, ok := row["visibility"]; ok {
			if err := json.Unmarshal(value, &model.APIVisibility); err != nil || model.APIVisibility == "" || strings.TrimSpace(model.APIVisibility) != model.APIVisibility {
				return nil, fmt.Errorf("model %q visibility must be a nonempty exact string", id)
			}
		}
		if value, ok := row["supported_endpoints"]; ok {
			if err := json.Unmarshal(value, &model.SupportedAPIs); err != nil {
				return nil, fmt.Errorf("model %q supported_endpoints must be a string array", id)
			}
			for _, endpoint := range model.SupportedAPIs {
				if endpoint == "" || strings.TrimSpace(endpoint) != endpoint {
					return nil, fmt.Errorf("model %q has invalid supported endpoint", id)
				}
			}
		}
		if model.APIEligible != nil && *model.APIEligible {
			model.Capabilities = codexModelCapabilities(discoveredAt)
		}
		result = append(result, model)
	}
	return result, nil
}

func codexModelCapabilities(discoveredAt time.Time) *core.ModelCapabilities {
	expiresAt := discoveredAt.Add(codexCatalogTTL)
	return &core.ModelCapabilities{
		SchemaVersion: core.ModelCapabilitiesSchemaVersion,
		Operations:    core.ModelOperationCapabilities{Chat: core.SupportSupported},
		Surfaces: core.ModelSurfaceCapabilities{
			ChatCompletions: core.SupportUnsupported,
			Responses:       core.SupportSupported,
		},
		Streaming: core.SupportSupported,
		Provenance: core.ModelCapabilityProvenance{
			Source: core.ModelCapabilitySourceUpstreamReported, Confidence: core.ModelCapabilityConfidenceHigh,
		},
		Freshness: core.ModelCapabilityFreshness{DiscoveredAt: &discoveredAt, ExpiresAt: &expiresAt},
	}
}

func exactCatalogIdentity(row map[string]json.RawMessage) (string, error) {
	for _, field := range []string{"id", "slug", "name"} {
		value, exists := row[field]
		if !exists || bytes.Equal(value, []byte("null")) {
			continue
		}
		var id string
		if err := json.Unmarshal(value, &id); err != nil {
			return "", fmt.Errorf("model %s identity must be a string", field)
		}
		if id == "" || strings.TrimSpace(id) != id {
			return "", fmt.Errorf("model %s identity is empty or not exact", field)
		}
		return id, nil
	}
	return "", fmt.Errorf("model identity is missing")
}

func decodeOptionalString(row map[string]json.RawMessage, key string, destination *string) {
	if value, ok := row[key]; ok {
		var decoded string
		if json.Unmarshal(value, &decoded) == nil && decoded != "" {
			*destination = decoded
		}
	}
}

type codexSSEEvent struct {
	Type        string          `json:"type"`
	Delta       string          `json:"delta"`
	ItemID      string          `json:"item_id"`
	OutputIndex int             `json:"output_index"`
	Response    map[string]any  `json:"response"`
	Item        map[string]any  `json:"item"`
	Raw         json.RawMessage `json:"-"`
	Frame       []byte          `json:"-"`
	FrameClosed bool            `json:"-"`
}

type codexSSEReader struct {
	reader *bufio.Reader
}

func newCodexSSEReader(reader io.Reader) *codexSSEReader {
	return &codexSSEReader{reader: bufio.NewReader(reader)}
}

func (r *codexSSEReader) Next() (codexSSEEvent, error) {
	var data []string
	var frame []byte
	hasFields := false
	size := 0
	for {
		line, err := r.reader.ReadString('\n')
		size += len(line)
		if size > codexMaxSSEEventBytes {
			return codexSSEEvent{}, fmt.Errorf("SSE event exceeds size limit")
		}
		if err != nil && err != io.EOF {
			return codexSSEEvent{}, err
		}
		frame = append(frame, line...)
		parsedLine := strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
		if parsedLine == "" {
			if len(data) > 0 {
				event, decodeErr := decodeCodexSSEEvent(strings.Join(data, "\n"))
				event.Frame = bytesClone(frame)
				event.FrameClosed = true
				return event, decodeErr
			}
			if hasFields {
				return codexSSEEvent{Frame: bytesClone(frame), FrameClosed: true}, nil
			}
			frame = frame[:0]
			size = 0
			hasFields = false
		} else {
			hasFields = true
			if strings.HasPrefix(parsedLine, ":") {
				continue
			}
			field, value, found := strings.Cut(parsedLine, ":")
			if !found {
				value = ""
			}
			if strings.HasPrefix(value, " ") {
				value = value[1:]
			}
			if field == "data" {
				data = append(data, value)
			}
		}
		if err == io.EOF {
			if len(data) > 0 {
				event, decodeErr := decodeCodexSSEEvent(strings.Join(data, "\n"))
				event.Frame = bytesClone(frame)
				return event, decodeErr
			}
			return codexSSEEvent{}, io.EOF
		}
	}
}

func decodeCodexSSEEvent(data string) (codexSSEEvent, error) {
	if data == "[DONE]" {
		return codexSSEEvent{}, fmt.Errorf("Codex stream ended without a terminal response event")
	}
	var event codexSSEEvent
	if err := json.Unmarshal([]byte(data), &event); err != nil {
		return event, fmt.Errorf("malformed Codex SSE event: %w", err)
	}
	if event.Type == "" {
		return event, fmt.Errorf("Codex SSE event type is missing")
	}
	event.Raw = append(event.Raw[:0], data...)
	return event, nil
}

type codexResponsesStreamIter struct {
	body   io.ReadCloser
	reader *codexSSEReader
	done   bool
}

func newCodexResponsesStreamIter(body io.ReadCloser) *codexResponsesStreamIter {
	return &codexResponsesStreamIter{body: body, reader: newCodexSSEReader(body)}
}

func (s *codexResponsesStreamIter) Next() ([]byte, error) {
	if s.done {
		return nil, io.EOF
	}
	event, err := s.reader.Next()
	if err != nil {
		if err == io.EOF {
			err = fmt.Errorf("Codex SSE stream ended without a terminal response event")
		}
		return nil, invocationDecodeError(err)
	}
	if !event.FrameClosed {
		return nil, invocationDecodeError(fmt.Errorf("Codex SSE stream ended with an incomplete event frame"))
	}
	if event.Type == "response.completed" || event.Type == "response.incomplete" {
		if err := validateTerminalResponse(event.Response, event.Type); err != nil {
			return nil, invocationDecodeError(err)
		}
		s.done = true
	} else if event.Type == "response.failed" || event.Type == "error" {
		s.done = true
	}
	return event.Frame, nil
}

func (s *codexResponsesStreamIter) Close() error { return s.body.Close() }

func readCodexFinalResponse(reader io.Reader) (map[string]any, error) {
	events := newCodexSSEReader(reader)
	output := make([]any, 0)
	var text strings.Builder
	for {
		event, err := events.Next()
		if err != nil {
			if err == io.EOF {
				return nil, fmt.Errorf("Codex SSE stream ended without a terminal response event")
			}
			return nil, err
		}
		if event.Type == "" {
			continue
		}
		switch event.Type {
		case "response.output_text.delta":
			text.WriteString(event.Delta)
		case "response.output_item.done":
			if len(event.Item) > 0 {
				output = append(output, event.Item)
			}
		case "response.completed", "response.incomplete":
			terminalOutput, _ := event.Response["output"].([]any)
			if len(terminalOutput) == 0 && len(output) > 0 {
				event.Response["output"] = output
				terminalOutput = output
			}
			if len(terminalOutput) == 0 && text.Len() > 0 {
				event.Response["output"] = []any{map[string]any{
					"type": "message", "role": "assistant",
					"content": []any{map[string]any{"type": "output_text", "text": text.String()}},
				}}
			}
			if err := validateTerminalResponse(event.Response, event.Type); err != nil {
				return nil, err
			}
			if _, ok := event.Response["output"].([]any); !ok {
				return nil, fmt.Errorf("terminal Codex response output is missing")
			}
			return event.Response, nil
		case "response.failed", "error":
			return nil, fmt.Errorf("Codex response failed")
		default:
			continue
		}
	}
}

func validateTerminalResponse(response map[string]any, eventType string) error {
	if len(response) == 0 {
		return fmt.Errorf("%s did not contain a response", eventType)
	}
	if id, _ := response["id"].(string); strings.TrimSpace(id) == "" {
		return fmt.Errorf("%s did not contain a response ID", eventType)
	}
	if status, _ := response["status"].(string); status != "" {
		want := strings.TrimPrefix(eventType, "response.")
		if status != want {
			return fmt.Errorf("%s carried a contradictory response status", eventType)
		}
	}
	return nil
}

type codexStreamIter struct {
	body         io.ReadCloser
	reader       *codexSSEReader
	model        string
	queue        [][]byte
	started      bool
	terminal     bool
	doneQueued   bool
	toolOrdinals map[string]int
	nextTool     int
}

func newCodexStreamIter(body io.ReadCloser, model string) *codexStreamIter {
	return &codexStreamIter{body: body, reader: newCodexSSEReader(body), model: model, toolOrdinals: map[string]int{}}
}

func (s *codexStreamIter) Next() ([]byte, error) {
	for len(s.queue) == 0 {
		if s.terminal && !s.doneQueued {
			s.doneQueued = true
			return []byte("data: [DONE]\n\n"), nil
		}
		if s.terminal {
			return nil, io.EOF
		}
		event, err := s.reader.Next()
		if err != nil {
			if err == io.EOF {
				err = fmt.Errorf("Codex SSE stream ended without a terminal response event")
			}
			return nil, invocationDecodeError(err)
		}
		if err := s.consume(event); err != nil {
			return nil, invocationDecodeError(err)
		}
	}
	chunk := s.queue[0]
	s.queue = s.queue[1:]
	return chunk, nil
}

func (s *codexStreamIter) consume(event codexSSEEvent) error {
	switch event.Type {
	case "":
		return nil
	case "response.created", "response.in_progress", "response.output_item.done",
		"response.content_part.added", "response.content_part.done",
		"response.output_text.done", "response.function_call_arguments.done",
		"response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
		"response.reasoning_summary_text.done", "response.reasoning_text.done":
		return nil
	case "response.output_text.delta":
		s.emit(map[string]any{"content": event.Delta}, nil, nil)
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		s.emit(map[string]any{"reasoning_content": event.Delta}, nil, nil)
	case "response.output_item.added":
		switch event.Item["type"] {
		case "message", "reasoning":
			return nil
		case "function_call":
			id, _ := event.Item["call_id"].(string)
			if id == "" {
				return fmt.Errorf("Codex function call ID is missing")
			}
			name, _ := event.Item["name"].(string)
			ordinal := s.nextTool
			s.nextTool++
			s.toolOrdinals[id] = ordinal
			if itemID, _ := event.Item["id"].(string); itemID != "" {
				s.toolOrdinals[itemID] = ordinal
			}
			s.emit(map[string]any{"tool_calls": []any{map[string]any{
				"index": ordinal, "id": id, "type": "function",
				"function": map[string]any{"name": name, "arguments": ""},
			}}}, nil, nil)
		default:
			return fmt.Errorf("unsupported Codex output item type %q", event.Item["type"])
		}
	case "response.function_call_arguments.delta":
		ordinal, ok := s.toolOrdinals[event.ItemID]
		if !ok {
			return fmt.Errorf("Codex function argument delta has no matching call")
		}
		s.emit(map[string]any{"tool_calls": []any{map[string]any{
			"index": ordinal, "function": map[string]any{"arguments": event.Delta},
		}}}, nil, nil)
	case "response.completed", "response.incomplete":
		if err := validateTerminalResponse(event.Response, event.Type); err != nil {
			return err
		}
		finish := any("stop")
		if event.Type == "response.incomplete" {
			finish = "length"
		} else if s.nextTool > 0 {
			finish = "tool_calls"
		}
		chat, err := codexResponseToChat(s.model, event.Response)
		if err != nil {
			return err
		}
		usage := chat["usage"]
		s.emit(map[string]any{}, finish, usage)
		s.terminal = true
	case "response.failed", "error":
		return fmt.Errorf("Codex response failed")
	default:
		return fmt.Errorf("unsupported Codex streaming event %q", event.Type)
	}
	return nil
}

func (s *codexStreamIter) emit(delta map[string]any, finish, usage any) {
	if !s.started {
		s.started = true
		s.appendChunk(map[string]any{"role": "assistant"}, nil, nil)
	}
	s.appendChunk(delta, finish, usage)
}

func (s *codexStreamIter) appendChunk(delta map[string]any, finish, usage any) {
	chunk := map[string]any{
		"object": "chat.completion.chunk", "model": s.model,
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
	}
	if usage != nil {
		chunk["usage"] = usage
	}
	encoded, _ := json.Marshal(chunk)
	s.queue = append(s.queue, append(append([]byte("data: "), encoded...), []byte("\n\n")...))
}

func (s *codexStreamIter) Close() error { return s.body.Close() }

func codexResponseToChat(model string, response map[string]any) (map[string]any, error) {
	converted := translate.ResponsesToChatWithReport(model, response)
	if err := converted.RejectMaterialLoss(); err != nil {
		return nil, fmt.Errorf("Codex response contains unsupported output: %w", err)
	}
	chat := converted.Value
	var reasoning strings.Builder
	for _, raw := range anySlice(response["output"]) {
		item, ok := raw.(map[string]any)
		if !ok || item["type"] != "reasoning" {
			continue
		}
		for _, rawSummary := range anySlice(item["summary"]) {
			summary, ok := rawSummary.(map[string]any)
			if !ok {
				continue
			}
			if text, _ := summary["text"].(string); text != "" {
				reasoning.WriteString(text)
			}
		}
	}
	if reasoning.Len() == 0 {
		return chat, nil
	}
	choices := anySlice(chat["choices"])
	if len(choices) == 0 {
		return chat, nil
	}
	choice, _ := choices[0].(map[string]any)
	message, _ := choice["message"].(map[string]any)
	if message != nil {
		message["reasoning_content"] = reasoning.String()
	}
	return chat, nil
}

func anySlice(value any) []any {
	values, _ := value.([]any)
	return values
}

func invocationDecodeError(err error) error {
	return &InvocationError{Msg: "Codex streaming response is invalid", CircuitFailure: true, Cause: err}
}

func invocationHTTPError(response *http.Response) error {
	retryAfter := parseRetryAfterHeader(response.Header.Get("Retry-After"), time.Now())
	status := response.StatusCode
	message := "Codex upstream request failed"
	if raw, err := readLimited(response.Body, codexMaxErrorBytes); err == nil {
		if diagnostic := codexErrorDiagnostic(raw); diagnostic != "" {
			message += " (" + diagnostic + ")"
		}
	}
	return &InvocationError{
		Msg: message, Status: status, RetryAfter: retryAfter,
		Retryable:        status == http.StatusRequestTimeout || status == http.StatusTooManyRequests || status >= 500,
		FailoverEligible: status == http.StatusTooManyRequests || status >= 500,
	}
}

func codexErrorDiagnostic(raw []byte) string {
	var envelope struct {
		Error struct {
			Code  any    `json:"code"`
			Type  string `json:"type"`
			Param string `json:"param"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &envelope) != nil {
		return ""
	}
	values := []struct {
		name  string
		value string
	}{
		{name: "code", value: fmt.Sprint(envelope.Error.Code)},
		{name: "type", value: envelope.Error.Type},
		{name: "param", value: envelope.Error.Param},
	}
	parts := make([]string, 0, len(values))
	for _, value := range values {
		if value.value != "" && value.value != "<nil>" && codexErrorIdentifier.MatchString(value.value) {
			parts = append(parts, value.name+"="+value.value)
		}
	}
	return strings.Join(parts, ", ")
}

func catalogHTTPError(response *http.Response) error {
	return &CatalogError{
		Code: "http_error", Detail: "Codex catalog request failed", Status: response.StatusCode,
		RetryAfter: parseRetryAfterHeader(response.Header.Get("Retry-After"), time.Now()),
	}
}

func parseRetryAfterHeader(value string, now time.Time) time.Duration {
	if seconds, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64); err == nil && seconds >= 0 {
		return time.Duration(seconds) * time.Second
	}
	if when, err := http.ParseTime(value); err == nil && when.After(now) {
		return when.Sub(now)
	}
	return 0
}

func readLimited(reader io.Reader, limit int64) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > limit {
		return nil, fmt.Errorf("response exceeds size limit")
	}
	return raw, nil
}

var _ Provider = (*CodexProvider)(nil)
var _ ResponsesProvider = (*CodexProvider)(nil)
var _ ResponsesStreamProvider = (*CodexProvider)(nil)
