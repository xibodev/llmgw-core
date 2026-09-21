package providers

// This file adapts the Google Cloud Code Assist protocol from xibodev/facet-studio
// commit 7c768444ac5704b07ffe0efc1cd0889a471e624f under the MIT License. See NOTICE.

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

const (
	antigravityDefaultBaseURL = "https://cloudcode-pa.googleapis.com"
	antigravityUserAgent      = "antigravity/1.15.8"
	antigravityGoogAPIClient  = "google-cloud-sdk vscode_cloudshelleditor/0.1"
	antigravityMaxBodyBytes   = 32 << 20
)

// ErrExperimentalAntigravityStreamingUnsupported reports that this adapter
// currently buffers Cloud Code Assist SSE and exposes only Complete.
var ErrExperimentalAntigravityStreamingUnsupported = errors.New("experimental antigravity streaming is unsupported")

// AntigravityTokenSource supplies a current access token and, when known, the
// Cloud AI companion project ID. Returning an empty project ID triggers
// loadCodeAssist discovery. The adapter never persists either value.
type AntigravityTokenSource func(ctx context.Context) (accessToken, projectID string, err error)

// AntigravityUnauthorizedHandler lets an owner recover a token rejected by the
// upstream service. The adapter invokes it at most once per operation.
type AntigravityUnauthorizedHandler func(ctx context.Context, rejectedAccessToken string) error

// AntigravityProjectObserver reports project discovery without assuming how or
// whether the owner stores it.
type AntigravityProjectObserver func(ctx context.Context, accessToken, projectID string) error

// ExperimentalAntigravityProvider is a storage-neutral experimental adapter
// for Google's undocumented Cloud Code Assist v1internal API.
type ExperimentalAntigravityProvider struct {
	tokenSource  AntigravityTokenSource
	unauthorized AntigravityUnauthorizedHandler
	project      AntigravityProjectObserver
	client       *http.Client
	baseURL      string
}

// NewExperimentalAntigravityProvider constructs the opt-in experimental
// adapter. It does not implement OAuth, refresh tokens, or credential storage.
// A blank baseURL selects Google's Cloud Code Assist endpoint.
func NewExperimentalAntigravityProvider(tokenSource AntigravityTokenSource, client *http.Client, baseURL string) *ExperimentalAntigravityProvider {
	if client == nil {
		client = &http.Client{Timeout: 120 * time.Second}
	}
	if strings.TrimSpace(baseURL) == "" {
		baseURL = antigravityDefaultBaseURL
	}
	return &ExperimentalAntigravityProvider{
		tokenSource: tokenSource,
		client:      client,
		baseURL:     strings.TrimRight(baseURL, "/"),
	}
}

// SetUnauthorizedHandler configures one-shot recovery from an upstream 401.
func (p *ExperimentalAntigravityProvider) SetUnauthorizedHandler(handler AntigravityUnauthorizedHandler) {
	p.unauthorized = handler
}

// SetProjectObserver configures notification when loadCodeAssist discovers a
// project that was absent from the token source.
func (p *ExperimentalAntigravityProvider) SetProjectObserver(observer AntigravityProjectObserver) {
	p.project = observer
}

func (p *ExperimentalAntigravityProvider) ListModels(ctx context.Context, _ *core.Credential) ([]core.ModelInfo, error) {
	models, rejectedToken, err := p.listModels(ctx)
	recovered, recoveryErr := p.recoverUnauthorized(ctx, rejectedToken, err)
	if recoveryErr != nil {
		return nil, recoveryErr
	}
	if recovered {
		models, _, err = p.listModels(ctx)
	}
	return models, err
}

func (p *ExperimentalAntigravityProvider) listModels(ctx context.Context) ([]core.ModelInfo, string, error) {
	token, projectID, err := p.authentication(ctx)
	if err != nil {
		return nil, "", err
	}
	if projectID == "" {
		projectID, err = p.loadCodeAssist(ctx, token)
		if err != nil {
			return nil, token, err
		}
		if p.project != nil {
			if err := p.project(ctx, token, projectID); err != nil {
				return nil, token, fmt.Errorf("antigravity project observation: %w", err)
			}
		}
	}

	var response struct {
		Models map[string]struct {
			DisplayName string `json:"displayName"`
		} `json:"models"`
	}
	if err := p.postJSON(ctx, token, "/v1internal:fetchAvailableModels", map[string]any{
		"project": projectID,
	}, "antigravity model discovery", &response, false); err != nil {
		return nil, token, err
	}

	ids := make([]string, 0, len(response.Models))
	for id := range response.Models {
		if strings.TrimSpace(id) != "" {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	models := make([]core.ModelInfo, 0, len(ids))
	for _, id := range ids {
		models = append(models, core.ModelInfo{
			ID:          id,
			Object:      "model",
			OwnedBy:     "google-antigravity",
			Description: response.Models[id].DisplayName,
			Capabilities: &core.ModelCapabilities{
				SchemaVersion: core.ModelCapabilitiesSchemaVersion,
				Operations: core.ModelOperationCapabilities{
					Chat: core.SupportSupported,
				},
				Surfaces: core.ModelSurfaceCapabilities{
					ChatCompletions: core.SupportSupported,
				},
				Inputs: core.ModelInputCapabilities{
					Text: core.SupportSupported,
				},
				Tools:     core.SupportSupported,
				Reasoning: core.SupportSupported,
				Streaming: core.SupportUnknown,
				Provenance: core.ModelCapabilityProvenance{
					Source:     core.ModelCapabilitySourceInferred,
					Confidence: core.ModelCapabilityConfidenceMedium,
				},
			},
		})
	}
	return models, token, nil
}

func (p *ExperimentalAntigravityProvider) Complete(ctx context.Context, model string, payload map[string]any, _ *core.Credential) (map[string]any, error) {
	response, rejectedToken, err := p.complete(ctx, model, payload)
	recovered, recoveryErr := p.recoverUnauthorized(ctx, rejectedToken, err)
	if recoveryErr != nil {
		return nil, recoveryErr
	}
	if recovered {
		response, _, err = p.complete(ctx, model, payload)
	}
	return response, err
}

func (p *ExperimentalAntigravityProvider) complete(ctx context.Context, model string, payload map[string]any) (map[string]any, string, error) {
	if strings.TrimSpace(model) == "" {
		return nil, "", fmt.Errorf("antigravity model is required")
	}
	token, projectID, err := p.authentication(ctx)
	if err != nil {
		return nil, "", err
	}
	if projectID == "" {
		projectID, err = p.loadCodeAssist(ctx, token)
		if err != nil {
			return nil, token, err
		}
		if p.project != nil {
			if err := p.project(ctx, token, projectID); err != nil {
				return nil, token, fmt.Errorf("antigravity project observation: %w", err)
			}
		}
	}

	request, err := mapAntigravityRequest(payload)
	if err != nil {
		return nil, token, err
	}
	requestID := newAntigravityID("agent")
	envelope := map[string]any{
		"project":     projectID,
		"model":       model,
		"request":     request,
		"requestType": "agent",
		"userAgent":   "antigravity",
		"requestId":   requestID,
	}

	var body bytes.Buffer
	if err := p.postJSON(ctx, token, "/v1internal:streamGenerateContent?alt=sse", envelope, "antigravity completion", &body, true); err != nil {
		return nil, token, err
	}
	response, err := parseAntigravitySSE(&body, model, requestID)
	return response, token, err
}

func (p *ExperimentalAntigravityProvider) recoverUnauthorized(ctx context.Context, rejectedToken string, err error) (bool, error) {
	if p.unauthorized == nil || !antigravityHTTPStatus(err, http.StatusUnauthorized) {
		return false, nil
	}
	if recoveryErr := p.unauthorized(ctx, rejectedToken); recoveryErr != nil {
		return false, fmt.Errorf("antigravity unauthorized recovery: %w", recoveryErr)
	}
	return true, nil
}

func antigravityHTTPStatus(err error, status int) bool {
	var operationError *core.ProviderOperationError
	return errors.As(err, &operationError) && operationError.Failure.StatusCode == status
}

func (p *ExperimentalAntigravityProvider) Stream(context.Context, string, map[string]any, *core.Credential) (StreamIter, error) {
	return nil, ErrExperimentalAntigravityStreamingUnsupported
}

func (p *ExperimentalAntigravityProvider) authentication(ctx context.Context) (string, string, error) {
	if p.tokenSource == nil {
		return "", "", fmt.Errorf("antigravity token source is required")
	}
	token, projectID, err := p.tokenSource(ctx)
	if err != nil {
		return "", "", fmt.Errorf("antigravity token source: %w", err)
	}
	if strings.TrimSpace(token) == "" {
		return "", "", fmt.Errorf("antigravity token source returned an empty access token")
	}
	return token, strings.TrimSpace(projectID), nil
}

func (p *ExperimentalAntigravityProvider) loadCodeAssist(ctx context.Context, token string) (string, error) {
	var response struct {
		Project string `json:"cloudaicompanionProject"`
	}
	err := p.postJSON(ctx, token, "/v1internal:loadCodeAssist", map[string]any{
		"metadata": antigravityClientMetadata(),
	}, "antigravity project discovery", &response, false)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(response.Project) == "" {
		return "", fmt.Errorf("antigravity project discovery returned no project ID")
	}
	return response.Project, nil
}

func (p *ExperimentalAntigravityProvider) postJSON(ctx context.Context, token, path string, payload any, op string, result any, sse bool) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal antigravity request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create antigravity request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", antigravityUserAgent)
	req.Header.Set("X-Goog-Api-Client", antigravityGoogAPIClient)
	metadata, _ := json.Marshal(antigravityClientMetadata())
	req.Header.Set("Client-Metadata", string(metadata))
	if sse {
		req.Header.Set("Accept", "text/event-stream")
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return core.NewProviderOperationError(op, 0, "", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, antigravityMaxBodyBytes))
		return core.NewProviderOperationError(op, resp.StatusCode, resp.Header.Get("Retry-After"), nil)
	}

	limited := io.LimitReader(resp.Body, antigravityMaxBodyBytes)
	if destination, ok := result.(*bytes.Buffer); ok {
		if _, err := destination.ReadFrom(limited); err != nil {
			return core.NewProviderOperationError(op+" response read", resp.StatusCode, "", err)
		}
		return nil
	}
	if err := json.NewDecoder(limited).Decode(result); err != nil {
		return core.NewProviderOperationError(op+" response decode", resp.StatusCode, "", err)
	}
	return nil
}

func antigravityClientMetadata() map[string]string {
	return map[string]string{
		"ideType":    "IDE_UNSPECIFIED",
		"platform":   "PLATFORM_UNSPECIFIED",
		"pluginType": "GEMINI",
	}
}

type antigravityRequest struct {
	Contents          []antigravityContent   `json:"contents"`
	Tools             []antigravityTool      `json:"tools,omitempty"`
	SystemInstruction *antigravityContent    `json:"systemInstruction,omitempty"`
	GenerationConfig  *antigravityGeneration `json:"generationConfig,omitempty"`
}

type antigravityContent struct {
	Role  string            `json:"role,omitempty"`
	Parts []antigravityPart `json:"parts"`
}

type antigravityPart struct {
	Text             string                       `json:"text,omitempty"`
	Thought          bool                         `json:"thought,omitempty"`
	ThoughtSignature string                       `json:"thoughtSignature,omitempty"`
	FunctionCall     *antigravityFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *antigravityFunctionResponse `json:"functionResponse,omitempty"`
}

type antigravityFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type antigravityFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type antigravityTool struct {
	FunctionDeclarations []antigravityFunctionDeclaration `json:"functionDeclarations"`
}

type antigravityFunctionDeclaration struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type antigravityGeneration struct {
	MaxOutputTokens int     `json:"maxOutputTokens,omitempty"`
	Temperature     float64 `json:"temperature,omitempty"`
	HasTemperature  bool    `json:"-"`
}

func (g antigravityGeneration) MarshalJSON() ([]byte, error) {
	value := map[string]any{}
	if g.MaxOutputTokens > 0 {
		value["maxOutputTokens"] = g.MaxOutputTokens
	}
	if g.HasTemperature {
		value["temperature"] = g.Temperature
	}
	return json.Marshal(value)
}

func mapAntigravityRequest(payload map[string]any) (antigravityRequest, error) {
	request := antigravityRequest{}
	toolNames := map[string]string{}

	messages, ok := payload["messages"].([]any)
	if !ok {
		return request, fmt.Errorf("antigravity messages must be an array")
	}
	for i, raw := range messages {
		message, ok := raw.(map[string]any)
		if !ok {
			return request, fmt.Errorf("antigravity message %d must be an object", i)
		}
		role, _ := message["role"].(string)
		text, err := openAIMessageText(message["content"])
		if err != nil {
			return request, fmt.Errorf("antigravity message %d content: %w", i, err)
		}
		switch role {
		case "system", "developer":
			if text != "" {
				if request.SystemInstruction == nil {
					request.SystemInstruction = &antigravityContent{Parts: []antigravityPart{}}
				}
				request.SystemInstruction.Parts = append(request.SystemInstruction.Parts, antigravityPart{Text: text})
			}
		case "user":
			if text != "" {
				request.Contents = append(request.Contents, antigravityContent{Role: "user", Parts: []antigravityPart{{Text: text}}})
			}
		case "assistant":
			content := antigravityContent{Role: "model"}
			if text != "" {
				content.Parts = append(content.Parts, antigravityPart{Text: text})
			}
			if reasoning, _ := message["reasoning_content"].(string); reasoning != "" {
				content.Parts = append(content.Parts, antigravityPart{Text: reasoning, Thought: true})
			}
			calls, err := mapAssistantToolCalls(message["tool_calls"], toolNames)
			if err != nil {
				return request, fmt.Errorf("antigravity message %d tool calls: %w", i, err)
			}
			content.Parts = append(content.Parts, calls...)
			if len(content.Parts) > 0 {
				request.Contents = append(request.Contents, content)
			}
		case "tool":
			name, _ := message["name"].(string)
			if callID, _ := message["tool_call_id"].(string); name == "" {
				name = toolNames[callID]
			}
			if name == "" {
				return request, fmt.Errorf("antigravity message %d tool result has no resolvable function name", i)
			}
			request.Contents = append(request.Contents, antigravityContent{Role: "user", Parts: []antigravityPart{{
				FunctionResponse: &antigravityFunctionResponse{Name: name, Response: map[string]any{"result": text}},
			}}})
		default:
			return request, fmt.Errorf("antigravity message %d has unsupported role %q", i, role)
		}
	}

	if rawTools, exists := payload["tools"]; exists {
		tools, ok := rawTools.([]any)
		if !ok {
			return request, fmt.Errorf("antigravity tools must be an array")
		}
		declarations := make([]antigravityFunctionDeclaration, 0, len(tools))
		for i, raw := range tools {
			tool, ok := raw.(map[string]any)
			if !ok {
				return request, fmt.Errorf("antigravity tool %d must be an object", i)
			}
			if kind, _ := tool["type"].(string); kind != "function" {
				continue
			}
			function, ok := tool["function"].(map[string]any)
			if !ok {
				return request, fmt.Errorf("antigravity tool %d function must be an object", i)
			}
			name, _ := function["name"].(string)
			if name == "" {
				return request, fmt.Errorf("antigravity tool %d function name is required", i)
			}
			description, _ := function["description"].(string)
			declarations = append(declarations, antigravityFunctionDeclaration{Name: name, Description: description, Parameters: function["parameters"]})
		}
		if len(declarations) > 0 {
			request.Tools = []antigravityTool{{FunctionDeclarations: declarations}}
		}
	}

	config := &antigravityGeneration{}
	if value, exists := payload["max_tokens"]; exists {
		maxTokens, ok := positiveInt(value)
		if !ok {
			return request, fmt.Errorf("antigravity max_tokens must be a positive integer")
		}
		config.MaxOutputTokens = maxTokens
	}
	if value, exists := payload["temperature"]; exists {
		temperature, ok := number(value)
		if !ok {
			return request, fmt.Errorf("antigravity temperature must be numeric")
		}
		config.Temperature = temperature
		config.HasTemperature = true
	}
	if config.MaxOutputTokens > 0 || config.HasTemperature {
		request.GenerationConfig = config
	}
	return request, nil
}

func openAIMessageText(value any) (string, error) {
	switch content := value.(type) {
	case nil:
		return "", nil
	case string:
		return content, nil
	case []any:
		var text strings.Builder
		for _, raw := range content {
			part, ok := raw.(map[string]any)
			if !ok {
				return "", fmt.Errorf("content part must be an object")
			}
			kind, _ := part["type"].(string)
			if kind != "text" && kind != "input_text" {
				return "", fmt.Errorf("content part type %q is unsupported", kind)
			}
			value, _ := part["text"].(string)
			text.WriteString(value)
		}
		return text.String(), nil
	default:
		return "", fmt.Errorf("must be a string or text-part array")
	}
}

func mapAssistantToolCalls(value any, names map[string]string) ([]antigravityPart, error) {
	if value == nil {
		return nil, nil
	}
	calls, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("must be an array")
	}
	parts := make([]antigravityPart, 0, len(calls))
	for i, raw := range calls {
		call, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("tool call %d must be an object", i)
		}
		function, ok := call["function"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("tool call %d function must be an object", i)
		}
		name, _ := function["name"].(string)
		if name == "" {
			return nil, fmt.Errorf("tool call %d function name is required", i)
		}
		arguments := map[string]any{}
		switch value := function["arguments"].(type) {
		case nil:
		case string:
			if value != "" && json.Unmarshal([]byte(value), &arguments) != nil {
				return nil, fmt.Errorf("tool call %d arguments must be a JSON object", i)
			}
		case map[string]any:
			arguments = value
		default:
			return nil, fmt.Errorf("tool call %d arguments must be a JSON object", i)
		}
		if id, _ := call["id"].(string); id != "" {
			names[id] = name
		}
		signature, _ := function["thought_signature"].(string)
		if signature == "" {
			signature, _ = call["thought_signature"].(string)
		}
		parts = append(parts, antigravityPart{ThoughtSignature: signature, FunctionCall: &antigravityFunctionCall{Name: name, Args: arguments}})
	}
	return parts, nil
}

func positiveInt(value any) (int, bool) {
	n, ok := number(value)
	if !ok || n <= 0 || n != float64(int(n)) {
		return 0, false
	}
	return int(n), true
}

func number(value any) (float64, bool) {
	switch n := value.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		value, err := n.Float64()
		return value, err == nil
	default:
		return 0, false
	}
}

type antigravitySSEEnvelope struct {
	Response struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text                  string                   `json:"text"`
					Thought               bool                     `json:"thought"`
					ThoughtSignature      string                   `json:"thoughtSignature"`
					ThoughtSignatureSnake string                   `json:"thought_signature"`
					FunctionCall          *antigravityFunctionCall `json:"functionCall"`
				} `json:"parts"`
			} `json:"content"`
			FinishReason string `json:"finishReason"`
		} `json:"candidates"`
		Usage struct {
			PromptTokens     int `json:"promptTokenCount"`
			CompletionTokens int `json:"candidatesTokenCount"`
			TotalTokens      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	} `json:"response"`
}

func parseAntigravitySSE(reader io.Reader, model, requestID string) (map[string]any, error) {
	var content strings.Builder
	var reasoning strings.Builder
	toolCalls := []any{}
	finishReason := "stop"
	usage := map[string]any{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

	err := readSSEData(reader, func(data []byte) error {
		if bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
			return nil
		}
		var event antigravitySSEEnvelope
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("decode antigravity SSE event: %w", err)
		}
		for _, candidate := range event.Response.Candidates {
			for _, part := range candidate.Content.Parts {
				if part.Thought {
					reasoning.WriteString(part.Text)
				} else {
					content.WriteString(part.Text)
				}
				if part.FunctionCall != nil {
					arguments, err := json.Marshal(part.FunctionCall.Args)
					if err != nil {
						return fmt.Errorf("encode antigravity tool arguments: %w", err)
					}
					signature := part.ThoughtSignature
					if signature == "" {
						signature = part.ThoughtSignatureSnake
					}
					function := map[string]any{"name": part.FunctionCall.Name, "arguments": string(arguments)}
					if signature != "" {
						function["thought_signature"] = signature
					}
					toolCalls = append(toolCalls, map[string]any{
						"id":       newAntigravityID("call"),
						"type":     "function",
						"function": function,
					})
				}
			}
			if candidate.FinishReason == "MAX_TOKENS" {
				finishReason = "length"
			}
		}
		if event.Response.Usage.TotalTokens > 0 {
			usage = map[string]any{
				"prompt_tokens":     event.Response.Usage.PromptTokens,
				"completion_tokens": event.Response.Usage.CompletionTokens,
				"total_tokens":      event.Response.Usage.TotalTokens,
			}
		}
		return nil
	})
	if err != nil {
		return nil, core.NewProviderOperationError("antigravity completion response decode", http.StatusOK, "", err)
	}
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}
	message := map[string]any{"role": "assistant", "content": content.String()}
	if reasoning.Len() > 0 {
		message["reasoning_content"] = reasoning.String()
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}
	return map[string]any{
		"id":      requestID,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       message,
			"finish_reason": finishReason,
		}},
		"usage": usage,
	}, nil
}

func readSSEData(reader io.Reader, handle func([]byte) error) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64<<10), antigravityMaxBodyBytes)
	var data bytes.Buffer
	flush := func() error {
		if data.Len() == 0 {
			return nil
		}
		value := bytes.TrimSuffix(data.Bytes(), []byte("\n"))
		data.Reset()
		return handle(value)
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			value := strings.TrimPrefix(line, "data:")
			value = strings.TrimPrefix(value, " ")
			data.WriteString(value)
			data.WriteByte('\n')
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return flush()
}

func newAntigravityID(prefix string) string {
	var value [12]byte
	if _, err := rand.Read(value[:]); err != nil {
		return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
	}
	return prefix + "_" + hex.EncodeToString(value[:])
}

var _ Provider = (*ExperimentalAntigravityProvider)(nil)
