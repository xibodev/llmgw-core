package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"slices"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// openAIAccess is how one operation authenticates.
type openAIAccess struct {
	// authorization is the Authorization header, empty without a key.
	authorization string
	headers       map[string]string
}

// access reads a credential as the gateway reads an OpenAI-compatible key:
// the API key, or else the token, normalized by openAIBearerKey. A
// credential of a kind whose material is no bearer token is refused, so
// such material is never sent.
func (p *OpenAICompatible) access(credential *core.Credential) (openAIAccess, error) {
	var access openAIAccess
	key := ""
	if credential != nil {
		switch credential.TokenType {
		case core.TokenTypeGCPServiceAccount, core.TokenTypeAnthropicSetupToken:
			return openAIAccess{}, core.NewConfigurationError("an OpenAI-compatible provider cannot use a "+credential.TokenType+" credential", nil)
		}
		key, access.headers = credential.APIKey, credential.Headers
		if strings.TrimSpace(key) == "" {
			key = credential.Token
		}
	}
	key = openAIBearerKey(key)
	if key != "" {
		access.authorization = "Bearer " + key
	}
	return access, nil
}

// openAIBearerKey normalizes a key as the gateway does: trimmed, without a
// "Bearer " prefix, and empty for "free" and "none".
func openAIBearerKey(value string) string {
	key := strings.TrimSpace(value)
	if strings.HasPrefix(strings.ToLower(key), "bearer ") {
		key = strings.TrimSpace(key[len("bearer "):])
	}
	if strings.EqualFold(key, "free") || strings.EqualFold(key, "none") {
		return ""
	}
	return key
}

// header returns the headers of one request: the configured headers, then
// the credential's, then the transport's own, which neither replaces. The
// gateway marks a request with an image for every OpenAI-compatible
// upstream: GitHub Copilot rejects images without the mark, and others
// ignore it.
func (p *OpenAICompatible) header(access openAIAccess, accept string, vision bool) http.Header {
	header := http.Header{}
	for name, value := range p.headers {
		header.Set(name, value)
	}
	for name, value := range access.headers {
		header.Set(name, value)
	}
	header.Set("Content-Type", core.ContentTypeJSON)
	if access.authorization != "" {
		header.Set("Authorization", access.authorization)
	}
	if accept != "" {
		header.Set("Accept", accept)
	}
	if vision {
		header.Set("Copilot-Vision-Request", "true")
	}
	return header
}

// openAIPayload decodes the JSON object body. Numbers decode as the
// gateway's facades decode them, so a value reaches the upstream as the
// gateway renders it.
func openAIPayload(request core.Request) (map[string]any, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return nil, openAIInvalid("an OpenAI-compatible request body must be JSON", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(request.Body))
	var payload map[string]any
	err = decoder.Decode(&payload)
	if err == nil {
		if _, trailing := decoder.Token(); trailing != io.EOF {
			err = errors.New("the body continues after its JSON object")
		}
	}
	if err != nil || payload == nil {
		return nil, openAIInvalid("the OpenAI-compatible request body is not a JSON object", err)
	}
	return payload, nil
}

// openAIEncode renders a body as the gateway does: keys sorted, HTML left
// unescaped and no trailing newline.
func openAIEncode(payload map[string]any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(payload); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buffer.Bytes(), "\n"), nil
}

// post sends payload to path and returns the response, open whatever its
// status.
func (p *OpenAICompatible) post(ctx context.Context, path string, header http.Header, payload map[string]any) (*http.Response, error) {
	body, err := openAIEncode(payload)
	if err != nil {
		return nil, openAIInvalid("the "+p.label+" request could not be encoded", err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, core.NewConfigurationError("the "+p.label+" request could not be created", err)
	}
	request.Header = header.Clone()
	response, err := p.client.Do(request)
	if err != nil {
		return nil, transportFailure(ctx, p.label+" could not be reached", err)
	}
	return response, nil
}

// read reads and closes an answer within the gateway's bound.
func (p *OpenAICompatible) read(ctx context.Context, response *http.Response) ([]byte, error) {
	defer response.Body.Close()
	return readInvocationResponseBody(ctx, response, p.label)
}

// refused reports an answer the upstream refused, classified as the
// gateway classifies its status, with the upstream's Retry-After.
func (p *OpenAICompatible) refused(response *http.Response, raw []byte) error {
	return httpStatusFailure(p.label, response, raw, p.now())
}

// checkChat refuses a Chat answer the gateway would not use: one that is
// not a JSON object with a first choice, or that carries an error beside a
// success status, which the gateway retries.
func (p *OpenAICompatible) checkChat(raw []byte) error {
	var answer map[string]any
	if json.Unmarshal(raw, &answer) != nil || len(answer) == 0 {
		return unusableResponse("the "+p.label+" completion is not JSON", nil)
	}
	if openAISoftError(answer) {
		return &core.ProviderError{
			Message: p.label + " answered with an error", Class: core.ProviderErrorUpstream,
			Classification: core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true},
		}
	}
	choices, _ := answer["choices"].([]any)
	if len(choices) == 0 {
		return unusableResponse("the "+p.label+" completion has no choices", nil)
	}
	if _, ok := choices[0].(map[string]any); !ok {
		return unusableResponse("the "+p.label+" completion has an invalid choice", nil)
	}
	return nil
}

// openAISoftError reports an error field the gateway reads as an error: a
// nonblank string, an object with a nonblank message, or any other value
// that is not an object with a string message.
func openAISoftError(answer map[string]any) bool {
	switch value := answer["error"].(type) {
	case nil:
		return false
	case string:
		return strings.TrimSpace(value) != ""
	case map[string]any:
		if message, ok := value["message"].(string); ok {
			return strings.TrimSpace(message) != ""
		}
	}
	return true
}

// responsesAnswer decodes a Responses answer the gateway would use: a JSON
// object with an output array.
func (p *OpenAICompatible) responsesAnswer(raw []byte) (map[string]any, error) {
	var answer map[string]any
	if json.Unmarshal(raw, &answer) != nil || len(answer) == 0 {
		return nil, unusableResponse("the "+p.label+" response is not JSON", nil)
	}
	if _, ok := answer["output"].([]any); !ok {
		return nil, unusableResponse("the "+p.label+" response has no output", nil)
	}
	return answer, nil
}

// openAIStream reports the request's losses through core.LossReporter.
type openAIStream struct {
	core.StreamIter
	losses []core.Loss
}

var _ core.LossReporter = (*openAIStream)(nil)

// Losses implements core.LossReporter. They are known before the first
// frame.
func (s *openAIStream) Losses() []core.Loss { return slices.Clone(s.losses) }

// openAIFrames replays frames already rendered.
type openAIFrames struct{ frames [][]byte }

func (s *openAIFrames) Next() ([]byte, error) {
	if len(s.frames) == 0 {
		return nil, io.EOF
	}
	frame := s.frames[0]
	s.frames = s.frames[1:]
	return frame, nil
}

func (s *openAIFrames) Close() error {
	s.frames = nil
	return nil
}

// openAIChatImages reports a Chat message with an image part.
func openAIChatImages(messages []any) bool {
	for _, raw := range messages {
		message, _ := raw.(map[string]any)
		parts, _ := message["content"].([]any)
		for _, rawPart := range parts {
			part, _ := rawPart.(map[string]any)
			switch part["type"] {
			case "image_url", "input_image", "image":
				return true
			}
		}
	}
	return false
}

// openAIResponsesImages reports an image anywhere in a Responses input.
func openAIResponsesImages(value any) bool {
	switch current := value.(type) {
	case []any:
		return slices.ContainsFunc(current, openAIResponsesImages)
	case map[string]any:
		switch current["type"] {
		case "input_image", "image_url":
			return true
		}
		for _, nested := range current {
			if openAIResponsesImages(nested) {
				return true
			}
		}
	}
	return false
}

// openAIInvalid reports a request no target could serve.
func openAIInvalid(message string, err error) *core.ProviderError {
	return &core.ProviderError{Message: message, Class: core.ProviderErrorInvalidRequest, Cause: err}
}
