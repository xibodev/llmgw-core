package providers

import (
	"context"
	"encoding/json"
	"maps"
	"slices"
	"strings"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

// embed performs one embeddings request as the gateway does: one upstream
// call per input, in order, to embedContent, or on Vertex AI to predict,
// which serves every embedding model but Gemini Embedding 2. It returns
// the OpenAI embeddings envelope the gateway returns.
func (p *Google) embed(ctx context.Context, request core.Request) (core.Response, error) {
	body, err := googleJSONBody(request)
	if err != nil {
		return core.Response{}, err
	}
	values, losses, err := googleEmbeddingInputs(body)
	if err != nil {
		return core.Response{}, err
	}
	access, err := p.access(request.Credential)
	if err != nil {
		return core.Response{}, err
	}
	predict := p.deployment == GoogleVertexAI && !strings.HasPrefix(strings.ToLower(strings.TrimPrefix(request.Model, "models/")), "gemini-embedding-2")
	action := "embedContent"
	if predict {
		action = "predict"
	}
	endpoint, err := p.modelURL(request.Model, access.project, action)
	if err != nil {
		return core.Response{}, err
	}
	authorization, err := p.authorization(access)
	if err != nil {
		return core.Response{}, err
	}
	data := make([]any, 0, len(values))
	totalTokens := 0
	for index, text := range values {
		payload := map[string]any{"content": map[string]any{"parts": []any{map[string]any{"text": text}}}}
		if predict {
			payload = map[string]any{"instances": []any{map[string]any{"content": text}}}
		}
		decoded, err := p.post(ctx, authorization, endpoint, payload)
		if err != nil {
			return core.Response{}, err
		}
		vector, tokens := googleEmbedding(decoded)
		if len(vector) == 0 {
			message := p.label() + ": embedding response contained no vector"
			return core.Response{}, &core.ProviderError{
				Message: message, Class: core.ProviderErrorUpstream, Classification: core.ProviderErrorClassification{FailoverEligible: true},
				Cause: &InvocationError{Msg: message, FailoverEligible: true},
			}
		}
		totalTokens += tokens
		data = append(data, map[string]any{"object": "embedding", "index": index, "embedding": vector})
	}
	encoded, err := json.Marshal(map[string]any{
		"object": "list", "data": data, "model": strings.TrimPrefix(strings.TrimSpace(request.Model), "models/"),
		"usage": map[string]any{"prompt_tokens": totalTokens, "total_tokens": totalTokens},
	})
	if err != nil {
		return core.Response{}, unusableResponse("the "+p.label()+" answer could not be encoded", err)
	}
	return core.Response{Body: encoded, ContentType: core.ContentTypeJSON, Losses: losses}, nil
}

// googleEmbeddingInputs reads an embeddings body as the gateway's native
// embeddings route does: the input is a nonblank string or a nonempty array
// of nonblank strings, dimensions cannot be served and float is the only
// encoding. Another target may serve what Google refuses, so a refusal of
// a well-formed request permits failover. Every other field is dropped and
// reported.
func googleEmbeddingInputs(body map[string]any) ([]string, []core.Loss, error) {
	unsupported := func(message string) error {
		return &core.ProviderError{Message: message, Class: core.ProviderErrorUnsupported, Classification: core.ProviderErrorClassification{FailoverEligible: true}}
	}
	values, ok := googleEmbeddingStrings(body["input"])
	switch {
	case googleEmbeddingInputEmpty(body["input"]):
		return nil, nil, &core.ProviderError{Message: "'input' is required (a string or an array of strings)", Class: core.ProviderErrorInvalidRequest}
	case !ok:
		return nil, nil, unsupported("native embedding input must be a string or array of strings")
	case body["dimensions"] != nil:
		return nil, nil, unsupported("dimensions is not supported by this native embeddings provider")
	}
	if value := body["encoding_format"]; value != nil {
		format, text := value.(string)
		if !text {
			return nil, nil, &core.ProviderError{Message: "encoding_format must be a string", Class: core.ProviderErrorInvalidRequest}
		}
		if format = strings.TrimSpace(format); format != "" && format != "float" {
			return nil, nil, unsupported("only encoding_format 'float' is supported by this native embeddings provider")
		}
	}
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(body)) {
		switch field {
		case "model", "input", "dimensions", "encoding_format":
			continue
		}
		if body[field] != nil {
			losses = append(losses, core.Loss{Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Google does not send this embeddings field"})
		}
	}
	return values, losses, nil
}

// googleEmbeddingInputEmpty reports an input that holds nothing to embed.
func googleEmbeddingInputEmpty(input any) bool {
	switch typed := input.(type) {
	case nil:
		return true
	case string:
		return strings.TrimSpace(typed) == ""
	case []any:
		return len(typed) == 0
	}
	return false
}

// googleEmbeddingStrings returns the texts of an input Google can embed,
// as the caller wrote them.
func googleEmbeddingStrings(input any) ([]string, bool) {
	switch typed := input.(type) {
	case string:
		return []string{typed}, strings.TrimSpace(typed) != ""
	case []any:
		values := make([]string, 0, len(typed))
		for _, raw := range typed {
			text, ok := raw.(string)
			if !ok || strings.TrimSpace(text) == "" {
				return nil, false
			}
			values = append(values, text)
		}
		return values, len(values) > 0
	}
	return nil, false
}

// googleEmbedding reads a vector and its token count from either answer:
// embedContent's embedding, or predict's first prediction.
func googleEmbedding(decoded map[string]any) ([]any, int) {
	if embedding, ok := decoded["embedding"].(map[string]any); ok {
		values, _ := embedding["values"].([]any)
		usage, _ := decoded["usageMetadata"].(map[string]any)
		tokens := 0
		if count, ok := usage["promptTokenCount"].(float64); ok {
			tokens = int(count)
		}
		return values, tokens
	}
	predictions, _ := decoded["predictions"].([]any)
	if len(predictions) == 0 {
		return nil, 0
	}
	prediction, _ := predictions[0].(map[string]any)
	embeddings, _ := prediction["embeddings"].(map[string]any)
	values, _ := embeddings["values"].([]any)
	statistics, _ := embeddings["statistics"].(map[string]any)
	tokens := 0
	if count, ok := statistics["token_count"].(float64); ok {
		tokens = int(count)
	}
	return values, tokens
}
