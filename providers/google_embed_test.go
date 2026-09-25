package providers

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"sync/atomic"
	"testing"

	translate "github.com/xibodev/llm-translate"
	core "github.com/xibodev/llmgw-core"
)

func googleEmbeddingsRequest(model, body string, credential *core.Credential) core.Request {
	return core.Request{Surface: core.ModelSurfaceEmbeddings, Model: model, Body: []byte(body), ContentType: core.ContentTypeJSON, Credential: credential}
}

// Ported from the gateway's TestGoogleEmbeddingTransportsNormalizeOpenAIEnvelope,
// with the bytes the gateway sends and returns.
func TestGoogleEmbeddingTransportsNormalizeOpenAIEnvelope(t *testing.T) {
	t.Parallel()
	t.Run("AI Studio", func(t *testing.T) {
		t.Parallel()
		fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, `{"embedding":{"values":[0.1,0.2,0.3]}}`))
		provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
		response, err := provider.Invoke(context.Background(), googleEmbeddingsRequest("gemini-embedding-001",
			`{"model":"gemini-embedding-001","input":"hello <world>","encoding_format":" float ","dimensions":null,"user":"u-1"}`, googleKey("key")))
		if err != nil {
			t.Fatal(err)
		}
		want := googleCall{method: http.MethodPost, path: "/models/gemini-embedding-001:embedContent", apiKey: "key",
			body: `{"content":{"parts":[{"text":"hello \u003cworld\u003e"}]}}`}
		if calls := fake.take(); !reflect.DeepEqual(calls, []googleCall{want}) {
			t.Fatalf("upstream = %+v", calls)
		}
		body := `{"data":[{"embedding":[0.1,0.2,0.3],"index":0,"object":"embedding"}],"model":"gemini-embedding-001","object":"list","usage":{"prompt_tokens":0,"total_tokens":0}}`
		loss := core.Loss{Path: "user", Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Google does not send this embeddings field"}
		if string(response.Body) != body || !reflect.DeepEqual(response.Losses, []core.Loss{loss}) {
			t.Fatalf("response = %s, losses = %+v", response.Body, response.Losses)
		}
	})

	t.Run("Vertex preserves input order", func(t *testing.T) {
		t.Parallel()
		var calls atomic.Int32
		fake, base := newGoogleFake(t, func(*http.Request) (int, string) {
			call := calls.Add(1)
			return http.StatusOK, fmt.Sprintf(`{"predictions":[{"embeddings":{"values":[%d,0.5],"statistics":{"token_count":%d}}}]}`, call, call+1)
		})
		provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "project", RequestType: "paygo"})
		response, err := provider.Invoke(context.Background(), googleEmbeddingsRequest("models/text-embedding-005", `{"input":["first","second"]}`, googleKey("key")))
		if err != nil {
			t.Fatal(err)
		}
		path := "/v1/projects/project/locations/global/publishers/google/models/text-embedding-005:predict"
		want := []googleCall{
			{method: http.MethodPost, path: path, apiKey: "key", requestType: "shared", body: `{"instances":[{"content":"first"}]}`},
			{method: http.MethodPost, path: path, apiKey: "key", requestType: "shared", body: `{"instances":[{"content":"second"}]}`},
		}
		if got := fake.take(); !reflect.DeepEqual(got, want) {
			t.Fatalf("upstream = %+v", got)
		}
		body := `{"data":[{"embedding":[1,0.5],"index":0,"object":"embedding"},{"embedding":[2,0.5],"index":1,"object":"embedding"}],` +
			`"model":"text-embedding-005","object":"list","usage":{"prompt_tokens":5,"total_tokens":5}}`
		if string(response.Body) != body {
			t.Fatalf("response = %s", response.Body)
		}
	})

	t.Run("Vertex Gemini embedding 2 uses embedContent", func(t *testing.T) {
		t.Parallel()
		fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, `{"embedding":{"values":[0.1,0.2]},"usageMetadata":{"promptTokenCount":3}}`))
		provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "project"})
		response, err := provider.Invoke(context.Background(), googleEmbeddingsRequest("gemini-embedding-2", `{"input":"hello"}`, googleKey("key")))
		if err != nil {
			t.Fatal(err)
		}
		calls := fake.take()
		if len(calls) != 1 || calls[0].path != "/v1/projects/project/locations/global/publishers/google/models/gemini-embedding-2:embedContent" ||
			string(response.Body) != `{"data":[{"embedding":[0.1,0.2],"index":0,"object":"embedding"}],"model":"gemini-embedding-2","object":"list","usage":{"prompt_tokens":3,"total_tokens":3}}` {
			t.Fatalf("upstream = %+v, response = %s", calls, response.Body)
		}
	})
}

// The gateway's native embeddings route refuses what Google cannot embed
// before anything is sent. A well-formed request another target may serve
// permits failover.
func TestGoogleEmbeddingsRefuseWhatTheGatewayRefuses(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, `{"embedding":{"values":[0.1]}}`))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	for body, want := range map[string]struct {
		class   core.ProviderErrorClass
		message string
	}{
		`{}`:                             {core.ProviderErrorInvalidRequest, "'input' is required (a string or an array of strings)"},
		`{"input":"  "}`:                 {core.ProviderErrorInvalidRequest, "'input' is required (a string or an array of strings)"},
		`{"input":[]}`:                   {core.ProviderErrorInvalidRequest, "'input' is required (a string or an array of strings)"},
		`{"input":[1,2]}`:                {core.ProviderErrorUnsupported, "native embedding input must be a string or array of strings"},
		`{"input":["a"," "]}`:            {core.ProviderErrorUnsupported, "native embedding input must be a string or array of strings"},
		`{"input":{"text":"a"}}`:         {core.ProviderErrorUnsupported, "native embedding input must be a string or array of strings"},
		`{"input":"a","dimensions":256}`: {core.ProviderErrorUnsupported, "dimensions is not supported by this native embeddings provider"},
		`{"input":"a","encoding_format":"base64"}`: {core.ProviderErrorUnsupported, "only encoding_format 'float' is supported by this native embeddings provider"},
		`{"input":"a","encoding_format":5}`:        {core.ProviderErrorInvalidRequest, "encoding_format must be a string"},
	} {
		_, err := provider.Invoke(context.Background(), googleEmbeddingsRequest("gemini-embedding-001", body, googleKey("k")))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != want.class || err.Error() != want.message ||
			core.ClassifyError(err).FailoverEligible != (want.class == core.ProviderErrorUnsupported) {
			t.Errorf("body %s: %v", body, err)
		}
	}
	if calls := fake.take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
	// An answer without a vector is unusable, and another target may serve.
	_, empty := newGoogleFake(t, googleAnswer(http.StatusOK, `{"embedding":{}}`))
	provider = newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: empty})
	_, err := provider.Invoke(context.Background(), googleEmbeddingsRequest("gemini-embedding-001", `{"input":"a"}`, googleKey("k")))
	if err == nil || err.Error() != "ai_studio: embedding response contained no vector" ||
		core.ClassifyError(err) != (core.ProviderErrorClassification{FailoverEligible: true}) {
		t.Fatalf("error = %v", err)
	}
	if _, err := provider.Stream(context.Background(), googleEmbeddingsRequest("gemini-embedding-001", `{"input":"a"}`, googleKey("k"))); core.ClassifyError(err).Disposition() != core.DispositionFailover {
		t.Fatalf("stream = %v, want a refusal that permits failover", err)
	}
}
