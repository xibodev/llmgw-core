package providers

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// Ported from the gateway's TestGenerateImagesDecodesInlineData, with the
// body the gateway sends.
func TestGoogleGenerateImagesDecodesInlineData(t *testing.T) {
	t.Parallel()
	png := []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3}
	encoded := base64.StdEncoding.EncodeToString(png)
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, `{"candidates":[{"content":{"parts":[
          {"text":"here you go"},
          {"inlineData":{"mimeType":"image/png","data":"`+encoded+`"}},
          {"inlineData":{"mimeType":"image/png","data":"not base64!"}},
          {"inlineData":{"mimeType":"image/webp","data":"`+encoded+`"}}
        ]}}],
        "usageMetadata":{"candidatesTokensDetails":[{"modality":"IMAGE","tokenCount":1120}]}}`))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "proj"})
	result, err := provider.GenerateImages(context.Background(), core.GenerateImagesRequest{Model: "gemini-3.1-flash-image", Prompt: "an origami <crane>", Count: 1}, googleKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 || string(result.Images[0].Data) != string(png) || result.Images[0].MimeType != "image/png" {
		t.Fatalf("images = %+v", result.Images)
	}
	// Image cost arrives as modality-tagged tokens, so accounting stays uniform.
	if result.Usage["candidatesTokensDetails"] == nil {
		t.Fatalf("per-modality usage was dropped: %+v", result.Usage)
	}
	calls := fake.take()
	if len(calls) != 1 || calls[0].path != "/v1/projects/proj/locations/global/publishers/google/models/gemini-3.1-flash-image:generateContent" ||
		calls[0].body != `{"contents":[{"parts":[{"text":"an origami \u003ccrane\u003e"}],"role":"user"}],"generationConfig":{"responseModalities":["TEXT","IMAGE"]}}` {
		t.Fatalf("upstream = %+v", calls)
	}
	// Without a bound every decodable image is returned.
	result, err = provider.GenerateImages(context.Background(), core.GenerateImagesRequest{Model: "gemini-3.1-flash-image", Prompt: "a crane"}, googleKey("k"))
	if err != nil || len(result.Images) != 2 || result.Images[1].MimeType != "image/webp" {
		t.Fatalf("images = %+v, err = %v", result.Images, err)
	}
}

// Ported from the gateway's TestGenerateImagesRefusesTextOnlyModel.
func TestGoogleGenerateImagesRefusesTextOnlyModel(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, `{"candidates":[{"content":{"parts":[{"text":"I cannot draw"}]}}]}`))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	_, err := provider.GenerateImages(context.Background(), core.GenerateImagesRequest{Model: "gemini-3.5-flash", Prompt: "a crane", Count: 1}, googleKey("k"))
	var failure *core.ProviderError
	if !errors.As(err, &failure) || !strings.Contains(err.Error(), "no image data") || failure.Class != core.ProviderErrorUnsupported ||
		core.ClassifyError(err).Disposition() != core.DispositionFailover {
		t.Fatalf("err = %v, want a clear text-only-model refusal", err)
	}
	// A blank prompt is refused before anything is sent.
	if _, err := provider.GenerateImages(context.Background(), core.GenerateImagesRequest{Model: "gemini-3.5-flash", Prompt: " "}, googleKey("k")); !errors.As(err, &failure) ||
		failure.Class != core.ProviderErrorInvalidRequest || err.Error() != "ai_studio: a prompt is required" {
		t.Fatalf("err = %v", err)
	}
	if calls := fake.take(); len(calls) != 1 {
		t.Fatalf("upstream = %+v", calls)
	}
}
