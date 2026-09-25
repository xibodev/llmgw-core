package providers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// ListModels lists the service's voices as models, as the gateway does.
// The request is signed as a synthesis is, and one refused with 403 learns
// the service's clock from its Date and is repeated once. Each voice is a
// Microsoft model serving audio speech, labeled with its friendly name or
// else its locale and gender; a voice without a short name is skipped.
// Unlike the gateway, which lists nothing when the list cannot be read, a
// failure is a catalog failure a product can tell from an empty list.
func (p *EdgeTTS) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	token, err := edgeTTSToken(credential)
	if err != nil {
		return nil, err
	}
	response, err := p.fetchVoices(ctx, token)
	if err == nil && response.StatusCode == http.StatusForbidden {
		p.learn(response.Header.Get("Date"))
		_ = response.Body.Close()
		response, err = p.fetchVoices(ctx, token)
	}
	if err != nil {
		return nil, err
	}
	voices, err := p.decodeVoices(ctx, response)
	if err != nil {
		return nil, err
	}
	models := make([]core.ModelInfo, 0, len(voices))
	for _, voice := range voices {
		if strings.TrimSpace(voice.ShortName) == "" {
			continue
		}
		label := voice.FriendlyName
		if label == "" {
			label = fmt.Sprintf("%s (%s, %s)", voice.ShortName, voice.Locale, voice.Gender)
		}
		models = append(models, core.ModelInfo{
			ID: voice.ShortName, Object: "model", OwnedBy: "microsoft", Vendor: "microsoft", DisplayName: label,
			LegacyCapabilities: map[string]any{"tts": true, "audio": true}, SupportedAPIs: []string{"/v1/audio/speech"},
		})
	}
	return models, nil
}

// edgeTTSVoice is the part of a voice list entry the catalog reads.
type edgeTTSVoice struct {
	ShortName    string `json:"ShortName"`
	FriendlyName string `json:"FriendlyName"`
	Locale       string `json:"Locale"`
	Gender       string `json:"Gender"`
}

// fetchVoices asks for the voice list with the headers of Edge's
// read-aloud feature. A transport failure's cause is redacted, because the
// client quotes the signed URL.
func (p *EdgeTTS) fetchVoices(ctx context.Context, token string) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, p.voicesURL(token), nil)
	if err != nil {
		return nil, core.NewConfigurationError("the Edge TTS voice list request could not be created", &edgeTTSRedactedError{err})
	}
	request.Header.Set("User-Agent", edgeTTSUserAgent)
	request.Header.Set("Accept", "*/*")
	response, err := p.client.Do(request)
	if err != nil {
		return nil, catalogFailure(ctx, CatalogCodeTransportError, "Edge TTS could not reach its voice list.", 0, 0, &edgeTTSRedactedError{err})
	}
	return response, nil
}

// decodeVoices reads the voice list as decodeCatalogResponse reads a
// catalog, and closes it: a refusal is classified by its status, and an
// accepted list, bounded to 8 MiB, must be a JSON array of voice objects.
func (p *EdgeTTS) decodeVoices(ctx context.Context, response *http.Response) ([]edgeTTSVoice, error) {
	defer response.Body.Close()
	status := response.StatusCode
	if status < 200 || status >= 300 {
		code, detail := CatalogCodeHTTPError, fmt.Sprintf("Provider catalog returned HTTP %d.", status)
		if status == http.StatusUnauthorized || status == http.StatusForbidden {
			code, detail = CatalogCodeAuthenticationFailed, "Provider credential was rejected by the catalog API."
		}
		return nil, catalogFailure(ctx, code, detail, status, retryAfterDelay(response.Header.Get("Retry-After"), p.now()), nil)
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, catalogMaxResponseBytes+1))
	switch {
	case len(raw) > catalogMaxResponseBytes:
		return nil, catalogFailure(ctx, CatalogCodeNotDiscoverable, "Provider catalog response exceeded the size limit.", status, 0, nil)
	case err != nil:
		return nil, catalogFailure(ctx, CatalogCodeTransportError, "Provider catalog response could not be read.", status, 0, err)
	}
	var voices []edgeTTSVoice
	err = json.Unmarshal(raw, &voices)
	var syntax *json.SyntaxError
	switch {
	case errors.As(err, &syntax):
		return nil, catalogFailure(ctx, CatalogCodeInvalidJSON, "Provider catalog response was not valid JSON.", status, 0, err)
	case err != nil || voices == nil:
		return nil, catalogFailure(ctx, CatalogCodeInvalidShape, "Provider catalog response was not a list of voices.", status, 0, err)
	}
	return voices, nil
}
