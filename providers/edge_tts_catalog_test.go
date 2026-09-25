package providers

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// edgeTTSVoiceList serves a voice list the way the service does: it
// refuses an unsigned request with 403. answer writes each signed reply.
func edgeTTSVoiceList(t *testing.T, answer func(w http.ResponseWriter, attempt int)) (*EdgeTTS, func() []*http.Request) {
	t.Helper()
	var mu sync.Mutex
	var requests []*http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests = append(requests, r.Clone(r.Context()))
		attempt := len(requests)
		mu.Unlock()
		query := r.URL.Query()
		if r.URL.Path != "/tts/voices/list" || query.Get("Sec-MS-GEC") == "" || query.Get("Sec-MS-GEC-Version") == "" {
			http.Error(w, "missing request signature", http.StatusForbidden)
			return
		}
		answer(w, attempt)
	}))
	t.Cleanup(server.Close)
	provider := newFixtureEdgeTTS(t, &edgeTTSService{}, func(config *EdgeTTSConfig) {
		config.BaseURL, config.Client = server.URL+"/tts", server.Client()
	})
	return provider, func() []*http.Request {
		mu.Lock()
		defer mu.Unlock()
		return requests
	}
}

// Ported from the gateway's TestEdgeTTSSynthesizeAgainstMockService, which
// lists the mock's voices.
func TestEdgeTTSListsTheServicesVoices(t *testing.T) {
	t.Parallel()
	provider, requests := edgeTTSVoiceList(t, func(w http.ResponseWriter, attempt int) {
		if attempt == 1 {
			// The service's clock is an hour ahead of the fixture's.
			w.Header().Set("Date", edgeTTSFixtureNow.Add(time.Hour).Format(time.RFC1123))
			w.WriteHeader(http.StatusForbidden)
			return
		}
		_, _ = io.WriteString(w, `[{"ShortName":"en-US-TestNeural","FriendlyName":"Test voice","Locale":"en-US","Gender":"Female"},`+
			`{"ShortName":"fr-FR-FixtureNeural","Locale":"fr-FR","Gender":"Male","Status":"GA"},{"ShortName":" "},null]`)
	})
	models, err := provider.ListModels(t.Context(), &core.Credential{APIKey: edgeTTSFixtureToken})
	row := func(id, label string) core.ModelInfo {
		return core.ModelInfo{
			ID: id, Object: "model", OwnedBy: "microsoft", Vendor: "microsoft", DisplayName: label,
			LegacyCapabilities: map[string]any{"tts": true, "audio": true}, SupportedAPIs: []string{"/v1/audio/speech"},
		}
	}
	if want := []core.ModelInfo{row("en-US-TestNeural", "Test voice"), row("fr-FR-FixtureNeural", "fr-FR-FixtureNeural (fr-FR, Male)")}; err != nil || !reflect.DeepEqual(models, want) {
		t.Fatalf("models = %+v, err = %v", models, err)
	}
	if capabilities := core.InferCapabilities(models[0], edgeTTSFixtureNow, edgeTTSFixtureNow); capabilities.Operations.AudioOut != core.SupportSupported {
		t.Fatalf("capabilities = %+v", capabilities)
	}
	sent := requests()
	if len(sent) != 2 {
		t.Fatalf("requests = %d, want a retry after the refusal", len(sent))
	}
	for index, want := range []string{edgeTTSFixtureSignature, edgeTTSSkewedSignature} {
		query := sent[index].URL.Query()
		if query.Get("Sec-MS-GEC") != want || query.Get("Ocp-Apim-Subscription-Key") != edgeTTSFixtureToken || query.Get("Sec-MS-GEC-Version") != "1-140.0.3485.14" ||
			sent[index].Header.Get("User-Agent") != edgeTTSUserAgent || sent[index].Header.Get("Accept") != "*/*" {
			t.Fatalf("request %d = %s %v", index, sent[index].URL, sent[index].Header)
		}
	}
}

func TestEdgeTTSVoiceListFailures(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, body, code string
		status           int
	}{
		{name: "unauthorized", body: `{"error":"fixture-secret"}`, code: CatalogCodeAuthenticationFailed, status: 401},
		{name: "forbidden twice", body: `{"error":"fixture-secret"}`, code: CatalogCodeAuthenticationFailed, status: 403},
		{name: "limited", body: `fixture-secret`, code: CatalogCodeHTTPError, status: 429},
		{name: "an object", body: `{"voices":[]}`, code: CatalogCodeInvalidShape, status: 200},
		{name: "null", body: `null`, code: CatalogCodeInvalidShape, status: 200},
		{name: "a number row", body: `[17]`, code: CatalogCodeInvalidShape, status: 200},
		{name: "a number name", body: `[{"ShortName":5}]`, code: CatalogCodeInvalidShape, status: 200},
		{name: "malformed", body: `[{"ShortName":`, code: CatalogCodeInvalidJSON, status: 200},
	} {
		provider, requests := edgeTTSVoiceList(t, func(w http.ResponseWriter, _ int) {
			w.Header().Set("Retry-After", "9")
			w.WriteHeader(tc.status)
			_, _ = io.WriteString(w, tc.body)
		})
		models, err := provider.ListModels(t.Context(), nil)
		failure := catalogCode(t, err)
		if models != nil || failure.Code != tc.code || failure.Status != tc.status || tc.status == 429 && failure.RetryAfter != 9*time.Second {
			t.Fatalf("%s: models = %+v, failure = %+v", tc.name, models, failure)
		}
		want := 1
		if tc.status == http.StatusForbidden {
			want = 2
		}
		if len(requests()) != want {
			t.Fatalf("%s: requests = %d, want %d", tc.name, len(requests()), want)
		}
	}

	closed := httptest.NewServer(http.NotFoundHandler())
	closed.Close()
	provider := newFixtureEdgeTTS(t, &edgeTTSService{}, func(config *EdgeTTSConfig) { config.BaseURL = closed.URL + "/tts" })
	_, err := provider.ListModels(t.Context(), &core.Credential{APIKey: "llmgw_edge_subscription_secret_123456"})
	var urlErr *url.Error
	if catalogCode(t, err).Code != CatalogCodeTransportError || !errors.As(err, &urlErr) || strings.Contains(errors.Unwrap(errors.Unwrap(err)).Error(), "secret_123456") {
		t.Fatalf("transport failure = %#v", err)
	}
	if _, err := provider.ListModels(t.Context(), &core.Credential{APIKey: "unsafe token"}); core.ClassifyError(err).Disposition() != core.DispositionFailover || err == nil {
		t.Fatalf("an unsafe token: err = %v, want a configuration error", err)
	}
}
