package providers

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// A 401 must reach the Runtime as status 401 from every call an operation
// makes, discovery included, so it refreshes the credential and replays.
// The legacy error stays the cause, with the status and Retry-After the
// gateway reads from it.
func TestAntigravityClassifiesUpstreamStatuses(t *testing.T) {
	t.Parallel()
	for name, tc := range map[string]struct {
		status     int
		retryAfter string
		class      core.ProviderErrorClass
		want       core.ProviderErrorClassification
	}{
		"rejected token": {status: 401, class: core.ProviderErrorAuth, want: core.ProviderErrorClassification{StatusCode: 401}},
		"forbidden":      {status: 403, class: core.ProviderErrorForbidden, want: core.ProviderErrorClassification{StatusCode: 403}},
		"bad request":    {status: 400, class: core.ProviderErrorUpstream, want: core.ProviderErrorClassification{StatusCode: 400}},
		"rate limited": {status: 429, retryAfter: "7", class: core.ProviderErrorRateLimited,
			want: core.ProviderErrorClassification{StatusCode: 429, Retryable: true, FailoverEligible: true, RetryAfter: 7 * time.Second}},
		"unavailable": {status: 503, class: core.ProviderErrorUpstream,
			want: core.ProviderErrorClassification{StatusCode: 503, Retryable: true, FailoverEligible: true, CircuitFailure: true}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server, _ := recordingAntigravityServer(t, func(w http.ResponseWriter, _ *http.Request) {
				if tc.retryAfter != "" {
					w.Header().Set("Retry-After", tc.retryAfter)
				}
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, `{"error":{"status":"FIXTURE","message":"Bearer fixture-secret"}}`)
			})
			provider := newFixtureAntigravity(t, server, nil)
			chat := antigravityChatRequest(`{"messages":[{"role":"user","content":"hi"}]}`, fixtureAntigravityCredential("fixture-project"))
			failures := map[string]func() error{
				"completion": func() error { _, err := provider.Invoke(context.Background(), chat); return err },
				"catalog":    func() error { _, err := provider.ListModels(context.Background(), chat.Credential); return err },
				"discovery": func() error {
					_, err := provider.ListModels(context.Background(), fixtureAntigravityCredential(""))
					return err
				},
				"images": func() error {
					_, err := provider.GenerateImages(context.Background(), core.GenerateImagesRequest{Model: "gemini-fixture-image", Prompt: "draw"}, chat.Credential)
					return err
				},
			}
			for operation, fail := range failures {
				err := fail()
				assertCodexFailure(t, err, tc.class, tc.want)
				var legacy *core.ProviderOperationError
				if !errors.As(err, &legacy) || legacy.Failure.StatusCode != tc.status || legacy.Failure.RetryAfter != tc.retryAfter {
					t.Fatalf("%s cause = %#v", operation, legacy)
				}
				if strings.Contains(err.Error(), "fixture-secret") || strings.Contains(err.Error(), "FIXTURE") {
					t.Fatalf("%s error exposed the upstream body: %v", operation, err)
				}
			}
		})
	}
}

func TestAntigravityClassifiesMalformedResponsesAndTransportFailures(t *testing.T) {
	t.Parallel()
	malformed := core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}
	for name, body := range map[string]string{
		"malformed event":   "data: {not-json}\n\n",
		"malformed catalog": `{"models":[]}`,
	} {
		server, _ := recordingAntigravityServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, body) })
		provider := newFixtureAntigravity(t, server, nil)
		var err error
		if name == "malformed catalog" {
			_, err = provider.ListModels(context.Background(), fixtureAntigravityCredential("fixture-project"))
		} else {
			_, err = provider.Invoke(context.Background(), antigravityChatRequest(`{"messages":[]}`, fixtureAntigravityCredential("fixture-project")))
		}
		assertCodexFailure(t, err, core.ProviderErrorUpstream, malformed)
	}

	provider, err := NewAntigravity(AntigravityConfig{BaseURL: "http://antigravity.invalid", Client: &http.Client{Transport: codexRoundTrip(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("fixture connection reset")
	})}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = provider.ListModels(context.Background(), fixtureAntigravityCredential("fixture-project"))
	assertCodexFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true})

	// Go's client timeout wraps context.DeadlineExceeded too, but while the
	// caller still waits it is a slow upstream.
	slow, release := blockingCodexServer(nil)
	defer slow.Close()
	defer close(release)
	timed, err := NewAntigravity(AntigravityConfig{BaseURL: slow.URL, Client: &http.Client{Timeout: 50 * time.Millisecond}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = timed.ListModels(context.Background(), fixtureAntigravityCredential("fixture-project"))
	assertCodexFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	server, _ := recordingAntigravityServer(t, antigravityUpstream(t))
	_, err = newFixtureAntigravity(t, server, nil).ListModels(ctx, fixtureAntigravityCredential("fixture-project"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want the caller's cancellation", err)
	}
	assertCodexFailure(t, err, core.ProviderErrorTransport, core.ProviderErrorClassification{})
}

func TestNewAntigravityRefusesARelativeBaseURL(t *testing.T) {
	t.Parallel()
	for _, base := range []string{"cloudcode.example.test", "/v1internal", "ftp://cloudcode.example.test"} {
		if _, err := NewAntigravity(AntigravityConfig{BaseURL: base}); err == nil {
			t.Fatalf("base URL %q accepted", base)
		}
	}
	if _, err := NewAntigravity(AntigravityConfig{}); err != nil {
		t.Fatalf("default base URL: %v", err)
	}
}

// Antigravity reads the upstream stream to its end, as the gateway does, so
// it streams nothing and says so before anything is sent.
func TestAntigravityDoesNotStream(t *testing.T) {
	t.Parallel()
	server, calls := recordingAntigravityServer(t, antigravityUpstream(t))
	provider := newFixtureAntigravity(t, server, nil)
	_, err := provider.Stream(context.Background(), antigravityChatRequest(`{"stream":true,"messages":[]}`, fixtureAntigravityCredential("fixture-project")))
	assertCodexFailure(t, err, core.ProviderErrorUnsupported, core.ProviderErrorClassification{FailoverEligible: true})
	if !errors.Is(err, ErrExperimentalAntigravityStreamingUnsupported) || len(calls()) != 0 {
		t.Fatalf("err = %v", err)
	}
	var surface *core.SurfaceError
	if _, err := provider.Stream(context.Background(), core.Request{Surface: core.ModelSurfaceMessages, Model: "m"}); !errors.As(err, &surface) {
		t.Fatalf("err = %v, want a SurfaceError", err)
	}
	if surfaces := provider.NativeSurfaces("gemini-fixture"); len(surfaces) != 1 || surfaces[0] != core.ModelSurfaceChatCompletions {
		t.Fatalf("native surfaces = %v", surfaces)
	}
}

func TestAntigravityListsTheGatewaysRows(t *testing.T) {
	t.Parallel()
	server, calls := recordingAntigravityServer(t, antigravityUpstream(t))
	models, err := newFixtureAntigravity(t, server, nil).ListModels(context.Background(), fixtureAntigravityCredential("fixture-project"))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(models)
	assertSameJSON(t, got, readAntigravityFixture(t, "antigravity", "catalog-current.golden.json"))
	if sent := calls(); len(sent) != 1 || sent[0].path != "/v1internal:fetchAvailableModels" || sent[0].body != `{"project":"fixture-project"}` ||
		sent[0].header.Get("Authorization") != "Bearer fixture-access" {
		t.Fatalf("catalog request = %+v", sent)
	}
}

// Image generation keeps the legacy adapter's capability: one image, from a
// model both rosters of the same catalog name.
func TestAntigravityGeneratesImagesAsTheLegacyAdapterDoes(t *testing.T) {
	t.Parallel()
	encoded := base64.StdEncoding.EncodeToString([]byte("fixture-png"))
	server, calls := recordingAntigravityServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1internal:streamGenerateContent" {
			_, _ = fmt.Fprintf(w, "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":%q}}]}}],\"usageMetadata\":{\"totalTokenCount\":7}}}\n\n", encoded)
			return
		}
		antigravityUpstream(t)(w, r)
	})
	legacy := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) { return "fixture-access", "fixture-project", nil }, server.Client(), server.URL)
	request := core.GenerateImagesRequest{Model: " gemini-fixture-image ", Prompt: "draw a fixture"}
	if _, err := legacy.GenerateImages(context.Background(), request, nil); err != nil {
		t.Fatal(err)
	}
	provider := newFixtureAntigravity(t, server, nil)
	result, err := provider.GenerateImages(context.Background(), request, fixtureAntigravityCredential("fixture-project"))
	if err != nil || len(result.Images) != 1 || string(result.Images[0].Data) != "fixture-png" || result.Images[0].MimeType != "image/png" || result.Usage["totalTokenCount"] != float64(7) {
		t.Fatalf("result = %+v, err = %v", result, err)
	}
	sent := calls()
	if len(sent) != 4 || !reflect.DeepEqual(withoutRequestID(sent[3]), withoutRequestID(sent[1])) || !strings.Contains(sent[3].body, `"responseModalities":["TEXT","IMAGE"]`) {
		t.Fatalf("upstream = %+v", sent)
	}

	for name, tc := range map[string]struct {
		request core.GenerateImagesRequest
		class   core.ProviderErrorClass
	}{
		"two images":      {core.GenerateImagesRequest{Model: "gemini-fixture-image", Prompt: "draw", Count: 2}, core.ProviderErrorInvalidRequest},
		"no prompt":       {core.GenerateImagesRequest{Model: "gemini-fixture-image"}, core.ProviderErrorInvalidRequest},
		"a chat model":    {core.GenerateImagesRequest{Model: "gemini-fixture-pro", Prompt: "draw"}, core.ProviderErrorUnsupported},
		"a retired model": {core.GenerateImagesRequest{Model: "gemini-fixture-retired-image", Prompt: "draw"}, core.ProviderErrorUnsupported},
	} {
		_, err := provider.GenerateImages(context.Background(), tc.request, fixtureAntigravityCredential("fixture-project"))
		want := core.ProviderErrorClassification{FailoverEligible: tc.class == core.ProviderErrorUnsupported}
		assertCodexFailure(t, err, tc.class, want)
		var legacy *core.ProviderOperationError
		if !errors.As(err, &legacy) || legacy.Failure.StatusCode != http.StatusBadRequest {
			t.Fatalf("%s cause = %#v, want the legacy 400", name, legacy)
		}
	}
	if sent := calls(); len(sent) != 2 || !strings.Contains(sent[0].path, "fetchAvailableModels") {
		t.Fatalf("refused images reached the upstream: %+v", sent)
	}
}
