package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

type resolvedProject struct {
	credential *core.Credential
	projectID  string
}

// A credential without a project discovers it with the request the legacy
// adapter sends, and reports it once the operation succeeds. A credential
// that names one sends no discovery and reports nothing.
func TestAntigravityDiscoversAMissingProjectAndReportsItAfterSuccess(t *testing.T) {
	t.Parallel()
	server, calls := recordingAntigravityServer(t, antigravityUpstream(t))
	var reported []resolvedProject
	provider := newFixtureAntigravity(t, server, func(_ context.Context, credential *core.Credential, projectID string) {
		reported = append(reported, resolvedProject{credential, projectID})
	})
	legacy := NewExperimentalAntigravityProvider(func(context.Context) (string, string, error) {
		return "fixture-access", "", nil
	}, server.Client(), server.URL)
	if _, err := legacy.ListModels(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	legacyCalls := calls()
	credential := fixtureAntigravityCredential("")
	if _, err := provider.ListModels(context.Background(), credential); err != nil {
		t.Fatal(err)
	}
	if got := calls(); !reflect.DeepEqual(got, legacyCalls) || got[1].body != `{"project":"fixture-project-discovered"}` {
		t.Fatalf("discovery = %+v, want the legacy %+v", got, legacyCalls)
	}
	if len(reported) != 1 || reported[0].credential != credential || reported[0].projectID != "fixture-project-discovered" {
		t.Fatalf("reported = %+v", reported)
	}

	named := fixtureAntigravityCredential("fixture-project")
	if _, err := provider.Invoke(context.Background(), antigravityChatRequest(`{"messages":[{"role":"user","content":"hi"}]}`, named)); err != nil {
		t.Fatal(err)
	}
	if got := calls(); len(got) != 1 || !strings.Contains(got[0].body, `"project":"fixture-project"`) || len(reported) != 1 {
		t.Fatalf("upstream = %+v, reported = %+v", got, reported)
	}
}

func TestAntigravityReportsNoProjectFromAFailedOperation(t *testing.T) {
	t.Parallel()
	server, calls := recordingAntigravityServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1internal:loadCodeAssist" {
			_, _ = io.WriteString(w, `{"cloudaicompanionProject":"fixture-project-discovered"}`)
			return
		}
		w.WriteHeader(http.StatusUnauthorized)
	})
	reported := 0
	provider := newFixtureAntigravity(t, server, func(context.Context, *core.Credential, string) { reported++ })
	_, err := provider.Invoke(context.Background(), antigravityChatRequest(`{"messages":[{"role":"user","content":"hi"}]}`, fixtureAntigravityCredential("")))
	if core.ClassifyError(err).StatusCode != http.StatusUnauthorized || reported != 0 || len(calls()) != 2 {
		t.Fatalf("err = %v, reported %d times", err, reported)
	}
}

// An account without a project cannot be served until it has one, which no
// retry changes; another target may serve the request.
func TestAntigravityWithoutAProjectIsAConfigurationError(t *testing.T) {
	t.Parallel()
	server, _ := recordingAntigravityServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, `{"currentTier":{}}`) })
	_, err := newFixtureAntigravity(t, server, nil).ListModels(context.Background(), fixtureAntigravityCredential(""))
	assertCodexFailure(t, err, core.ProviderErrorConfiguration, core.ProviderErrorClassification{FailoverEligible: true})
}

// Antigravity serves only Google OAuth tokens, so any other credential is a
// configuration error and nothing is sent.
func TestAntigravityFailsClosedWithoutAToken(t *testing.T) {
	t.Parallel()
	server, calls := recordingAntigravityServer(t, antigravityUpstream(t))
	provider := newFixtureAntigravity(t, server, nil)
	for name, credential := range map[string]*core.Credential{
		"none": nil, "API key": {APIKey: "fixture-key"}, "blank token": {Token: " ", Metadata: map[string]string{core.CredentialMetadataProjectID: "fixture-project"}},
	} {
		t.Run(name, func(t *testing.T) {
			configuration := core.ProviderErrorClassification{FailoverEligible: true}
			_, err := provider.Invoke(context.Background(), antigravityChatRequest(`{"messages":[]}`, credential))
			assertCodexFailure(t, err, core.ProviderErrorConfiguration, configuration)
			_, err = provider.ListModels(context.Background(), credential)
			assertCodexFailure(t, err, core.ProviderErrorConfiguration, configuration)
			_, err = provider.GenerateImages(context.Background(), core.GenerateImagesRequest{Model: "gemini-fixture-image", Prompt: "draw"}, credential)
			assertCodexFailure(t, err, core.ProviderErrorConfiguration, configuration)
		})
	}
	if sent := calls(); len(sent) != 0 {
		t.Fatalf("upstream received %d requests without a token", len(sent))
	}
}

// Antigravity refuses what it cannot serve before anything is sent. A
// request that cannot be mapped permits failover, since another target may
// carry it, while a malformed body does not.
func TestAntigravityRefusesWhatItCannotServeBeforeSending(t *testing.T) {
	t.Parallel()
	server, calls := recordingAntigravityServer(t, antigravityUpstream(t))
	provider := newFixtureAntigravity(t, server, nil)
	unsupported := core.ProviderErrorClassification{FailoverEligible: true}
	for name, tc := range map[string]struct {
		body, contentType string
		class             core.ProviderErrorClass
	}{
		"form body":          {body: `messages=hi`, contentType: "application/x-www-form-urlencoded", class: core.ProviderErrorInvalidRequest},
		"array body":         {body: `[]`, class: core.ProviderErrorInvalidRequest},
		"trailing body":      {body: `{"messages":[]} {}`, class: core.ProviderErrorInvalidRequest},
		"messages as text":   {body: `{"messages":"hi"}`, class: core.ProviderErrorUnsupported},
		"image content":      {body: `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/x.png"}}]}]}`, class: core.ProviderErrorUnsupported},
		"unknown role":       {body: `{"messages":[{"role":"narrator","content":"hi"}]}`, class: core.ProviderErrorUnsupported},
		"fractional length":  {body: `{"max_tokens":1.5,"messages":[]}`, class: core.ProviderErrorUnsupported},
		"unnamed tool reply": {body: `{"messages":[{"role":"tool","tool_call_id":"call-unknown","content":"x"}]}`, class: core.ProviderErrorUnsupported},
	} {
		t.Run(name, func(t *testing.T) {
			request := antigravityChatRequest(tc.body, fixtureAntigravityCredential("fixture-project"))
			if tc.contentType != "" {
				request.ContentType = tc.contentType
			}
			want := core.ProviderErrorClassification{}
			if tc.class == core.ProviderErrorUnsupported {
				want = unsupported
			}
			_, err := provider.Invoke(context.Background(), request)
			assertCodexFailure(t, err, tc.class, want)
		})
	}
	_, err := provider.Invoke(context.Background(), core.Request{Surface: core.ModelSurfaceResponses, Model: "gemini-fixture", Credential: fixtureAntigravityCredential("fixture-project")})
	var surface *core.SurfaceError
	if !errors.As(err, &surface) || surface.Surface != core.ModelSurfaceResponses {
		t.Fatalf("err = %v, want a SurfaceError", err)
	}
	if sent := calls(); len(sent) != 0 {
		t.Fatalf("upstream received %d requests Antigravity cannot serve", len(sent))
	}
}
