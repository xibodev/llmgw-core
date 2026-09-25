package providers_test

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/xibodev/llm-provider-auth/browseroauth"
	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

// Each credential refreshes with the client it was granted to, as the
// gateway selects it by profile, and keeps that client for the next one.
func TestAntigravityRefreshUsesTheClientTheCredentialWasGrantedTo(t *testing.T) {
	t.Parallel()
	manual := func(mode, secret string) map[string]string {
		return map[string]string{
			core.CredentialMetadataOAuthProfile: providers.AntigravityOAuthProfileConsumerManual, core.CredentialMetadataOAuthClientID: "fixture-manual-client",
			core.CredentialMetadataOAuthClientMode: mode, core.CredentialMetadataOAuthClientSecret: secret,
			core.CredentialMetadataOAuthRedirectURI: "https://callback.example.test/oauth",
		}
	}
	for name, tc := range map[string]struct {
		metadata map[string]string
		body     string
	}{
		"runtime client":               {runtimeLogin(), antigravityRuntimeRefresh},
		"runtime client without an ID": {map[string]string{core.CredentialMetadataOAuthProfile: providers.AntigravityOAuthProfileRuntimeSecret}, antigravityRuntimeRefresh},
		"public client": {map[string]string{
			core.CredentialMetadataOAuthProfile: providers.AntigravityOAuthProfilePublicPKCE, core.CredentialMetadataOAuthClientID: "fixture-public-client",
		}, "client_id=fixture-public-client&grant_type=refresh_token&refresh_token=refresh-1"},
		"confidential manual client": {manual("confidential", "fixture-manual-secret"),
			"client_id=fixture-manual-client&client_secret=fixture-manual-secret&grant_type=refresh_token&refresh_token=refresh-1"},
		"public manual client": {manual("Public", ""), "client_id=fixture-manual-client&grant_type=refresh_token&refresh_token=refresh-1"},
		"no metadata":          {nil, antigravityRuntimeRefresh},
		"no profile, another client": {map[string]string{
			core.CredentialMetadataOAuthClientID: "fixture-other-client", core.CredentialMetadataOAuthClientSecret: "fixture-other-secret",
		}, "client_id=fixture-other-client&client_secret=fixture-other-secret&grant_type=refresh_token&refresh_token=refresh-1"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			config, requests := antigravityOAuthServer(t, http.StatusOK, map[string]any{"access_token": "access-2", "expires_in": 3600}, `{}`)
			_, coordinator, _ := expiredAntigravityCredential(t, tc.metadata, providers.NewAntigravityRefresh(config))
			got, err := coordinator.Token(context.Background(), "antigravity-user")
			if err != nil {
				t.Fatal(err)
			}
			if sent := requests(); len(sent) == 0 || sent[0].path != "/token" || sent[0].body != tc.body {
				t.Fatalf("token request = %+v, want %s", sent, tc.body)
			}
			for key, value := range tc.metadata {
				if got.Metadata[key] != value {
					t.Fatalf("refresh lost %s: %+v", key, got.Metadata)
				}
			}
		})
	}
}

// A credential whose client Antigravity cannot authenticate as fails
// before anything is sent, and is left as it was.
func TestAntigravityRefreshRefusesAClientItCannotAuthenticateAs(t *testing.T) {
	t.Parallel()
	for name, metadata := range map[string]map[string]string{
		"runtime profile, another client": {core.CredentialMetadataOAuthProfile: providers.AntigravityOAuthProfileRuntimeSecret, core.CredentialMetadataOAuthClientID: "fixture-other-client"},
		"manual client without a secret": {core.CredentialMetadataOAuthProfile: providers.AntigravityOAuthProfileConsumerManual,
			core.CredentialMetadataOAuthClientID: "fixture-manual-client", core.CredentialMetadataOAuthClientMode: "confidential"},
		"manual client without a mode": {core.CredentialMetadataOAuthProfile: providers.AntigravityOAuthProfileConsumerManual, core.CredentialMetadataOAuthClientID: "fixture-manual-client"},
		"unknown profile":              {core.CredentialMetadataOAuthProfile: "fixture-profile"},
		"another client, no secret":    {core.CredentialMetadataOAuthClientID: "fixture-other-client"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			config, requests := antigravityOAuthServer(t, http.StatusOK, map[string]any{"access_token": "access-2"}, `{}`)
			store, coordinator, record := expiredAntigravityCredential(t, metadata, providers.NewAntigravityRefresh(config))
			if _, err := coordinator.Token(context.Background(), "antigravity-user"); err == nil || tokenstore.IsTerminal(err) {
				t.Fatalf("err = %v, want a refusal that is not terminal", err)
			}
			if sent := requests(); len(sent) != 0 {
				t.Fatalf("requests = %+v", sent)
			}
			if stored, err := store.Load(context.Background(), "antigravity-user"); err != nil || stored.Revision != record.Revision {
				t.Fatalf("stored = %v, err = %v", stored, err)
			}
		})
	}
}

// The gateway never revokes an Antigravity connection, so by default a
// rejected grant fails the refresh and leaves the credential; the endpoint's
// error is still found, with its own terminal verdict. With
// AntigravityRevokeOnTerminal the Coordinator revokes it. A transient
// failure never revokes.
func TestAntigravityRefreshRevokesARejectedGrantOnlyWhenAsked(t *testing.T) {
	t.Parallel()
	for name, tc := range map[string]struct {
		status           int
		code             string
		revokeOnTerminal bool
		revoked          bool
	}{
		"rejected grant":                    {status: http.StatusBadRequest, code: "invalid_grant"},
		"rejected grant, revoking":          {status: http.StatusBadRequest, code: "invalid_grant", revokeOnTerminal: true, revoked: true},
		"temporarily unavailable":           {status: http.StatusServiceUnavailable, code: "temporarily_unavailable"},
		"temporarily unavailable, revoking": {status: http.StatusServiceUnavailable, code: "temporarily_unavailable", revokeOnTerminal: true},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			config, _ := antigravityOAuthServer(t, tc.status, map[string]any{"error": tc.code, "error_description": "fixture"}, `{}`)
			var options []providers.AntigravityRefreshOption
			if tc.revokeOnTerminal {
				options = append(options, providers.AntigravityRevokeOnTerminal())
			}
			store, coordinator, record := expiredAntigravityCredential(t, runtimeLogin(), providers.NewAntigravityRefresh(config, options...))
			_, err := coordinator.Token(context.Background(), "antigravity-user")
			var endpoint *browseroauth.EndpointError
			if !errors.As(err, &endpoint) || endpoint.StatusCode != tc.status || endpoint.Code != tc.code || endpoint.Terminal() != (tc.code == "invalid_grant") {
				t.Fatalf("err = %v, want the endpoint's error", err)
			}
			if errors.Is(err, tokenstore.ErrRevoked) != tc.revoked || tokenstore.IsTerminal(err) != tc.revoked {
				t.Fatalf("err = %v, want revoked %v", err, tc.revoked)
			}
			stored, loadErr := store.Load(context.Background(), "antigravity-user")
			if tc.revoked && !errors.Is(loadErr, tokenstore.ErrNotFound) {
				t.Fatalf("a revoked grant left the credential: %v, %v", stored, loadErr)
			}
			if !tc.revoked && (loadErr != nil || stored.Revision != record.Revision) {
				t.Fatalf("the credential changed: %v, %v", stored, loadErr)
			}
		})
	}
}
