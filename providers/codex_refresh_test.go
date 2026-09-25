package providers_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	codexauth "github.com/xibodev/llm-provider-auth/codex"
	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

// fixtureJWT is an unsigned token carrying claims, the way Codex's ID and
// access tokens carry the account and the expiry.
func fixtureJWT(claims map[string]any) string {
	payload, _ := json.Marshal(claims)
	return "e30." + base64.RawURLEncoding.EncodeToString(payload) + ".fixture-signature"
}

func fixtureIDToken(accountID string) string {
	return fixtureJWT(map[string]any{"https://api.openai.com/auth": map[string]string{"chatgpt_account_id": accountID}})
}

// codexTokenEndpoint answers every refresh with status and response.
func codexTokenEndpoint(t *testing.T, status int, response map[string]any) (codexauth.Config, *atomic.Int32) {
	t.Helper()
	calls := &atomic.Int32{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body) != 3 || body["grant_type"] != "refresh_token" ||
			body["client_id"] != "fixture-client" || body["refresh_token"] != "refresh-1" {
			t.Errorf("refresh request = %v, %v", body, err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(response)
	}))
	t.Cleanup(server.Close)
	return codexauth.Config{
		ClientID: "fixture-client", HTTPClient: server.Client(),
		Endpoints: codexauth.Endpoints{OAuthTokenURL: server.URL + "/oauth/token"},
	}, calls
}

// expiredCodexCredential stores a credential the Coordinator must refresh
// before handing it out.
func expiredCodexCredential(t *testing.T, config codexauth.Config) (*tokenstore.Memory, *tokenstore.Coordinator, tokenstore.Record) {
	t.Helper()
	store := tokenstore.NewMemory()
	record, err := store.Save(context.Background(), "codex-user", tokenstore.Record{
		AccessToken: "access-1", RefreshToken: "refresh-1", IDToken: fixtureIDToken("account-fixture"),
		TokenType: "Bearer", AccountID: "account-fixture", Expiry: time.Now().Add(-time.Minute),
		Metadata: map[string]string{"label": "fixture"},
	})
	if err != nil {
		t.Fatal(err)
	}
	coordinator, err := tokenstore.NewCoordinator(store, providers.NewCodexRefresh(config))
	if err != nil {
		t.Fatal(err)
	}
	return store, coordinator, record
}

func TestCodexRefreshRotatesTokensAndKeepsTheAccount(t *testing.T) {
	t.Parallel()
	accessExpiry := time.Now().Add(2 * time.Hour).Truncate(time.Second)
	for name, tc := range map[string]struct {
		response    map[string]any
		wantRefresh string
		wantExpiry  func(time.Time) bool
	}{
		"full response": {
			response: map[string]any{
				"access_token": "access-2", "refresh_token": "refresh-2", "token_type": "Bearer",
				"id_token": fixtureIDToken("account-fixture"), "expires_in": 3600,
			},
			wantRefresh: "refresh-2",
			wantExpiry: func(expiry time.Time) bool {
				return expiry.After(time.Now().Add(59*time.Minute)) && expiry.Before(time.Now().Add(61*time.Minute))
			},
		},
		"only an access token": {
			response:    map[string]any{"access_token": fixtureJWT(map[string]any{"exp": accessExpiry.Unix()})},
			wantRefresh: "refresh-1",
			wantExpiry:  func(expiry time.Time) bool { return expiry.Equal(accessExpiry) },
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			config, calls := codexTokenEndpoint(t, http.StatusOK, tc.response)
			store, coordinator, _ := expiredCodexCredential(t, config)
			got, err := coordinator.Token(context.Background(), "codex-user")
			if err != nil {
				t.Fatal(err)
			}
			if got.AccessToken != tc.response["access_token"] || got.RefreshToken != tc.wantRefresh ||
				got.AccountID != "account-fixture" || got.TokenType != "Bearer" || got.Metadata["label"] != "fixture" ||
				!tc.wantExpiry(got.Expiry) || calls.Load() != 1 {
				t.Fatalf("refreshed = %v %+v after %d calls", got, got.Metadata, calls.Load())
			}
			if stored, err := store.Load(context.Background(), "codex-user"); err != nil || stored.Revision != got.Revision {
				t.Fatalf("stored = %v, err = %v", stored, err)
			}
		})
	}
}

// A grant the endpoint rejected for good revokes the credential; any other
// failure leaves it for the next attempt. Either way the wrapper returns the
// endpoint's error unchanged, which is what lets the Coordinator tell them
// apart.
func TestCodexRefreshRevokesOnlyARejectedGrant(t *testing.T) {
	t.Parallel()
	for name, tc := range map[string]struct {
		status   int
		code     string
		terminal bool
	}{
		"invalid grant":           {status: http.StatusBadRequest, code: "invalid_grant", terminal: true},
		"reused refresh token":    {status: http.StatusUnauthorized, code: "refresh_token_reused", terminal: true},
		"temporarily unavailable": {status: http.StatusServiceUnavailable, code: "temporarily_unavailable"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			config, _ := codexTokenEndpoint(t, tc.status, map[string]any{"error": tc.code, "error_description": "fixture"})
			store, coordinator, record := expiredCodexCredential(t, config)
			_, err := providers.NewCodexRefresh(config)(context.Background(), record)
			if refreshErr, ok := err.(*codexauth.RefreshError); !ok || refreshErr.StatusCode != tc.status || refreshErr.Terminal() != tc.terminal {
				t.Fatalf("refresh error = %T %v, want the endpoint's *codexauth.RefreshError", err, err)
			}
			_, err = coordinator.Token(context.Background(), "codex-user")
			if errors.Is(err, tokenstore.ErrRevoked) != tc.terminal || tokenstore.IsTerminal(err) != tc.terminal {
				t.Fatalf("Token err = %v, want revoked %v", err, tc.terminal)
			}
			stored, loadErr := store.Load(context.Background(), "codex-user")
			if tc.terminal && !errors.Is(loadErr, tokenstore.ErrNotFound) {
				t.Fatalf("a rejected grant left the credential: %v, %v", stored, loadErr)
			}
			if !tc.terminal && (loadErr != nil || stored.Revision != record.Revision) {
				t.Fatalf("a transient failure changed the credential: %v, %v", stored, loadErr)
			}
		})
	}
}

func TestCodexRefreshRefusesAnotherAccount(t *testing.T) {
	t.Parallel()
	for name, response := range map[string]map[string]any{
		"from the ID token": {"access_token": "access-2", "refresh_token": "refresh-2", "id_token": fixtureIDToken("account-other")},
		"from the response": {"access_token": "access-2", "refresh_token": "refresh-2", "account_id": "account-other"},
		"from the response over a matching ID token": {
			"access_token": "access-2", "id_token": fixtureIDToken("account-fixture"), "chatgpt_account_id": "account-other",
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			config, calls := codexTokenEndpoint(t, http.StatusOK, response)
			store, coordinator, record := expiredCodexCredential(t, config)
			if _, err := coordinator.Token(context.Background(), "codex-user"); !errors.Is(err, tokenstore.ErrIdentityChanged) {
				t.Fatalf("err = %v, want tokenstore.ErrIdentityChanged", err)
			}
			stored, err := store.Load(context.Background(), "codex-user")
			if err != nil || stored.Revision != record.Revision || stored.AccessToken != "access-1" || calls.Load() != 1 {
				t.Fatalf("the other account's tokens were kept: %v, %v", stored, err)
			}
		})
	}
}

type tokenRequest struct{ method, path, contentType, accept, body string }

// recordingTokenEndpoint answers every request with status and response,
// and hands each request to the test.
func recordingTokenEndpoint(t *testing.T, status int, response map[string]any) (codexauth.Config, <-chan tokenRequest) {
	t.Helper()
	requests := make(chan tokenRequest, 8)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		requests <- tokenRequest{r.Method, r.URL.Path, r.Header.Get("Content-Type"), r.Header.Get("Accept"), string(body)}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(response)
	}))
	t.Cleanup(server.Close)
	return codexauth.Config{
		ClientID: "fixture-configured-client", HTTPClient: server.Client(),
		Endpoints: codexauth.Endpoints{OAuthTokenURL: server.URL + "/oauth/token"},
	}, requests
}

func browserLogin() map[string]string {
	return map[string]string{
		core.CredentialMetadataOAuthProfile: providers.CodexOAuthProfileBrowser, core.CredentialMetadataOAuthClientID: "fixture-browser-client",
		core.CredentialMetadataOAuthClientMode: "public", core.CredentialMetadataOAuthRedirectURI: "http://localhost:1455/auth/callback",
	}
}

func deviceLogin() map[string]string {
	return map[string]string{core.CredentialMetadataOAuthProfile: providers.CodexOAuthProfileDevice, core.CredentialMetadataOAuthClientID: "fixture-client"}
}

// Each credential refreshes with the client it was granted to, in the
// request the gateway sends for its kind of login. The device body is the
// one the gateway's characterization golden records.
func TestCodexRefreshUsesTheClientTheCredentialWasGrantedTo(t *testing.T) {
	t.Parallel()
	for name, tc := range map[string]struct {
		metadata          map[string]string
		contentType, body string
	}{
		"device login": {deviceLogin(), "application/json",
			`{"client_id":"fixture-client","grant_type":"refresh_token","refresh_token":"fixture-refresh"}`},
		"browser login": {browserLogin(), "application/x-www-form-urlencoded",
			"client_id=fixture-browser-client&grant_type=refresh_token&refresh_token=fixture-refresh"},
		"no metadata": {nil, "application/json",
			`{"client_id":"fixture-configured-client","grant_type":"refresh_token","refresh_token":"fixture-refresh"}`},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			config, requests := recordingTokenEndpoint(t, http.StatusOK, map[string]any{
				"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600,
				"id_token": fixtureJWT(map[string]any{
					"email": "owner@example.test", "https://api.openai.com/auth": map[string]string{"chatgpt_account_id": "account-fixture"},
				}),
			})
			store := tokenstore.NewMemory()
			if _, err := store.Save(context.Background(), "codex-user", tokenstore.Record{
				AccessToken: "access-1", RefreshToken: "fixture-refresh", AccountID: "account-fixture",
				Expiry: time.Now().Add(-time.Minute), Metadata: tc.metadata,
			}); err != nil {
				t.Fatal(err)
			}
			coordinator, err := tokenstore.NewCoordinator(store, providers.NewCodexRefresh(config))
			if err != nil {
				t.Fatal(err)
			}
			got, err := coordinator.Token(context.Background(), "codex-user")
			if err != nil {
				t.Fatal(err)
			}
			want := tokenRequest{http.MethodPost, "/oauth/token", tc.contentType, "application/json", tc.body}
			if request := <-requests; request != want {
				t.Fatalf("token request = %+v, want %+v", request, want)
			}
			if got.AccessToken != "access-2" || got.RefreshToken != "refresh-2" || got.AccountID != "account-fixture" ||
				got.Expiry.Before(time.Now().Add(59*time.Minute)) || got.Metadata[core.CredentialMetadataAccountLabel] != "owner@example.test" {
				t.Fatalf("refreshed = %v %+v", got, got.Metadata)
			}
			// The next refresh must use the same client, so the binding survives.
			for key, value := range tc.metadata {
				if got.Metadata[key] != value {
					t.Fatalf("refresh lost %s: %+v", key, got.Metadata)
				}
			}
		})
	}
}

// A browser login takes its account from the ID token alone, as the
// gateway's does. A device login takes the response's first.
func TestCodexRefreshReadsTheAccountAsTheGatewayDoesForEachLogin(t *testing.T) {
	t.Parallel()
	config, _ := recordingTokenEndpoint(t, http.StatusOK, map[string]any{
		"access_token": "access-2", "account_id": "account-response", "id_token": fixtureIDToken("account-fixture"),
	})
	refresh := providers.NewCodexRefresh(config)
	for _, tc := range []struct {
		metadata map[string]string
		want     string
	}{
		{browserLogin(), "account-fixture"},
		{deviceLogin(), "account-response"},
	} {
		got, err := refresh(context.Background(), tokenstore.Record{RefreshToken: "fixture-refresh", Metadata: tc.metadata})
		if err != nil || got.AccountID != tc.want {
			t.Fatalf("%s refresh account = %q, err = %v, want %q", tc.metadata[core.CredentialMetadataOAuthProfile], got.AccountID, err, tc.want)
		}
	}
}

func TestCodexBrowserRefreshReportsARejectedGrantAsTheDeviceRefreshDoes(t *testing.T) {
	t.Parallel()
	config, _ := recordingTokenEndpoint(t, http.StatusBadRequest, map[string]any{"error": "invalid_grant", "error_description": "fixture"})
	store := tokenstore.NewMemory()
	record, err := store.Save(context.Background(), "codex-user", tokenstore.Record{
		AccessToken: "access-1", RefreshToken: "fixture-refresh", Expiry: time.Now().Add(-time.Minute), Metadata: browserLogin(),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = providers.NewCodexRefresh(config)(context.Background(), record)
	if refreshErr, ok := err.(*codexauth.RefreshError); !ok || refreshErr.StatusCode != http.StatusBadRequest || refreshErr.Code != "invalid_grant" || !refreshErr.Terminal() {
		t.Fatalf("browser refresh error = %T %v, want a terminal *codexauth.RefreshError", err, err)
	}
	coordinator, err := tokenstore.NewCoordinator(store, providers.NewCodexRefresh(config))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Token(context.Background(), "codex-user"); !errors.Is(err, tokenstore.ErrRevoked) {
		t.Fatalf("Token err = %v, want tokenstore.ErrRevoked", err)
	}
	if _, err := store.Load(context.Background(), "codex-user"); !errors.Is(err, tokenstore.ErrNotFound) {
		t.Fatalf("a rejected browser grant left the credential: %v", err)
	}
}
