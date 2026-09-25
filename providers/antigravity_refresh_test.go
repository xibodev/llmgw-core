package providers_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	antigravityauth "github.com/xibodev/llm-provider-auth/antigravity"
	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

type antigravityOAuthRequest struct{ path, authorization, body string }

// antigravityOAuthServer is a synthetic Google token endpoint, with the
// loadCodeAssist endpoint account discovery reads. It answers every refresh
// with status and response and every discovery with discovery, or with 503
// when discovery is empty. The returned function takes the requests so far.
func antigravityOAuthServer(t *testing.T, status int, response map[string]any, discovery string) (antigravityauth.Config, func() []antigravityOAuthRequest) {
	t.Helper()
	var mu sync.Mutex
	var requests []antigravityOAuthRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		requests = append(requests, antigravityOAuthRequest{r.URL.Path, r.Header.Get("Authorization"), string(body)})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/load" && discovery == "":
			w.WriteHeader(http.StatusServiceUnavailable)
		case r.URL.Path == "/load":
			_, _ = io.WriteString(w, discovery)
		default:
			w.WriteHeader(status)
			_ = json.NewEncoder(w).Encode(response)
		}
	}))
	t.Cleanup(server.Close)
	config := antigravityauth.Config{
		ClientID: "fixture-client", ClientSecret: "fixture-secret", ClientAuthMode: antigravityauth.ClientAuthModeClientSecretPost,
		HTTPClient: server.Client(), Endpoints: antigravityauth.Endpoints{TokenURL: server.URL + "/token", LoadCodeAssistURL: server.URL + "/load"},
	}
	return config, func() []antigravityOAuthRequest {
		mu.Lock()
		defer mu.Unlock()
		taken := requests
		requests = nil
		return taken
	}
}

// expiredAntigravityCredential stores a credential the Coordinator must
// refresh before handing it out.
func expiredAntigravityCredential(t *testing.T, metadata map[string]string, refresh tokenstore.RefreshFunc) (*tokenstore.Memory, *tokenstore.Coordinator, tokenstore.Record) {
	t.Helper()
	store := tokenstore.NewMemory()
	record, err := store.Save(context.Background(), "antigravity-user", tokenstore.Record{
		AccessToken: "access-1", RefreshToken: "refresh-1", TokenType: "Bearer", AccountID: "account-fixture",
		Expiry: time.Now().Add(-time.Minute), Metadata: metadata,
	})
	if err != nil {
		t.Fatal(err)
	}
	coordinator, err := tokenstore.NewCoordinator(store, refresh)
	if err != nil {
		t.Fatal(err)
	}
	return store, coordinator, record
}

// runtimeLogin is a login with the product's own confidential client, as
// the gateway stores one: its project, account label and client.
func runtimeLogin() map[string]string {
	return map[string]string{
		core.CredentialMetadataOAuthProfile: providers.AntigravityOAuthProfileRuntimeSecret, core.CredentialMetadataOAuthClientID: "fixture-client",
		core.CredentialMetadataProjectID: "fixture-project", core.CredentialMetadataAccountLabel: "owner@example.test",
	}
}

const antigravityRuntimeRefresh = "client_id=fixture-client&client_secret=fixture-secret&grant_type=refresh_token&refresh_token=refresh-1"

// A refresh keeps the account, which Google does not report, and the
// project, unless the refreshed token discovers one, as the gateway
// discovers it after every refresh.
func TestAntigravityRefreshRotatesTokensAndKeepsTheAccountAndProject(t *testing.T) {
	t.Parallel()
	for name, tc := range map[string]struct {
		response               map[string]any
		discovery, wantRefresh string
		wantProject            string
	}{
		"no rotation or project": {map[string]any{"access_token": "access-2", "expires_in": 3600}, `{}`, "refresh-1", "fixture-project"},
		"rotation and a project": {
			map[string]any{"access_token": "access-2", "refresh_token": "refresh-2", "token_type": "Bearer", "expires_in": 3600},
			`{"cloudaicompanionProject":"fixture-project-new","paidTier":{"id":"fixture-tier"}}`, "refresh-2", "fixture-project-new",
		},
		"failed discovery": {map[string]any{"access_token": "access-2", "expires_in": 3600}, "", "refresh-1", "fixture-project"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			config, requests := antigravityOAuthServer(t, http.StatusOK, tc.response, tc.discovery)
			store, coordinator, _ := expiredAntigravityCredential(t, runtimeLogin(), providers.NewAntigravityRefresh(config))
			got, err := coordinator.Token(context.Background(), "antigravity-user")
			if err != nil {
				t.Fatal(err)
			}
			if got.AccessToken != "access-2" || got.RefreshToken != tc.wantRefresh || got.AccountID != "account-fixture" || got.TokenType != "Bearer" ||
				got.Metadata[core.CredentialMetadataProjectID] != tc.wantProject || got.Metadata[core.CredentialMetadataAccountLabel] != "owner@example.test" ||
				got.Expiry.Before(time.Now().Add(59*time.Minute)) || got.Expiry.After(time.Now().Add(61*time.Minute)) {
				t.Fatalf("refreshed = %v %+v", got, got.Metadata)
			}
			want := []antigravityOAuthRequest{
				{path: "/token", body: antigravityRuntimeRefresh},
				{path: "/load", authorization: "Bearer access-2", body: `{"metadata":{"ideType":"ANTIGRAVITY"}}`},
			}
			if sent := requests(); len(sent) != 2 || sent[0] != want[0] || sent[1] != want[1] {
				t.Fatalf("requests = %+v, want %+v", sent, want)
			}
			if stored, err := store.Load(context.Background(), "antigravity-user"); err != nil || stored.Revision != got.Revision {
				t.Fatalf("stored = %v, err = %v", stored, err)
			}
		})
	}
}
