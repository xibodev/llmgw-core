package oauthflow_test

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/xibodev/llm-provider-auth/browseroauth"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/oauthflow"
)

// TestBrowserPKCEDriverThroughTheService runs a real browseroauth flow
// against a fake token endpoint that checks the PKCE proof.
func TestBrowserPKCEDriverThroughTheService(t *testing.T) {
	t.Parallel()
	var (
		mu        sync.Mutex
		verifiers []string
	)
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil || r.Form.Get("grant_type") != "authorization_code" ||
			r.Form.Get("code") != fixtureCode || r.Form.Get("redirect_uri") != fixtureRedirect || r.Form.Get("client_id") != "fixture-client" {
			http.Error(w, `{"error":"invalid_request"}`, http.StatusBadRequest)
			return
		}
		mu.Lock()
		verifiers = append(verifiers, r.Form.Get("code_verifier"))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"` + secretAccess + `","refresh_token":"` + secretRefresh + `","token_type":"Bearer","expires_in":3600}`))
	}))
	defer endpoint.Close()

	driver := oauthflow.BrowserPKCE{Config: browseroauth.Config{
		AuthorizeURL: "https://auth.example.test/authorize", TokenURL: endpoint.URL + "/token",
		ClientID: "fixture-client", ClientAuthMode: browseroauth.ClientAuthModePublicPKCE,
		Scopes: []string{"openid"}, HTTPClient: endpoint.Client(),
	}}
	credentials := core.NewMemoryCredentialStore()
	service, err := oauthflow.New(oauthflow.Options{
		Store: oauthflow.NewMemoryFlowStore(nil), Credentials: credentials,
		Drivers: func(string, oauthflow.Method) (oauthflow.Driver, error) { return driver, nil },
		CredentialKey: func(_ context.Context, completion oauthflow.Completion) (string, error) {
			return completion.Caller.ID + "/" + completion.Instance, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	view, err := service.Start(ctx, owner(), "fixture-provider", oauthflow.MethodBrowser, oauthflow.WithRedirectURI(fixtureRedirect))
	if err != nil {
		t.Fatal(err)
	}
	authorization, err := url.Parse(view.AuthorizationURL)
	if err != nil {
		t.Fatal(err)
	}
	query := authorization.Query()
	state, challenge := query.Get("state"), query.Get("code_challenge")
	if state == "" || challenge == "" || query.Get("code_challenge_method") != "S256" || query.Get("redirect_uri") != fixtureRedirect {
		t.Fatalf("authorization URL %q lacks PKCE or state", view.AuthorizationURL)
	}

	done, err := service.Callback(ctx, oauthflow.CompleteInput{Code: fixtureCode, State: state, RedirectURI: fixtureRedirect})
	if err != nil || done.Status != oauthflow.StatusComplete {
		t.Fatalf("Callback: view=%+v err=%v", done, err)
	}
	if len(verifiers) != 1 {
		t.Fatalf("token endpoint saw %d exchanges, want 1", len(verifiers))
	}
	digest := sha256.Sum256([]byte(verifiers[0]))
	if base64.RawURLEncoding.EncodeToString(digest[:]) != challenge {
		t.Fatal("the exchanged verifier does not prove the authorization URL's challenge")
	}
	for _, shown := range []oauthflow.View{view, done} {
		assertNoSecrets(t, shown)
		if text := visibleText(shown); strings.Contains(text, verifiers[0]) || strings.Contains(text, state) {
			t.Fatal("a view exposes the generated verifier or state")
		}
	}
	record, err := credentials.Load(ctx, "user-1/fixture-provider")
	if err != nil || record.AccessToken != secretAccess || record.RefreshToken != secretRefresh || record.Expiry.IsZero() {
		t.Fatalf("saved credential=%s err=%v", record, err)
	}
	_, err = service.Callback(ctx, oauthflow.CompleteInput{Code: fixtureCode, State: state})
	wantErr(t, err, oauthflow.ErrFlowNotFound, "replayed callback")
}
