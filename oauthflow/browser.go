package oauthflow

import (
	"context"
	"strings"
	"time"

	"github.com/xibodev/llm-provider-auth/browseroauth"
	"github.com/xibodev/llm-provider-auth/tokenstore"
)

// BrowserPKCE is a ready driver for a generic authorization-code flow with
// PKCE S256, built on llm-provider-auth's browseroauth. It serves
// MethodBrowser and MethodManual: browseroauth generates the state and
// verifier, and the Service keeps both on the server.
type BrowserPKCE struct {
	// Config names the provider's endpoints, client and scopes.
	Config browseroauth.Config
	// RedirectURI is used when the start request names none, such as the
	// registered loopback redirect of a manual flow.
	RedirectURI string
	// TTL is how long an attempt stays valid; zero uses DefaultTTL.
	TTL time.Duration
	// Record converts the token response. Nil copies its tokens and expiry;
	// a product sets it to add the account or project its provider reports.
	Record func(ctx context.Context, tokens browseroauth.TokenEnvelope) (tokenstore.Record, error)
}

var _ CodeDriver = BrowserPKCE{}

// Start implements Driver.
func (d BrowserPKCE) Start(_ context.Context, request StartRequest) (Authorization, error) {
	redirectURI := strings.TrimSpace(request.RedirectURI)
	if redirectURI == "" {
		redirectURI = strings.TrimSpace(d.RedirectURI)
	}
	authorization, err := d.Config.AuthorizationURL(redirectURI)
	if err != nil {
		return Authorization{}, err
	}
	return Authorization{
		AuthorizationURL: authorization.URL,
		ExpiresIn:        d.TTL,
		Secrets: Secrets{
			Verifier: authorization.CodeVerifier, State: authorization.State, RedirectURI: redirectURI,
		},
	}, nil
}

// Exchange implements CodeDriver. browseroauth's errors carry only the
// endpoint's sanitized error code and description.
func (d BrowserPKCE) Exchange(ctx context.Context, flow Flow, code string) (tokenstore.Record, error) {
	tokens, err := d.Config.Exchange(ctx, code, flow.Secrets.Verifier, flow.Secrets.RedirectURI)
	if err != nil {
		return tokenstore.Record{}, err
	}
	if d.Record != nil {
		return d.Record(ctx, tokens)
	}
	return tokenstore.Record{
		AccessToken: tokens.AccessToken, RefreshToken: tokens.RefreshToken, IDToken: tokens.IDToken,
		TokenType: tokens.TokenType, Expiry: tokens.ExpiresAt,
	}, nil
}
