package providers

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/xibodev/llm-provider-auth/browseroauth"
	codexauth "github.com/xibodev/llm-provider-auth/codex"
	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
)

// Values of core.CredentialMetadataOAuthProfile for a Codex credential: how
// its owner signed in, which decides how it refreshes.
const (
	CodexOAuthProfileDevice  = "device_client_id"
	CodexOAuthProfileBrowser = "browser_pkce"
)

// NewCodexRefresh returns how a tokenstore.Coordinator refreshes a Codex
// credential: the RefreshFunc a Runtime's RefreshFactory returns for a
// Codex instance.
//
// Each credential refreshes with the OAuth client it was granted to, as the
// gateway refreshes each connection: the one its
// core.CredentialMetadataOAuthClientID names, or config.ClientID when it
// names none. A browser login, whose core.CredentialMetadataOAuthProfile is
// CodexOAuthProfileBrowser, refreshes as that public PKCE client with a
// form-encoded request, as it was granted. Any other credential refreshes
// through config.Refresh. The profile decides, not the client mode: every
// Codex browser login is a public client.
//
// The refreshed record carries the account its tokens act for, so the
// Coordinator refuses a refresh that returns another account with
// tokenstore.ErrIdentityChanged, as the gateway always has. A device
// refresh takes it from the token response or else its ID token, a browser
// refresh from the ID token alone. The record carries the reported expiry,
// the account's label under core.CredentialMetadataAccountLabel, and, after
// a device refresh, the access token's own expiry when none is reported.
// Whatever the response omits, such as a refresh token it did not rotate,
// the Coordinator keeps from the current record.
//
// A grant the token endpoint rejected for good fails with a
// *codexauth.RefreshError, from either kind of login, whose Terminal method
// makes the Coordinator revoke the credential rather than retry it. Other
// errors are returned unchanged.
func NewCodexRefresh(config codexauth.Config) tokenstore.RefreshFunc {
	return func(ctx context.Context, current tokenstore.Record) (tokenstore.Record, error) {
		client := config
		if clientID := strings.TrimSpace(current.Metadata[core.CredentialMetadataOAuthClientID]); clientID != "" {
			client.ClientID = clientID
		}
		if strings.TrimSpace(current.Metadata[core.CredentialMetadataOAuthProfile]) == CodexOAuthProfileBrowser {
			return codexBrowserRefresh(ctx, client, current.RefreshToken)
		}
		tokens, err := client.Refresh(ctx, current.RefreshToken)
		if err != nil {
			return tokenstore.Record{}, err
		}
		return codexRecord(tokens), nil
	}
}

// codexBrowserRefresh refreshes a browser login through browseroauth, as
// the gateway does, and reports a rejected grant as the device refresh
// reports one.
func codexBrowserRefresh(ctx context.Context, config codexauth.Config, refreshToken string) (tokenstore.Record, error) {
	tokenURL := strings.TrimSpace(config.Endpoints.OAuthTokenURL)
	if tokenURL == "" {
		tokenURL = codexauth.OAuthTokenURL
	}
	browser := browseroauth.Config{
		TokenURL: tokenURL, ClientID: strings.TrimSpace(config.ClientID),
		ClientAuthMode: browseroauth.ClientAuthModePublicPKCE, HTTPClient: config.HTTPClient,
	}
	tokens, err := browser.Refresh(ctx, refreshToken)
	if err != nil {
		var endpoint *browseroauth.EndpointError
		if errors.As(err, &endpoint) {
			return tokenstore.Record{}, &codexauth.RefreshError{StatusCode: endpoint.StatusCode, Code: endpoint.Code, Description: endpoint.Description}
		}
		return tokenstore.Record{}, err
	}
	accountID, label := codexIDTokenIdentity(tokens.IDToken)
	refreshed := codexauth.TokenSet{
		AccessToken: tokens.AccessToken, RefreshToken: tokens.RefreshToken, IDToken: tokens.IDToken,
		TokenType: tokens.TokenType, AccountID: accountID, AccountLabel: label,
	}
	if !tokens.ExpiresAt.IsZero() {
		refreshed.ExpiresAt = tokens.ExpiresAt.Unix()
	}
	return codexRecord(refreshed), nil
}

// codexIDTokenIdentity reads the ChatGPT account and its label from an ID
// token, as the gateway reads a browser login's. The signature is not
// checked: the token came straight from the token endpoint over TLS.
func codexIDTokenIdentity(idToken string) (accountID, label string) {
	parts := strings.Split(idToken, ".")
	if len(parts) != 3 {
		return "", ""
	}
	raw, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", ""
	}
	var claims struct {
		Email     string `json:"email"`
		AccountID string `json:"chatgpt_account_id"`
		Profile   struct {
			Email string `json:"email"`
		} `json:"https://api.openai.com/profile"`
		Auth struct {
			AccountID string `json:"chatgpt_account_id"`
		} `json:"https://api.openai.com/auth"`
	}
	if json.Unmarshal(raw, &claims) != nil {
		return "", ""
	}
	accountID = strings.TrimSpace(claims.Auth.AccountID)
	if accountID == "" {
		accountID = strings.TrimSpace(claims.AccountID)
	}
	label = strings.TrimSpace(claims.Email)
	if label == "" {
		label = strings.TrimSpace(claims.Profile.Email)
	}
	return accountID, label
}

// codexRecord maps refreshed tokens onto a record.
func codexRecord(tokens codexauth.TokenSet) tokenstore.Record {
	record := tokenstore.Record{
		AccessToken: tokens.AccessToken, RefreshToken: tokens.RefreshToken, IDToken: tokens.IDToken,
		TokenType: tokens.TokenType, AccountID: strings.TrimSpace(tokens.AccountID),
	}
	if tokens.ExpiresAt > 0 {
		record.Expiry = time.Unix(tokens.ExpiresAt, 0).UTC()
	}
	if label := strings.TrimSpace(tokens.AccountLabel); label != "" {
		record.Metadata = map[string]string{core.CredentialMetadataAccountLabel: label}
	}
	return record
}
