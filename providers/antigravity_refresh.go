package providers

import (
	"context"
	"errors"
	"strings"

	antigravityauth "github.com/xibodev/llm-provider-auth/antigravity"
	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
)

// Values of core.CredentialMetadataOAuthProfile for an Antigravity
// credential, with the gateway's names: the OAuth client its owner signed
// in with, which decides how it refreshes.
const (
	// AntigravityOAuthProfileRuntimeSecret is the product's own
	// confidential client, whose secret stays in the product's
	// configuration.
	AntigravityOAuthProfileRuntimeSecret = "runtime_client_secret_post"
	// AntigravityOAuthProfilePublicPKCE is a public client, which proves a
	// login with PKCE alone.
	AntigravityOAuthProfilePublicPKCE = "public_pkce"
	// AntigravityOAuthProfileConsumerManual is a client its owner
	// registered and entered. The credential keeps its ID, its mode and
	// redirect URI and, for a confidential client, its secret.
	AntigravityOAuthProfileConsumerManual = "consumer_manual"
)

// Values of core.CredentialMetadataOAuthClientMode for a consumer_manual
// client, as the gateway stores them.
const (
	AntigravityOAuthClientPublic       = "public"
	AntigravityOAuthClientConfidential = "confidential"
)

// AntigravityRefreshOption changes what NewAntigravityRefresh does when a
// refresh fails.
type AntigravityRefreshOption func(*antigravityRefreshPolicy)

type antigravityRefreshPolicy struct{ revokeOnTerminal bool }

// AntigravityRevokeOnTerminal makes a grant that the token endpoint rejected
// for good, as with invalid_grant, revoke the credential: the refresh error
// keeps its terminal verdict, so the tokenstore.Coordinator revokes the
// credential and reports tokenstore.ErrRevoked. For a product that would
// rather drop a dead credential than keep it for its owner.
func AntigravityRevokeOnTerminal() AntigravityRefreshOption {
	return func(policy *antigravityRefreshPolicy) { policy.revokeOnTerminal = true }
}

// NewAntigravityRefresh returns how a tokenstore.Coordinator refreshes an
// Antigravity credential: the RefreshFunc a Runtime's RefreshFactory
// returns for an Antigravity instance.
//
// Each credential refreshes with the OAuth client it was granted to, chosen
// by its core.CredentialMetadataOAuthProfile as the gateway chooses it:
//
//   - runtime_client_secret_post: config's client, as a confidential
//     client. A credential that names another client fails, because
//     config's secret is not that client's.
//   - public_pkce: the client its core.CredentialMetadataOAuthClientID
//     names, or config's, without a secret.
//   - consumer_manual: the client the credential keeps, public or
//     confidential as its core.CredentialMetadataOAuthClientMode says, with
//     its core.CredentialMetadataOAuthClientSecret.
//   - none: config's client, or the client and secret the credential's
//     metadata names in its place.
//
// Config's endpoints and HTTP client serve every client.
//
// The refreshed record keeps the credential's account, since Google's
// token endpoint reports none, and its project, unless the refreshed token
// discovers one through config's DiscoverAccount, as the gateway looks it
// up after every refresh. A failed lookup is ignored. Whatever else the
// response omits, such as a refresh token Google did not rotate, the
// Coordinator keeps from the current record.
//
// A grant the token endpoint rejected for good fails that refresh but does
// not revoke the credential, because the gateway never revokes an
// Antigravity connection; its owner reauthorizes it instead.
// AntigravityRevokeOnTerminal makes it revoke. Either way errors.As still
// finds the endpoint's *browseroauth.EndpointError.
func NewAntigravityRefresh(config antigravityauth.Config, options ...AntigravityRefreshOption) tokenstore.RefreshFunc {
	var policy antigravityRefreshPolicy
	for _, option := range options {
		if option != nil {
			option(&policy)
		}
	}
	return func(ctx context.Context, current tokenstore.Record) (tokenstore.Record, error) {
		client, err := antigravityRefreshClient(config, current.Metadata)
		if err != nil {
			return tokenstore.Record{}, err
		}
		tokens, err := client.Refresh(ctx, current.RefreshToken)
		if err != nil {
			if !policy.revokeOnTerminal && tokenstore.IsTerminal(err) {
				err = &antigravityRefreshError{cause: err}
			}
			return tokenstore.Record{}, err
		}
		refreshed := tokenstore.Record{
			AccessToken: tokens.AccessToken, RefreshToken: tokens.RefreshToken, IDToken: tokens.IDToken,
			TokenType: tokens.TokenType, AccountID: current.AccountID,
		}
		if !tokens.ExpiresAt.IsZero() {
			refreshed.Expiry = tokens.ExpiresAt.UTC()
		}
		if account, err := client.DiscoverAccount(ctx, tokens.AccessToken); err == nil && account.ProjectID != "" {
			refreshed.Metadata = map[string]string{core.CredentialMetadataProjectID: account.ProjectID}
		}
		return refreshed, nil
	}
}

// antigravityRefreshClient returns the OAuth client a credential was
// granted to. Config's secret belongs to config's client, so it never goes
// out with another client's ID.
func antigravityRefreshClient(config antigravityauth.Config, metadata map[string]string) (antigravityauth.Config, error) {
	client := config
	configured := strings.TrimSpace(config.ClientID)
	clientID := strings.TrimSpace(metadata[core.CredentialMetadataOAuthClientID])
	secret := metadata[core.CredentialMetadataOAuthClientSecret]
	switch strings.TrimSpace(metadata[core.CredentialMetadataOAuthProfile]) {
	case "":
		if clientID != "" && clientID != configured {
			client.ClientID, client.ClientSecret = clientID, secret
		}
	case AntigravityOAuthProfileRuntimeSecret:
		if clientID != "" && clientID != configured {
			return antigravityauth.Config{}, errors.New("the Antigravity credential was granted to an OAuth client that is not configured; reauthorize it")
		}
		client.ClientAuthMode = antigravityauth.ClientAuthModeClientSecretPost
	case AntigravityOAuthProfilePublicPKCE:
		if clientID != "" {
			client.ClientID = clientID
		}
		client.ClientSecret, client.ClientAuthMode = "", antigravityauth.ClientAuthModePublicPKCE
	case AntigravityOAuthProfileConsumerManual:
		// Only the endpoints are config's: the client is the owner's own.
		client.ClientID, client.RedirectURI = clientID, strings.TrimSpace(metadata[core.CredentialMetadataOAuthRedirectURI])
		switch strings.ToLower(strings.TrimSpace(metadata[core.CredentialMetadataOAuthClientMode])) {
		case AntigravityOAuthClientPublic:
			client.ClientSecret, client.ClientAuthMode = "", antigravityauth.ClientAuthModePublicPKCE
		case AntigravityOAuthClientConfidential:
			if strings.TrimSpace(secret) == "" {
				return antigravityauth.Config{}, errors.New("the confidential Antigravity OAuth client has no secret; reauthorize it")
			}
			client.ClientSecret, client.ClientAuthMode = secret, antigravityauth.ClientAuthModeClientSecretPost
		default:
			return antigravityauth.Config{}, errors.New("the Antigravity OAuth client mode must be public or confidential")
		}
	default:
		return antigravityauth.Config{}, errors.New("the Antigravity OAuth client profile is unsupported")
	}
	return client, nil
}

// antigravityRefreshError is a refresh failure that must not revoke the
// credential, however final the token endpoint called it.
type antigravityRefreshError struct{ cause error }

func (e *antigravityRefreshError) Error() string { return e.cause.Error() }
func (e *antigravityRefreshError) Unwrap() error { return e.cause }

// Terminal answers tokenstore.IsTerminal, which takes the verdict of the
// outermost error that gives one, before the cause can.
func (e *antigravityRefreshError) Terminal() bool { return false }
