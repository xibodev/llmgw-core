package providers

import (
	"context"
	"strings"
	"time"

	codexauth "github.com/xibodev/llm-provider-auth/codex"
	"github.com/xibodev/llm-provider-auth/tokenstore"
)

// NewCodexRefresh returns how a tokenstore.Coordinator refreshes a Codex
// credential: the RefreshFunc a Runtime's RefreshFactory returns for a
// Codex instance.
//
// The refreshed record carries the account its tokens act for, from the
// token response or else its ID token, so the Coordinator refuses a refresh
// that returns another account with tokenstore.ErrIdentityChanged, as the
// gateway always has. It carries the expiry the response reports, or else
// the access token's own. With neither it has no expiry, and it is
// refreshed when Codex rejects it. Whatever the response omits, such as a
// refresh token it did not rotate, the Coordinator keeps from the current
// record.
//
// Errors are returned unchanged. A grant the token endpoint rejected for
// good is a *codexauth.RefreshError whose Terminal method makes the
// Coordinator revoke the credential rather than retry it.
func NewCodexRefresh(config codexauth.Config) tokenstore.RefreshFunc {
	return func(ctx context.Context, current tokenstore.Record) (tokenstore.Record, error) {
		tokens, err := config.Refresh(ctx, current.RefreshToken)
		if err != nil {
			return tokenstore.Record{}, err
		}
		return codexRecord(tokens), nil
	}
}

// codexRecord maps refreshed tokens onto a record. TokenSet has already
// resolved the account, from the response or its ID token, and the expiry.
func codexRecord(tokens codexauth.TokenSet) tokenstore.Record {
	record := tokenstore.Record{
		AccessToken: tokens.AccessToken, RefreshToken: tokens.RefreshToken, IDToken: tokens.IDToken,
		TokenType: tokens.TokenType, AccountID: strings.TrimSpace(tokens.AccountID),
	}
	if tokens.ExpiresAt > 0 {
		record.Expiry = time.Unix(tokens.ExpiresAt, 0).UTC()
	}
	return record
}
