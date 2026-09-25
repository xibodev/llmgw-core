package core

import (
	"context"
	"errors"
	"maps"
	"strings"
	"sync"

	"github.com/xibodev/llm-provider-auth/tokenstore"
)

// TokenTypeAPIKey marks a token-store record that holds a static API key
// rather than an OAuth access token.
const TokenTypeAPIKey = "api_key"

// Metadata keys under which a record carries what its OAuth grant needs
// later, the way the gateway's credential store keeps them. A refresh reads
// them so each credential refreshes with the client it was granted to, not
// whichever client is configured now.
const (
	// CredentialMetadataOAuthProfile names how the owner signed in, such as
	// a device or a browser login. The provider defines the values.
	CredentialMetadataOAuthProfile = "oauth_profile"
	// CredentialMetadataOAuthClientID is the OAuth client the grant belongs to.
	CredentialMetadataOAuthClientID = "oauth_client_id"
	// CredentialMetadataOAuthClientMode is how that client authenticates,
	// such as public or confidential.
	CredentialMetadataOAuthClientMode = "oauth_client_mode"
	// CredentialMetadataOAuthRedirectURI is the redirect URI of a browser
	// login.
	CredentialMetadataOAuthRedirectURI = "oauth_redirect_uri"
	// CredentialMetadataOAuthClientSecret is a confidential client's secret.
	CredentialMetadataOAuthClientSecret = "oauth_client_secret"
	// CredentialMetadataAccountLabel is a label for the account, such as an
	// email address, for display.
	CredentialMetadataAccountLabel = "account_label"
)

// ErrNoCredential reports that no credential resolves for a caller and a
// provider instance.
var ErrNoCredential = errors.New("core: no credential resolves for this caller and instance")

// CredentialStore holds the credentials a Runtime uses.
//
// It is the llm-provider-auth token store plus Resolve. The store's Lease must
// exclude every process that shares it: a file store holds an OS advisory lock
// on a separate lock file, and a database store holds a lease row with a
// holder and an expiry, never a transaction held across a refresh.
//
// Resolve names the credential for a caller and a provider instance. Which
// credential applies is product policy. Keeping it fresh is not: the Runtime
// refreshes OAuth credentials through tokenstore.Coordinator, which never lets
// a refresh change the account.
type CredentialStore interface {
	tokenstore.Store
	// Resolve returns the key of the credential that serves caller on
	// instance, or ErrNoCredential.
	Resolve(ctx context.Context, caller Caller, instance string) (key string, err error)
}

// APIKeyRecord returns the token-store record of a static API key. It has no
// expiry, so tokenstore.Coordinator never refreshes it.
func APIKeyRecord(apiKey string) tokenstore.Record {
	return tokenstore.Record{AccessToken: apiKey, TokenType: TokenTypeAPIKey}
}

// CredentialFromRecord converts a stored record into the credential a provider
// receives. The access token becomes APIKey for an API-key record and Token
// otherwise, and the key becomes ConnectionID. AccountID, TokenType and
// Metadata carry over, so a provider can address the account, tell what kind
// of credential it holds and read its provider-specific values. Metadata is
// copied, so the credential and the record never share a map. The refresh
// token stays behind, because only tokenstore.Coordinator refreshes. Record
// revisions are opaque strings, so CredentialRevision stays zero and evidence
// carries the revision instead.
func CredentialFromRecord(key string, record tokenstore.Record) *Credential {
	credential := &Credential{
		ConnectionID: key,
		AccountID:    record.AccountID,
		TokenType:    record.TokenType,
		Metadata:     maps.Clone(record.Metadata),
	}
	if record.TokenType == TokenTypeAPIKey {
		credential.APIKey = record.AccessToken
	} else {
		credential.Token = record.AccessToken
	}
	return credential
}

// MemoryCredentialStore is the in-memory reference CredentialStore:
// tokenstore.Memory plus a resolution table. Resolve prefers a binding for the
// exact caller and falls back to the instance's shared binding, the way a
// product prefers a user's own credential over a system one.
type MemoryCredentialStore struct {
	*tokenstore.Memory

	mu       sync.Mutex
	bindings map[credentialBinding]string
}

type credentialBinding struct {
	callerID string
	kind     CallerKind
	instance string
}

// NewMemoryCredentialStore returns an empty store.
func NewMemoryCredentialStore() *MemoryCredentialStore {
	return &MemoryCredentialStore{Memory: tokenstore.NewMemory(), bindings: map[credentialBinding]string{}}
}

// Bind makes key the credential of caller on instance.
func (s *MemoryCredentialStore) Bind(caller Caller, instance, key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.bindings[credentialBinding{callerID: caller.ID, kind: caller.Kind, instance: strings.TrimSpace(instance)}] = key
}

// BindShared makes key the credential of every caller on instance that has
// no binding of its own.
func (s *MemoryCredentialStore) BindShared(instance, key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.bindings[credentialBinding{instance: strings.TrimSpace(instance)}] = key
}

// Resolve implements CredentialStore.
func (s *MemoryCredentialStore) Resolve(ctx context.Context, caller Caller, instance string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	instance = strings.TrimSpace(instance)
	s.mu.Lock()
	defer s.mu.Unlock()
	if key, ok := s.bindings[credentialBinding{callerID: caller.ID, kind: caller.Kind, instance: instance}]; ok && caller.ID != "" {
		return key, nil
	}
	if key, ok := s.bindings[credentialBinding{instance: instance}]; ok {
		return key, nil
	}
	return "", ErrNoCredential
}
