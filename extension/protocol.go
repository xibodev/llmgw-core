package extension

import (
	"encoding/base64"
	"encoding/json"
	"net/http"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/oauthflow"
)

// PathPrefix is the path under which every route of version 1 of the
// protocol lives.
const PathPrefix = "/extension/v1/"

// Headers of an operation.
const (
	// HeaderSurface names the operation's core.ModelSurface.
	HeaderSurface = "X-Surface"
	// HeaderModel names the model the operation addresses.
	HeaderModel = "X-Model"
	// HeaderCredentialToken is the credential's secret: an access token, a
	// pasted token or an API key.
	HeaderCredentialToken = "X-Credential-Token"
	// HeaderCredentialAccount is the upstream account the credential acts for.
	HeaderCredentialAccount = "X-Credential-Account-ID"
	// HeaderCredentialType is the credential's token type, such as "Bearer"
	// or core.TokenTypeAPIKey.
	HeaderCredentialType = "X-Credential-Token-Type"
	// HeaderCredentialMetadata is the base64 of the JSON object of the
	// credential's metadata.
	HeaderCredentialMetadata = "X-Credential-Metadata"
	// HeaderLosses, on an invoke or stream answer, is the base64 of the JSON
	// list of the operation's losses.
	HeaderLosses = "X-Losses"
)

// CredentialKind is what a provider needs from the product to serve it.
type CredentialKind string

const (
	// CredentialOAuth is a record the owner obtains by signing in through the
	// provider's oauth routes. ProviderInfo.HasRefresh says whether the
	// daemon can refresh it.
	CredentialOAuth CredentialKind = "oauth"
	// CredentialToken is a token the owner obtains elsewhere and pastes.
	CredentialToken CredentialKind = "token"
	// CredentialNone means the provider needs no credential.
	CredentialNone CredentialKind = "none"
)

// ProviderInfo describes one provider a daemon serves.
type ProviderInfo struct {
	// ID addresses the provider in every route.
	ID string `json:"id"`
	// Name is the provider's display name.
	Name string `json:"name"`
	// Surfaces are the surfaces the provider serves natively.
	Surfaces []core.ModelSurface `json:"surfaces"`
	// HasOAuth reports the oauth routes, and HasRefresh the refresh route.
	HasOAuth   bool `json:"has_oauth"`
	HasRefresh bool `json:"has_refresh"`
	// Credential is what the provider needs. A daemon that predates the
	// field leaves it empty; CredentialKind interprets that.
	Credential CredentialKind `json:"credential,omitempty"`
	// OAuthMethods are the sign-in methods the oauth routes accept.
	OAuthMethods []oauthflow.Method `json:"oauth_methods,omitempty"`
}

// CredentialKind returns what the provider needs. For a daemon that predates
// the Credential field, it is CredentialOAuth when the provider has oauth
// routes, and otherwise empty, because such a daemon does not say whether a
// pasted token or no credential serves the provider.
func (p ProviderInfo) CredentialKind() CredentialKind {
	switch {
	case p.Credential != "":
		return p.Credential
	case p.HasOAuth:
		return CredentialOAuth
	}
	return ""
}

// InfoResponse answers info.
type InfoResponse struct {
	// Version is the daemon's own version.
	Version   string         `json:"version"`
	Providers []ProviderInfo `json:"providers"`
}

// ModelsResponse answers models.
type ModelsResponse struct {
	Models []core.ModelInfo `json:"models"`
}

// RefreshRequest asks refresh to renew a stored credential.
type RefreshRequest struct {
	Record tokenstore.Record `json:"record"`
}

// RefreshResponse answers refresh with the renewed credential or an error.
// Terminal reports a grant the provider rejected for good.
type RefreshResponse struct {
	Record   tokenstore.Record `json:"record,omitzero"`
	Terminal bool              `json:"terminal,omitempty"`
	Error    string            `json:"error,omitempty"`
}

// OAuthStartRequest asks oauth/start to begin a sign-in.
type OAuthStartRequest struct {
	Method      oauthflow.Method  `json:"method"`
	RedirectURI string            `json:"redirect_uri,omitempty"`
	Params      map[string]string `json:"params,omitempty"`
}

// OAuthStartResponse answers oauth/start.
type OAuthStartResponse struct {
	Authorization oauthflow.Authorization `json:"authorization,omitzero"`
	Error         string                  `json:"error,omitempty"`
}

// OAuthPollRequest asks oauth/poll whether the owner approved a device
// sign-in. Flow carries its secrets, because the daemon keeps none.
type OAuthPollRequest struct {
	Flow oauthflow.Flow `json:"flow"`
}

// OAuthPollResponse answers oauth/poll.
type OAuthPollResponse struct {
	Result oauthflow.PollResult `json:"result,omitzero"`
	Error  string               `json:"error,omitempty"`
}

// OAuthExchangeRequest asks oauth/exchange to trade an authorization code
// for a credential.
type OAuthExchangeRequest struct {
	Code string         `json:"code"`
	Flow oauthflow.Flow `json:"flow"`
}

// OAuthExchangeResponse answers oauth/exchange.
type OAuthExchangeResponse struct {
	Record tokenstore.Record `json:"record,omitzero"`
	Error  string            `json:"error,omitempty"`
}

// ErrorResponse is the body of a failed operation. Code repeats the status.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail describes a failed operation.
type ErrorDetail struct {
	Message string `json:"message"`
	Code    int    `json:"code,omitempty"`
}

// SetCredentialHeaders puts credential on an operation's headers. The token
// is the credential's Token, or its APIKey when it has no token, in which
// case the token type defaults to core.TokenTypeAPIKey. A nil credential
// sets nothing.
func SetCredentialHeaders(header http.Header, credential *core.Credential) {
	if credential == nil {
		return
	}
	token, tokenType := credential.Token, credential.TokenType
	if token == "" && credential.APIKey != "" {
		token = credential.APIKey
		if tokenType == "" {
			tokenType = core.TokenTypeAPIKey
		}
	}
	if token != "" {
		header.Set(HeaderCredentialToken, token)
	}
	if credential.AccountID != "" {
		header.Set(HeaderCredentialAccount, credential.AccountID)
	}
	if tokenType != "" {
		header.Set(HeaderCredentialType, tokenType)
	}
	if len(credential.Metadata) > 0 {
		if raw, err := json.Marshal(credential.Metadata); err == nil {
			header.Set(HeaderCredentialMetadata, base64.StdEncoding.EncodeToString(raw))
		}
	}
}

// CredentialFromHeaders reads the credential SetCredentialHeaders put on an
// operation, or nil when the operation carries none. A token of type
// core.TokenTypeAPIKey becomes the credential's APIKey, as
// core.CredentialFromRecord does, and any other token its Token.
// Undecodable metadata is ignored.
func CredentialFromHeaders(header http.Header) *core.Credential {
	token := header.Get(HeaderCredentialToken)
	accountID := header.Get(HeaderCredentialAccount)
	tokenType := header.Get(HeaderCredentialType)
	var metadata map[string]string
	if encoded := header.Get(HeaderCredentialMetadata); encoded != "" {
		if raw, err := base64.StdEncoding.DecodeString(encoded); err == nil {
			_ = json.Unmarshal(raw, &metadata)
		}
	}
	if token == "" && accountID == "" && tokenType == "" && len(metadata) == 0 {
		return nil
	}
	credential := &core.Credential{AccountID: accountID, TokenType: tokenType, Metadata: metadata}
	if tokenType == core.TokenTypeAPIKey {
		credential.APIKey = token
	} else {
		credential.Token = token
	}
	return credential
}

// SetLossesHeader puts an operation's losses on its answer. No losses set
// nothing.
func SetLossesHeader(header http.Header, losses []core.Loss) {
	if len(losses) == 0 {
		return
	}
	if raw, err := json.Marshal(losses); err == nil {
		header.Set(HeaderLosses, base64.StdEncoding.EncodeToString(raw))
	}
}

// LossesFromHeader reads the losses SetLossesHeader put on an answer, or nil
// when there are none or they do not decode.
func LossesFromHeader(header http.Header) []core.Loss {
	encoded := header.Get(HeaderLosses)
	if encoded == "" {
		return nil
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil
	}
	var losses []core.Loss
	if json.Unmarshal(raw, &losses) != nil {
		return nil
	}
	return losses
}
