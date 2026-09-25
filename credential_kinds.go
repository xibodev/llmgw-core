package core

// Credential kinds whose material is neither a static API key nor an OAuth
// access token.
//
// A Credential of such a kind names it in TokenType, carries the kind's
// material in Token, never in APIKey, and carries its extras in Metadata
// under the keys the kind documents. A token-store record of the same
// TokenType, holding the material as its access token without an expiry,
// becomes such a credential through CredentialFromRecord.
//
// The material is not a bearer token, and only a provider that serves the
// kind reads it. A product therefore binds such a credential only to an
// instance whose provider serves its kind.
const (
	// TokenTypeGCPServiceAccount marks a Google Cloud service-account key.
	// Token is the key's JSON document as Google issues it, which a
	// provider exchanges for short-lived access tokens and never sends.
	// Metadata may name, under CredentialMetadataProjectID, the project
	// calls are billed to when it is not the key's own. The spelling is
	// llm-provider-auth's gcp.CredentialKind.
	TokenTypeGCPServiceAccount = "gcp_service_account"
	// TokenTypeAnthropicSetupToken marks an Anthropic setup token, the
	// long-lived OAuth token of a Claude subscription. Token is the setup
	// token, which Anthropic takes as a bearer with its OAuth beta marker
	// rather than as an x-api-key; llm-provider-auth's anthropic package
	// validates it and applies those headers.
	TokenTypeAnthropicSetupToken = "anthropic_setup_token"
)
