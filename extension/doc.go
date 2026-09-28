// Package extension is the client side of the extension protocol: the HTTP
// protocol between a product and an extension daemon, a separate process that
// serves providers the product does not carry itself.
//
// A product keeps one Client per daemon. It reaches each provider the daemon
// serves through the Client, or through a Provider, which adapts one of them
// to core.Provider so a Runtime routes to it like any other provider.
//
// # Routes
//
// Every route lives under PathPrefix. A daemon started with a shared secret
// accepts only requests that carry it as "Authorization: Bearer <secret>".
//
//	GET  info                       the providers served, as InfoResponse
//	GET  {provider}/models          the catalog for the credential, as ModelsResponse
//	POST {provider}/invoke          one non-streaming operation
//	POST {provider}/stream          one streaming operation
//	POST {provider}/refresh         RefreshRequest, answered with RefreshResponse
//	POST {provider}/oauth/start     OAuthStartRequest, answered with OAuthStartResponse
//	POST {provider}/oauth/poll      OAuthPollRequest, answered with OAuthPollResponse
//	POST {provider}/oauth/exchange  OAuthExchangeRequest, answered with OAuthExchangeResponse
//
// An operation carries the surface's request body unchanged, with the surface
// in HeaderSurface, the model in HeaderModel and the body's encoding in
// Content-Type. invoke answers with the surface's response body; HeaderLosses,
// when present, holds the operation's losses. stream answers with the
// surface's SSE records.
//
// # Models
//
// A models answer lists core.ModelInfo rows. A row may say what its model
// serves in the fields a core catalog row uses: SupportedAPIs lists the
// endpoint path of each surface the model serves, as core.SurfacePath gives
// it, such as "/v1/audio/speech", and Capabilities or LegacyCapabilities
// give the operations it supports. A daemon reports them for a model that
// serves only some of its provider's surfaces, such as a speech model of a
// provider that also serves chat. A row that reports none of them serves
// every surface of its provider: Provider.ListModels lists it with the path
// of each surface in ProviderInfo.Surfaces, and keeps a row that reports
// what it serves as the daemon sent it. The fields are optional, so a daemon
// that sends none of them still speaks version 1 of the protocol.
//
// # Credentials
//
// The daemon keeps no credentials. The product stores each one, sends it with
// every operation in the credential headers (see SetCredentialHeaders), and
// serializes refreshes: it calls refresh under its own lease, as
// tokenstore.Coordinator does with RefreshFunc, and stores the record that
// comes back. Sign-in works the same way: a completed oauth exchange or poll
// returns a record for the product to store. ProviderInfo.CredentialKind says
// which credential a provider needs.
//
// # Errors
//
// A failed route answers with an error status and a JSON body, either
// ErrorResponse or {"error": "<message>"}. Client turns a failed operation
// into a *core.ProviderError classified from the status the way core
// classifies any upstream failure, so a router retries and fails over as it
// would for any other provider. A failed refresh returns a *RefreshError,
// whose Terminal method tells tokenstore.Coordinator whether to revoke the
// credential.
//
// # Security
//
// Anyone who can reach a daemon can use its keyless providers, and every
// credential a product sends passes through it. Bind a daemon to the loopback
// interface and start it with a secret; Client sends the secret on every
// request.
package extension
