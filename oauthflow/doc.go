// Package oauthflow runs server-side OAuth flows for products that sign
// users in to LLM providers: browser authorization codes with PKCE, device
// authorization, and codes the owner pastes back.
//
// A FlowStore keeps each flow bound to the core.Caller that started it and
// consumes it at most once. MemoryFlowStore is the in-memory reference, and
// can cap each caller's pending flows as the gateway does; run
// oauthflowtest.Run against every implementation. A Service runs the flows
// over a store, provider drivers and a core.CredentialStore:
//
//   - Start begins a flow and returns its View.
//   - Get reads it.
//   - Poll advances a device flow, never faster than its interval.
//   - Complete finishes a browser or manual flow with a code the owner
//     pasted, or one from a callback the product routed itself.
//   - Callback finishes a browser flow from the provider's redirect, found
//     by its OAuth state because the redirect carries no session.
//
// Verifiers, OAuth states, device codes, redirect URIs and tokens stay on
// the server. A View never holds them, and the state appears only inside the
// authorization URL shown to the owner. Completion saves the credential
// through core.CredentialStore under a key the product chooses.
//
// # Gateway routes
//
// The gateway's routes in go/internal/api/oauth.go map onto the Service
// without changing their requests or responses. The caller is the principal
// the route acts for: the SSO user, or the principal an administrator's
// route names.
//
//	start     POST .../connections/{provider_id}/oauth/start
//	          Start(ctx, caller, providerID, method,
//	              WithRedirectURI(oauthCallbackURL(r, registryID)),
//	              WithParams(connection name, source, client settings))
//	          device_code: MethodDevice (Codex, Copilot)
//	          browser: MethodBrowser (Antigravity), MethodManual (Codex)
//	          profile consumer_manual: MethodManual (Antigravity)
//	          flow_id and device_code both carry View.ID
//	get       POST .../oauth/poll with a browser flow_id
//	          Get: pending, or authorized once complete
//	poll      POST .../oauth/poll with device_code
//	          Poll: ErrSlowDown is slow_down, ErrAccessDenied is denied,
//	          ErrFlowExpired and ErrFlowNotFound are expired
//	complete  POST .../oauth/complete {flow_id, authorization_response}
//	          Complete(ctx, caller, flowID, CompleteInput{Code, State}),
//	          after manualAuthorizationCode splits the pasted value
//	callback  GET /oauth/callback/{provider_id}?code&state
//	          Callback(ctx, CompleteInput{Code, State, Error,
//	              RedirectURI: oauthCallbackURL(r, provider_id)})
//
// A complete View carries CredentialKey, from which the gateway loads the
// connection it returns. Refresh and revoke are not flows and stay on the
// credential store.
//
// # Facet Studio routes
//
// Facet Studio's routes in web/backend/api/oauth.go map the same way with
// core.LocalCaller(). Its statuses are pending, success (StatusComplete),
// error (StatusFailed) and expired.
//
//	start     POST /api/oauth/login {provider, method}
//	          Start(ctx, core.LocalCaller(), provider, MethodDevice or
//	              MethodBrowser, WithRedirectURI(buildOAuthRedirectURI(r)))
//	          api_key and setup_token logins save directly; they are not flows
//	get       GET /api/oauth/flows/{id}: Get
//	poll      POST /api/oauth/flows/{id}/poll: Poll; ErrSlowDown stays pending
//	complete  POST /api/oauth/flows/{id}/complete {code}
//	          Complete(ctx, core.LocalCaller(), id, CompleteInput{Code, State})
//	callback  GET /oauth/callback?state&code&error
//	          Callback(ctx, CompleteInput{Code, State, Error})
//
// # Drivers
//
// A product configures one driver per instance and method. BrowserPKCE
// serves any provider that llm-provider-auth's browseroauth can drive.
// Codex's device and code flows, the Copilot device flow and Antigravity's
// browser flow wrap their llm-provider-auth packages in a few lines each:
// the driver keeps what it needs later, such as a client id or a user code,
// in Secrets.DriverData.
package oauthflow
