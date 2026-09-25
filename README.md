# llmgw-core

Headless, embeddable LLM routing engine and resilient multi-provider proxy for Go.

## Highlights

- **Headless & Zero Database:** Pure Go `http.Handler`. Mounts anywhere (`net/http`, Chi, Gin, Fiber, Echo) without requiring SQLite, PostgreSQL, or any storage engine.
- **Bi-Directional Wire Translation:** Full Anthropic Messages API (`/v1/messages`) support translated dynamically to OpenAI backends (and SSE streaming) via `llm-translate`.
- **Resilient Multi-Provider Routing:** Route aliases, ordered failover chains, rate-limit 429 detection, and circuit breaking.
- **Pluggable Extension Hooks:**
  - `Authenticator`: Custom token, session, or API key validation.
  - `PolicyGate`: RBAC, project-level, or model-level permission gates.
  - `CredentialResolver`: Dynamic or multi-tenant credential overrides (BYOC).
  - `UsageHook`: Telemetry, cost tracking, and audit logging.
- **Shared Provider Contracts:** Typed connection/auth kinds, catalog and completion evidence, health classification, exact target publication policy, and a persistence-agnostic `ProviderConnector` interface.

## Installation

```bash
go get github.com/xibodev/llmgw-core
```

### Dependencies

- **llm-provider-auth v0.6.0.** Its `codex` package no longer supplies a
  client version, so `providers.NewCodex` and `NewCodexProvider` require
  `ClientVersion`, the version of the product making the call, and return an
  error when it is blank. The providers send it to the Codex catalog as
  `client_version`. `ResponsesURL` and `ModelsURL` still default to the
  canonical Codex endpoints.
- **llm-translate v0.3.0.** Conversions report the vendor fields inside
  messages that the target surface cannot carry, and the translation adapter
  enforces those losses; see [Loss policy](#loss-policy). The Anthropic and
  Codex providers have no loss report, so they keep serving the histories
  they served before. The Anthropic provider carries the `cache_control` of
  Chat text parts into the Messages request, system prompt included.

## Quick Start (Embedded in 15 Lines)

```go
package main

import (
    "net/http"
    "github.com/xibodev/llmgw-core"
    "github.com/xibodev/llmgw-core/providers"
)

func main() {
    engine := core.NewEngine(core.Config{
        Routes: map[string]core.RouteConfig{
            "smart": {
                Targets: []core.Target{
                    {Provider: "groq", Model: "llama-3.3-70b-versatile"},
                    {Provider: "openai", Model: "gpt-4o"},
                },
            },
        },
    })

    // Register backends
    engine.RegisterProvider("openai", providers.NewOpenAIProvider("openai", "https://api.openai.com/v1", os.Getenv("OPENAI_API_KEY"), nil))
    engine.RegisterProvider("groq", providers.NewOpenAIProvider("groq", "https://api.groq.com/openai/v1", os.Getenv("GROQ_API_KEY"), nil))

    // Mount on standard HTTP server
    http.Handle("/v1/", engine)
    http.ListenAndServe(":8080", nil)
}
```

## Provider registry

`providers.DefaultRegistry()` is the reviewed, validated manifest of curated
provider integrations. Core owns each entry's wire and safety facts: runtime,
protocol, auth methods, connection scope, risk classification, and anonymous
automation. A product layers its own curation on top with an overlay, and the
result is validated like a manifest:

```go
overlay, err := providers.DecodeOverlay(overlayJSON) // strict: unknown fields fail
registry, err := providers.DefaultRegistry().WithOverlay(overlay, providers.ValidationOptions{
    RuntimeTypes: []string{"openai_compatible", "anthropic"}, // what this product can execute
})
entry, ok := registry.Lookup("gpt") // ids and aliases, case-insensitive
```

An overlay can:

- **Override** presentation, priority, and curation fields of existing entries:
  label, description, categories, icon, domain, docs URL, priority,
  availability, common models, onboarding fields, additional aliases, and
  product-private `extensions`.
- **Add** product-only entries under their own ids.
- **Remove** entries the product does not offer.

An override cannot change wire or safety facts. A product that needs different
wire behavior adds its own entry instead. `Entries()` orders by priority, then
by manifest order, so an overlay without priorities keeps the manifest order.

Anonymous providers use one mechanism:

- `Registry.AnonymousProfiles` lists the entries curated for anonymous
  automation.
- `AdmitAnonymousModel` applies each provider's reviewed free-model rules to a
  raw catalog row. It fails closed: unknown providers are never admitted, and
  OpenCode Zen is admitted only from its verified metadata.
- `SelectVerificationModel` picks the model to probe.
- `DiscoverAnonymousModels` combines them.

Whether to enroll an anonymous provider remains the product's policy.

## Provider contract

The wire-level contract shared by products and transports:

- **`Caller`** identifies who a request acts for. `LocalCaller()` is the caller
  of a single-user application.
- **`Provider`** serves a `Request{Surface, Model, Body, ContentType,
  Credential}` for the surfaces it lists in `NativeSurfaces`.
  - It returns a `Response{Body, ContentType, Losses}`, or a `StreamIter` of
    complete SSE frames.
  - Bodies are bytes: JSON for the chat surfaces, multipart form data for
    speech-to-text, and binary for speech, images and video.
  - A request for any other surface fails with a `*SurfaceError`, and a
    translation adapter serves it instead.
- **Errors** classify themselves. `ClassifyError(err).Disposition()` says whether
  to retry, fail over, or stop. `ProviderError` carries a message that is safe to
  show and a cause that is never rendered.
- **`LossPolicy`** decides which translation and adaptation losses a product
  accepts, by field-path glob, class and severity. Allowed losses are still
  reported, in `Response.Losses` and through a stream's `LossReporter`.

Products supply three stores, each with an in-memory reference implementation:

- **`CredentialStore`** is the llm-provider-auth token store plus
  `Resolve(ctx, caller, instance)`, which returns the key of the credential that
  serves a caller. Which credential applies is product policy. The Runtime keeps
  it fresh through `tokenstore.Coordinator`. `APIKeyRecord` stores a static key
  that never refreshes. Reference: `NewMemoryCredentialStore`.
  - A provider receives `CredentialFromRecord(key, record)`: the access token
    as `APIKey` or `Token`, plus the record's `AccountID`, `TokenType` and a
    copy of its `Metadata`, such as a project ID.
  - A `Credential` prints and logs whether each secret is set, never its
    value. `Metadata` may hold secrets, so it is never serialized.
- **`CatalogStore`** keeps discovered catalogs, with a new revision on every save.
  `Save` must be atomic for readers in any process. Run `catalogtest.Run`
  against every implementation. Reference: `NewMemoryCatalogStore`.
  - A `ConditionalCatalogStore` adds `SaveIf(ctx, key, evidence, expected)`
    and `Delete`, so a discovery fences out an invalidation in any process.
    `Delete` leaves the key a new revision, which `Load` returns with
    `ErrCatalogNotFound`. `catalogtest.Run` checks both when present.
- **`EvidenceSink`** receives `AccountEvidence`: which credential served which
  operation, and the classified outcome. Reference: `MemoryEvidenceSink`.

### M5 foundations

The API the remaining transports build on. All of it is additive: no
existing vertical sends, returns or classifies anything differently.

```go
// package core
type ModelInfo struct {
    // ... the fields above, unchanged ...
    DisplayName        string         `json:"display_name,omitempty"`
    Vendor             string         `json:"vendor,omitempty"`
    Free               bool           `json:"free,omitempty"`
    LegacyCapabilities map[string]any `json:"legacy_capabilities,omitempty"`
}
func AdaptModelCapabilities(legacy map[string]any, surfaces []string, discoveredAt, verifiedAt time.Time) *ModelCapabilities
func InferCapabilities(model ModelInfo, discoveredAt, verifiedAt time.Time) *ModelCapabilities

type WirePreserver interface{ PreservesWire(model string, surface ModelSurface) bool }
func PreservesWire(provider Provider, model string, surface ModelSurface) bool

type TokenCounter interface {
    CountTokens(ctx context.Context, request TokenCountRequest) (TokenCount, error)
}
type TokenCountRequest struct { Request; Header http.Header }
type TokenCount struct{ InputTokens int64 }
func CountTokens(ctx context.Context, provider Provider, request TokenCountRequest) (TokenCount, error)
var ErrTokenCountUnsupported error

const TokenTypeGCPServiceAccount = "gcp_service_account"
const TokenTypeAnthropicSetupToken = "anthropic_setup_token"
```

- **Rows.** A gateway catalog row maps field for field: `Label` is
  `DisplayName`, `Capabilities` is `LegacyCapabilities`, `TypedCapabilities`
  is `Capabilities`, and `SupportedSurfaces` is `SupportedAPIs`.
- **Capabilities.** `AdaptModelCapabilities` is the gateway's inference,
  ported verbatim: inferred, medium confidence, and no expiry.
  `InferCapabilities` applies it to a row and ignores the row's own
  `Capabilities`.
- **Preservation.** `PreservesWire` is true when every layer serves the
  surface natively and the nearest `WirePreserver` says so. Serving Chat
  natively by converting it, as Codex does, is not preserving it, and a
  response transport label reads only this declaration. A decorator exposes
  what it wraps through `Unwrap() Provider`, as `translation.Adapter` does.
- **Token counts.** `CountTokens` finds the nearest `TokenCounter` the same
  way. `ErrTokenCountUnsupported` tells a product to estimate instead.
- **Credential kinds.** A credential whose `TokenType` names one carries the
  material in `Token` and its extras, such as `CredentialMetadataProjectID`,
  in `Metadata`. The material is no bearer token, so bind it only to a
  provider that serves its kind.
- **Wire kit.** Package `providers` shares the gateway's transport helpers
  with the verticals it holds:
  - `readInvocationResponseBody` reads an answer within 64 MiB, and
    `decodeCatalogResponse` a catalog within 8 MiB, checking every row.
  - `sseRecordReader` reads SSE records within 4 MiB with their bytes as
    sent, `sseFrameStream` relays them, and `readBoundedLine` bounds NDJSON.
  - `httpStatusFailure` reports a refused answer classified by the gateway's
    status set, with `Retry-After`. Its message never quotes the upstream;
    the cause holds the redacted diagnostic.
  - `extractError`, `retryAfterDelay`, `transportFailure`, `streamFailure`
    and `catalogFailure` complete it. A catalog failure's `*CatalogError`
    has a `CatalogCode*` code: the gateway's, without its `catalog_` prefix.

## Runtime

`runtime.Runtime` is the one value that owns a product's provider state: the
registry, the provider cache, the refresh coordinators, the catalog cache and
health. The core packages keep no package state and read no environment, and
an architecture test enforces both.

```go
rt, err := runtime.New(runtime.Options[Settings]{
    Settings:    settingsSource, // Snapshot() (Settings, generation)
    Providers:   buildProvider,  // func(Settings, instance) (core.Provider, error)
    Credentials: credentialStore,
    Refresh:     oauthRefresh,   // per instance; nil means credentials never refresh
    Catalogs:    catalogStore,
    Evidence:    evidenceSink,
    Registry:    registry,
})
response, err := rt.Invoke(ctx, caller, "openai", request)
```

The Runtime works with the product's own settings type:

- It rebuilds providers and coordinators when the settings generation changes.
- It resolves each caller's credential and refreshes an OAuth credential the
  upstream rejects with 401, once, before replaying. Static API keys are never
  replayed.
- It serves stored catalogs while they are fresh, sharing them across processes
  through the `CatalogStore`.
- Only upstream outcomes (an HTTP status or a transport failure) change an
  instance's health.

`translation.Adapter{Provider, Policy}` serves surfaces a provider lacks by
translating through llm-translate:

- Messages over Chat Completions, streaming included.
- Chat Completions over Responses.
- Responses over Chat Completions.

### Catalogs

`catalog.Service` keeps catalogs as the gateway does, as instance state over
a `CatalogStore`. A Runtime builds one from `Catalogs` and `CatalogTTL` that
behaves as `ListModels` always has; pass `Options.CatalogService` to choose:

```go
catalogs := catalog.New(catalog.Options{
    Store: store, TTL: time.Hour, KeepStale: true, SchemaVersion: 6,
    InstanceTTL: func(instance string) time.Duration { return 0 }, // 0: use TTL
})
```

- **Reads.** `Read` serves a catalog younger than the TTL, or discovers it and
  returns what is stored afterwards. `Cached` and `CachedLookup` never discover.
  With `KeepStale`, a failed discovery keeps the stored rows, marked stale.
- **Fences.** `Invalidate(ctx, key, catalog.Hard)` forgets a catalog and fences
  every discovery in flight, so a revoked credential's rows never come back.
  `catalog.Soft` forgets without fencing the operation that caused it.
- **Diagnostics** report synced, not_synced, empty or error, stale and
  from_cache, and never quote an error.
- **Schema.** `CatalogEvidence.SchemaVersion` stamps what a service stores; rows
  stamped otherwise are neither served nor kept.

The Runtime adds `ReadCatalog` (diagnostics), `CachedCatalog`, `CachedModel`
for catalog hooks such as `ZenConfig.Models`, and `CatalogService`.

### Transport planning

The gateway's transport-mode decisions, as pure helpers over `PlanTransport`:

- `TransportEvidence{Row, RefreshedAt}.Capabilities()` gives a row the catalog's
  freshness, an hour by default, without upgrading its confidence.
- `TransportInterfaces` offers the chat surfaces a row lists, native where
  `PreservesWireFor` says the provider preserves the wire for the credential.
  A `CredentialWirePreserver` answers per credential: anonymous Zen preserves
  nothing. Codex preserves Responses, Copilot Chat and listed Responses.
- `ParseSurfacePath`, `ParseTransportRequirement`, `ListingSurfaces` and
  `ResponseTransportMode` port the path and header parsing, the model list's
  native and emulated surfaces, and the response label.

`Runtime.Transparent` sends the body as is, once, only when a transparent plan
over the caller's stored catalog confirms it, and refuses a stream after that.
A refusal is a `*core.TransportRejectError`; each product words it.
`Runtime.TransportMode` labels a response.

### Loss policy

Request losses are checked against the `LossPolicy` before the provider is
called, and the response losses of `Invoke` once it answers. A stream's
response losses are reported but not enforced, because its frames are
already delivered. All losses, including the provider's own and those the
policy allows, are returned in `Response.Losses` or through the stream's
`LossReporter`.

Gemini returns a thought signature with each tool call and needs it back on
the next turn. No other surface can carry one, so llm-translate reports each
dropped signature as a material loss at its path, such as
`messages.1.tool_calls.0.extra_content.google.thought_signature`. The
default policy rejects unmatched material losses, so the adapter refuses:

- a Chat request whose history carries signatures, served over Responses;
- a Chat response that carries them, converted for a Messages or Responses
  client, after the provider has answered.

The refusal is a `*LossPolicyError`, which permits failover to a target that
serves the surface natively. Under the same policy, a Messages stream over
Chat Completions still serves signed tool calls and only reports the loss.

A product that serves Gemini through the adapter should consider allowing
the loss, which is still reported:

```go
policy := core.LossPolicy{Rules: []core.LossRule{
    {Path: "**.thought_signature", Class: translate.LossDropped, Action: core.LossAllow},
}}
```

Allowing it loses nothing on a request to a model other than Gemini: that
model cannot use the signature, and the caller's history keeps it. A client
that receives a translated Gemini response, though, never sees the
signature, so its next turn reaches Gemini without it.

Dropped `reasoning_details`, `reasoning_content` and `cache_control`, and
dropped reasoning items or thinking blocks, reported at `<path>.reasoning`,
are advisory: the default policy allows and reports them, and a product
that depends on them rejects them by path.

## Codex

`providers.Codex` is a whole provider vertical on the provider contract:
the Codex Responses transport, the catalog and what its rows mean, and,
with `NewCodexRefresh`, the credential refresh. None of it is left for a
product to normalize.

```go
rt, err := runtime.New(runtime.Options[Settings]{
    Settings: settingsSource,
    Providers: func(s Settings, instance string) (core.Provider, error) {
        codex, err := providers.NewCodex(providers.CodexConfig{
            Instructions:  s.CodexInstructions,
            ClientVersion: s.CodexClientVersion, // names the product; required
        })
        if err != nil {
            return nil, err
        }
        return translation.Adapter{Provider: codex}, nil // Messages over Chat
    },
    Refresh: func(s Settings, instance string) tokenstore.RefreshFunc {
        return providers.NewCodexRefresh(codexauth.Config{ClientID: s.CodexClientID})
    },
    Credentials: credentialStore,
})
```

- **Credential.** Each request authenticates with its own credential: the
  token is the bearer and `AccountID` becomes the `ChatGPT-Account-ID`
  header. A credential without a token, such as an API key, is a
  configuration error, and nothing is sent.
- **Surfaces.** Responses and Chat Completions are native, and both
  stream. A shorthand string `input` is sent as the one user message it
  abbreviates. Chat is converted to Responses with the gateway's Chat field
  policy: fields Codex cannot carry, such as `max_tokens` and `temperature`,
  are dropped and reported as advisory losses, and a field that would change
  the answer's structure is refused. `CodexConfig.ResponsesOnly` leaves Chat
  to a `translation.Adapter` and the product's loss policy instead.
- **Catalog.** `ListModels` returns only rows supported in the API and
  listed. A row that omits `supported_endpoints`, as current catalogs do,
  gets `/responses`. The parser was verified against the catalog of
  `providers.CodexVerifiedClientVersion`; it is not a default, and
  `ClientVersion` must still be set.
- **Refresh.** A rejected token fails with status 401, so the Runtime
  refreshes once and replays. Each credential refreshes with the OAuth client
  its metadata names (`core.CredentialMetadataOAuthClientID`, falling back
  to the configured client), and a browser login as that public PKCE client.
  The refreshed record keeps the account, taken from the token response or
  its ID token, so a refresh that returns another account is refused. A
  grant the token endpoint rejects for good revokes the credential.
- **Errors** are `*core.ProviderError`, with the transport's error as the
  cause.

`CodexProvider`, which reads tokens from a session source, is deprecated.
It shares the transport, so both send identical requests.

## Antigravity

`providers.Antigravity` is the Google Antigravity vertical on the provider
contract, over the undocumented Cloud Code Assist API: Chat, image
generation, the catalog and what its rows mean, the project each credential
uses, and, with `NewAntigravityRefresh`, the credential refresh.

```go
rt, err := runtime.New(runtime.Options[Settings]{
    Settings: settingsSource,
    Providers: func(s Settings, instance string) (core.Provider, error) {
        antigravity, err := providers.NewAntigravity(providers.AntigravityConfig{
            ProjectResolved: storeProject, // optional: keep a discovered project
        })
        if err != nil {
            return nil, err
        }
        return translation.Adapter{Provider: antigravity}, nil // Messages over Chat
    },
    Refresh: func(s Settings, instance string) tokenstore.RefreshFunc {
        return providers.NewAntigravityRefresh(antigravityauth.Config{
            ClientID: s.ClientID, ClientSecret: s.ClientSecret,
            ClientAuthMode: antigravityauth.ClientAuthModeClientSecretPost,
        })
    },
    Credentials: credentialStore,
})
```

- **Credential.** The token is the bearer, and the project is the one the
  credential's metadata names under `core.CredentialMetadataProjectID`.
  When it names none, Antigravity discovers it and, once the operation
  succeeds, hands it to `ProjectResolved`, for the product to store with a
  revision-fenced write. A credential without a token is a configuration
  error, and nothing is sent.
- **Surfaces.** Chat Completions only, sent exactly as the gateway sends
  it: the messages, `max_tokens`, `temperature` and `tools` are mapped to
  Gemini, and every other field is dropped and reported as a loss, a
  material one when it changes the answer's structure. Antigravity does not
  stream: it reads Cloud Code Assist's stream to its end, so `Stream`
  refuses and permits failover. `GenerateImages` implements
  `core.ImageGenerator`, one image from a model both catalog rosters name.
- **Catalog.** Every model in the root roster serves Chat, and one the
  image roster names also serves image generation. No row streams, and what
  the catalog omits stays unknown.
- **Refresh.** A rejected token fails with status 401, so the Runtime
  refreshes once and replays. Each credential refreshes with the OAuth
  client its `core.CredentialMetadataOAuthProfile` selects, as the gateway
  selects it, falling back to the configured client. It keeps its account
  and project, unless the new token discovers another project. A grant the
  token endpoint rejects for good fails without revoking the credential, as
  in the gateway; `providers.AntigravityRevokeOnTerminal()` revokes it.
- **Errors** are `*core.ProviderError`, with the transport's
  `*core.ProviderOperationError` as the cause.

`ExperimentalAntigravityProvider`, which reads tokens from a token source,
is deprecated. It shares the transport, so both send identical requests.

## OpenCode Zen

`providers.Zen` is the OpenCode Zen vertical: request shaping, anonymous
access and its invocation identity, both streams, and the catalog and what
its rows mean. Requests are shaped byte for byte as the gateway shapes them.

```go
provider, err := providers.NewZen(providers.ZenConfig{
    // The product's catalog row for a model, as ListModels returned it.
    // Zen serves the model on the endpoints the row lists.
    Models: catalog.Lookup,
})
if err != nil {
    return nil, err
}
return translation.Adapter{Provider: provider}, nil // Messages over Chat
```

- **Credential.** Optional. An API key is the bearer. Without one, or with
  `free`, `none` or `public`, Zen sends the anonymous identity: the public
  bearer and the OpenCode CLI's request admission. A model the catalog knows
  anonymous access does not admit is refused before anything is sent. Every
  request carries the `zen.InvocationIdentity` in its context, or a new one;
  create one per inbound request so retries reuse it.
- **Surfaces.** Chat Completions for every model, and Responses for a model
  whose row lists `/responses`. Chat for a Responses-only model is converted
  to Responses as the gateway converts it. A model the catalog lacks follows
  the gateway's cold-catalog contract: Muse models serve Responses only.
  Chat fields the gateway does not forward are dropped as advisory losses.
- **Streams.** Native streams pass Zen's records through byte for byte. A
  Responses stream keeps typed JSON events up to the terminal one and fails
  if that event never arrives.
- **Catalog.** Keyed, `ListModels` reads Zen's `/models` as the gateway does.
  Anonymous, it lists the models `zen.Normalize` admits from `models.dev` and
  the live catalog, each tagged `providers.ModelTagFree` with its endpoint.
- **Errors** are `*core.ProviderError` with the upstream status and
  `Retry-After`; a catalog failure's `*providers.CatalogError` code is the
  gateway's catalog error code without its `catalog_` prefix.

The `zen.Client` discovery and completion methods are deprecated. They share
the normalizer, admission and identity code, and behave as before.

## GitHub Copilot

`providers.Copilot` is the GitHub Copilot vertical on the provider
contract: the session exchange, request shaping, the catalog and what its
rows mean. It is configured through llm-provider-auth's `copilot` client.

```go
Providers: func(s Settings, instance string) (core.Provider, error) {
    copilot, err := providers.NewCopilot(providers.CopilotConfig{
        Auth:                s.CopilotAuth,    // *copilot.Client with AllowProxy on
        EditorPluginVersion: "my-product/1.0", // names the product; required
        UserAgent:           "MyProductChat/1.0",
    })
    if err != nil {
        return nil, err
    }
    return translation.Adapter{Provider: copilot}, nil
},
```

- **Credential.** A request's credential token is the caller's GitHub
  OAuth token. `Auth` exchanges it for a session, which is kept in memory
  per credential until it nears expiry, and concurrent requests share one
  exchange. A request without a credential uses the product's own token
  through `Auth`, configured, cached or from the gh CLI; `Auth` caches that
  session on disk when it has a `CacheDir`. A credential without a token is
  a configuration error, and nothing is sent.
- **Requests** are shaped byte for byte as the gateway's Copilot transport
  shapes them: the Chat fields it forwards, the OpenAI-compatible
  transport's (`temperature`, `top_p`, `max_tokens`, `max_completion_tokens`,
  `stop`, `tools`, `tool_choice`, `reasoning_effort`, `stream_options`,
  `metadata`, `parallel_tool_calls` and `thinking`), sent to the API base the
  session names with the gateway's editor headers. Chat served over
  Responses converts only the fields the gateway's Chat facade forwards.
  Other fields are dropped and reported as losses.
- **Surfaces.** Chat Completions is native for every model and streams.
  Responses is native for a model whose catalog row lists it, so list the
  models first. Chat for a row that lists only Responses is served over
  Responses, and a model that lists reasoning efforts takes `max_tokens` as
  `max_completion_tokens`, unless `DisableAdaptation` or a request's
  `force_api_support` says otherwise. The Adapter serves Messages, and
  Responses for the other models, over Chat.
- **Catalog.** `ListModels` returns every row with the capabilities the
  gateway derives: surfaces from `supported_endpoints`, and vision, tools,
  structured output, streaming, reasoning and limits from `capabilities`.
  What a row omits stays unknown.
- **401.** A session Copilot rejects is replaced once and the request
  replayed. A second 401, or GitHub rejecting the OAuth token, fails with
  status 401, which the Runtime refreshes or reports.
- **Errors** are `*core.ProviderError`. The auth client's error stays the
  cause, so `errors.Is(err, copilot.ErrOAuthTokenRejected)` lets a product
  add guidance that names its own sign-in.

## OpenAI-compatible and Bedrock

`providers.OpenAICompatible` is the gateway's generic OpenAI transport:
its openai_compatible, openai and litellm instances, the anonymous
catalogs the registry curates, and Amazon Bedrock. Requests are shaped
byte for byte as the gateway shapes them.

```go
provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{
    BaseURL:    s.BaseURL,     // required, such as https://api.openai.com/v1
    RegistryID: s.RegistryID,  // "openai", "llm7", "pollinations", ...
    Models:     catalog.Lookup, // the product's catalog row for a model
})
// Or: providers.NewBedrock(s.Region, s.BaseURL, providers.OpenAICompatibleConfig{Models: catalog.Lookup})
if err != nil {
    return nil, err
}
return translation.Adapter{Provider: provider}, nil // Messages over Chat
```

- **Credential.** Optional. The API key, or else the token, is the
  bearer, without any `Bearer ` prefix; no key, `free` or `none` sends no
  Authorization. The credential's headers are sent too. A service-account
  or setup-token credential is refused before anything is sent.
- **Surfaces.** Chat Completions for every model, and Responses for a
  model whose row lists it or of the `openai` entry. Both are declared
  through `core.WirePreserver`, so a product labels them native. The
  Adapter serves Responses over Chat for the other models, but does not
  stream it.
- **Requests.** Chat carries the fields the gateway's transport forwards,
  and drops the others as advisory losses unless `ForwardAllFields` is
  set. Responses is forwarded as sent, with the request's model and
  stream flag. `ForceAPISupport`, or a request's `force_api_support`,
  serves Chat over Responses for a model whose row lists only Responses,
  retrying a 400 that names `temperature` or `top_p` once without it,
  and sends `max_tokens` to a reasoning model as `max_completion_tokens`.
- **Catalog.** `/models` rows keep their vendor, display name and
  endpoints; a capabilities block is distilled into `LegacyCapabilities`,
  and `Capabilities` is what `core.InferCapabilities` derives. Without a
  key, kilo_code, llm7, ovh_ai_endpoints and pollinations list only the
  models `AdmitAnonymousModel` admits, marked `Free`. Pollinations takes
  Chat at `/v1/chat/completions` and lists a bare JSON array.
- **Bedrock.** `NewBedrock(region, baseURL, config)` sends a Bedrock API
  key as a bearer to the base URL, or to the region's bedrock-runtime
  endpoint, us-east-1 by default. Nothing is signed, and a malformed
  region is refused.
- **Errors** are `*core.ProviderError`, classified by the gateway's status
  set with `Retry-After`. A native Responses 404 or 405 is a
  `*core.SurfaceError`, and a catalog failure's `*providers.CatalogError`
  has a `CatalogCode*` code.

## Anthropic

`providers.Anthropic` is the Anthropic Messages vertical: Messages passed
through as the gateway passes them, token counts, and the catalog and what
its rows mean.

```go
provider, err := providers.NewAnthropic(providers.AnthropicConfig{
    BaseURL:  s.AnthropicBaseURL, // optional; https://api.anthropic.com
    Preamble: gatewayPreamble,    // optional: func(ctx) string
})
```

- **Credential.** An API key is sent as `x-api-key`. A setup token, a
  credential of kind `core.TokenTypeAnthropicSetupToken`, goes through
  llm-provider-auth's `anthropic.HeaderSource` as the OAuth bearer with its
  beta marker; like the gateway, the header source also recognizes one given
  as an API key. Without a credential nothing authenticates, for an
  Anthropic-compatible endpoint that needs none. A credential of another
  kind is refused before anything is sent.
- **Surfaces.** Messages only, and preserved: the body passes through with
  the request's model and the operation's stream flag, the answer comes back
  as Anthropic sent it, and `core.PreservesWire` reports it, so a product
  labels it native. A setup token's completion is requested as a stream and
  assembled, as in the gateway. A stream is relayed byte for byte and fails
  if it ends before `message_stop`. `translation.Adapter` has no route from
  Chat Completions to Messages, so a Chat client needs another target.
- **Preamble.** The hook's text goes before the system prompt, as the
  gateway puts its preamble there. A body's `_llmgw_preamble`, which
  llm-translate also reads, supplies it when the hook returns none, and
  never reaches Anthropic.
- **Token counts.** `CountTokens` implements `core.TokenCounter` over
  `/v1/messages/count_tokens`, forwarding the request's `anthropic-version`
  and `anthropic-beta` values that hold no control character.
- **Catalog.** `ListModels` reads the one page of `/v1/models` the gateway
  reads. Every row serves Messages, with the registry's static
  capabilities, fresh for an hour.
- **Errors** are `*core.ProviderError`; a refusal keeps its status and
  `Retry-After`.

## Azure OpenAI

`providers.AzureOpenAI` serves one Azure OpenAI resource as the gateway
serves it: Chat Completions, and the resource's own deployments as its
catalog.

```go
provider, err := providers.NewAzureOpenAI(providers.AzureOpenAIConfig{
    BaseURL: s.AzureEndpoint, // the portal's endpoint; /openai/v1 is added
})
if err != nil {
    return nil, err // not a resource endpoint
}
return translation.Adapter{Provider: provider}, nil // Messages and Responses over Chat
```

- **Credential.** An API key, sent as `api-key` and never as a bearer. A
  request without one is refused before anything is sent.
- **Base URL.** The portal's origin, `/openai` or `/openai/v1`. Anything
  else, a proxy mount included, is refused when the provider is built.
- **Surfaces.** Chat Completions, sent to `/openai/v1/chat/completions`
  without an api-version. The body is rebuilt from the fields the
  gateway's transport forwards, and every other field is reported as an
  advisory loss. It is not the client's body, so Azure OpenAI declares no
  wire preservation and its answers are labelled translated, as the gateway
  labels them. The answer comes back as Azure sent it.
- **Catalog.** The resource's succeeded, chat-callable deployments, listed
  at the pinned api-version `2023-03-15-preview` and never from `/models`.
  A page is continued by cursor only on the resource's origin, and no
  redirect off the origin is followed, since it would carry the api-key.
- **Errors** are `*core.ProviderError`. A catalog failure's code is the
  gateway's: `authentication_failed` for a rejected key and
  `not_discoverable` for any other.

## Google

`providers.Google` is the Gemini vertical for both of Google's deployments,
AI Studio and Vertex AI, over their native `generateContent` API: Chat,
embeddings, image generation, the catalog and what its rows mean, and the
credential each deployment takes. Requests are shaped byte for byte as the
gateway shapes them.

```go
Providers: func(s Settings, instance string) (core.Provider, error) {
    google, err := providers.NewGoogle(providers.GoogleConfig{
        Deployment:  providers.GoogleVertexAI, // or providers.GoogleAIStudio
        Project:     s.VertexProject,          // optional: a key names its own
        Location:    s.VertexLocation,         // optional: "global"
        RequestType: s.VertexRequestType,      // optional: "paygo" or "dedicated"
    })
    if err != nil {
        return nil, err
    }
    return translation.Adapter{Provider: google}, nil // Messages and Responses over Chat
},
```

- **Credential.** An API key travels in `x-goog-api-key`, never in a URL,
  and a token is the bearer. On Vertex AI a credential of kind
  `core.TokenTypeGCPServiceAccount` holds a service-account key in `Token`.
  Each instance exchanges it for cloud-platform tokens and caches them
  until shortly before they expire, sharing them with no other instance.
  The key's project fills an unset `Project` and must match a set one; a
  project the credential's metadata names under
  `core.CredentialMetadataProjectID` bills calls to it instead. Vertex AI
  without a credential is a configuration error and sends nothing, while
  AI Studio sends what it has, as the gateway does. Other credential kinds
  are refused.
- **Surfaces.** Chat Completions and embeddings, for every model. Chat is
  converted to Gemini `contents` with the gateway's mapping: each message's
  string content, system and developer text as `systemInstruction`,
  `max_tokens` and `temperature`. Every other field is dropped and reported
  as a loss, a material one for tools, for content that is not text and
  for fields that change the answer's structure. Embeddings embed each
  input with one call, and refuse `dimensions` and encodings other than
  float, as the gateway does. `GenerateImages` implements
  `core.ImageGenerator`. As in the gateway, Google does not stream: `Stream`
  refuses and permits failover. Video generation is not served yet.
- **Catalog.** AI Studio's catalog is paged, and a model's
  `supportedGenerationMethods` say what it serves. Vertex AI lists the
  managed models of the instance's location on its v1beta1 publisher
  route, which only an OAuth principal may read: with an API key alone the
  catalog fails with `CatalogCodeNotDiscoverable`, and without a project
  with `CatalogCodeConfigurationIncomplete`. A walk is bounded at 100
  pages and 20,000 rows. Rows carry the gateway's legacy capabilities and
  the typed capabilities it infers from them.
- **Errors** are `*core.ProviderError`, classified by the gateway's status
  set. A refusal names the cause Google's error envelope gives, such as
  exhausted billing or a model the project cannot use, and its cause's
  diagnostic is the gateway's own message, redacted. An empty answer names
  its reason, such as reasoning that spent all of `max_tokens`.

## Ollama

`providers.Ollama` is the Ollama vertical: a keyless daemon over its native
`/api/chat`, and its `/api/tags` catalog.

```go
Providers: func(s Settings, instance string) (core.Provider, error) {
    ollama, err := providers.NewOllama(providers.OllamaConfig{
        BaseURL: s.OllamaBaseURL, // the daemon root; empty is http://127.0.0.1:11434
    })
    if err != nil {
        return nil, err // a base OllamaBaseURLIssue objects to, such as .../v1
    }
    return translation.Adapter{Provider: ollama}, nil // Messages and Responses over Chat
},
```

- **Base URL.** `OllamaBaseURLIssue` is the gateway's diagnostic: the base
  is the daemon's native root, not its OpenAI-compatible `/v1` URL.
  `NewOllama` refuses a base it objects to.
- **Requests** are converted to `/api/chat` byte for byte as the gateway
  converts them: the messages, `temperature`, `top_p`, `tools`, and
  `max_tokens` as `num_predict`. Other fields are dropped and reported as
  losses. The gateway's quirks stay: a developer message is sent as a system
  one, an empty message is dropped, and content that is not a string is sent
  as Go prints it, so an image never reaches the model as an image. Such
  content is reported as a material loss at its path, as is each part of it
  that is not text, such as an image.
- **Answers.** A reply becomes a Chat completion with usage from Ollama's
  evaluation counts, and a stream becomes Chat chunks ending in `[DONE]`,
  each as the gateway renders it. A stream reports its losses through
  `core.LossReporter`.
- **Catalog.** `ListModels` names each row by its name, or else its model,
  with its `details.family` as the vendor. Ollama reports no surfaces or
  capabilities.
- **Errors** are `*core.ProviderError`, classified by the gateway's status
  set with `Retry-After`. A stream that breaks off may be repeated, as in
  the gateway.

## Edge TTS

`providers.EdgeTTS` is the vertical for the speech service behind Microsoft
Edge's read-aloud feature, on the `audio_speech` surface. Core has no
websocket client, so the product supplies the dialer.

```go
type WebSocketDialer func(ctx context.Context, url string, header http.Header, subprotocols []string) (WebSocketConn, *http.Response, error)
type WebSocketConn interface {
    WriteText(ctx context.Context, data []byte) error
    Read(ctx context.Context) (messageType int, data []byte, err error)
    Close() error
}

speech, err := providers.NewEdgeTTS(providers.EdgeTTSConfig{Dial: dial}) // Dial is required
```

- **Dialer.** Over gorilla/websocket it is a `Dialer` with the given
  `Subprotocols`, and a connection whose `WriteText` is
  `WriteMessage(websocket.TextMessage, data)` and whose `Read` is
  `ReadMessage`, each applying the context's deadline. A refused handshake
  returns its response, whose `Date` teaches the clock skew. `Close` must be
  safe to call concurrently with the other methods.
- **Requests.** `Invoke` takes an OpenAI speech request and answers with
  `audio/mpeg`. The model names the voice, and `default` the configured
  default voice; a body `voice` naming another is reported as a loss, not
  read. `speed` becomes the prosody rate as the gateway maps it, and a
  `response_format` other than mp3 is refused. `Synthesize` takes a voice,
  text and rate, as the gateway's speech endpoint calls its synthesizer.
- **Frames** are the gateway's byte for byte: the speech configuration, then
  SSML with the text cleaned, escaped and split into 4096-byte messages,
  each over its own connection signed with `Sec-MS-GEC`. A voice or rate the
  SSML cannot carry is refused, and a chunk never ends inside a character.
  A 403 teaches the instance its clock skew, and the dial is retried once.
- **Credential.** Optional. An API key, or a token, is the access token;
  without one the read-aloud feature's public token is sent.
- **Catalog.** `ListModels` lists the voices, each a Microsoft model serving
  `/v1/audio/speech`. A list that cannot be read is a catalog failure.
- **Errors** are `*core.ProviderError`. A refused handshake carries its
  status, and no error quotes the signed URL.

## Execution

`execution` holds the primitives failover is built from. Endpoints, policy
gates, translation-loss routing and affinity stay product code: a product
orders its candidates and hands the list over.

- **`HealthTracker`** keeps each key's circuit and cooldown behind a mutex,
  with an injectable clock. Consecutive circuit failures open the circuit at
  `FailureThreshold`, for `OpenDuration` or what a `Backoff` returns. After
  that it is half-open: the next failure reopens it and a success closes it.
  A Retry-After keeps the key unavailable until then. Policies can differ per
  key. `Observe`, the default reading of an error, uses `core.ClassifyError`,
  and a product can replace it.
- **`Execute`** tries candidates in order. It skips unavailable ones, stops at
  a terminal disposition, and moves on after a failover or retryable one,
  repeating the candidate first only when `Retry` asks. It records outcomes
  and returns the result with a trace of every candidate reached. A canceled
  context stops it and records nothing.
- **`ExecuteStream`** commits to a candidate once a frame carries output,
  content or reasoning, as the product's predicate decides. Earlier frames are
  held back, so a failure before output moves to the next candidate unseen.
  After output nothing fails over: a failure reaches the caller as the
  stream's `*AfterOutputError`.

```go
tracker := execution.NewHealthTracker(execution.HealthOptions{})
executor := execution.Executor[core.Target]{
    Health: tracker,
    Key:    func(target core.Target) string { return target.Provider },
}
result, err := execution.ExecuteStream(ctx, executor, targets, open, carriesOutput)
if err != nil {
    return err // result.Attempts still holds the trace
}
defer result.Value.Close()
```

### Per-instance resilience

`execution.Resilient(provider, key, policy)` guards one instance, as the
gateway's resilience wrapper did:

- While `policy.Health` reports the key unavailable, every operation fails
  with a `*CircuitOpenError`, a 503 that permits failover, and nothing is
  sent.
- A try whose failure is retryable repeats up to `Retry.Attempts`, waiting
  `ExponentialBackoff` or a longer Retry-After. A stream repeats only its
  opening. `RepeatableRequest` never repeats a stateful Responses request,
  or one that may pay for a second result, such as an image.
- The outcome of each operation's last try moves the circuit. `ListModels`
  passes through, and the wrapper unwraps, so wire preservation and token
  counts read the provider's declarations.
- `TransientClassification` is the gateway's reading, which a policy opts
  into: only 408, 429, 500, 502, 503 and 504 repeat, and every repeat counts
  against the circuit. Core's default reading is unchanged.

`runtime.Options.Policy` returns an instance's `runtime.Policy`, and the
Runtime wraps the provider it binds. Its circuits live in one tracker it
owns, keyed by instance, so they outlive settings changes:

```go
Policy: func(s Settings, instance string) runtime.Policy {
    p := s.PolicyFor(instance) // the gateway's retry and circuit settings
    return runtime.Policy{
        Retry: execution.Retry{Attempts: p.RetryMaxAttempts,
            Delay: execution.ExponentialBackoff(p.InitialBackoff, p.Multiplier, p.MaxBackoff)},
        Circuit: execution.HealthPolicy{FailureThreshold: p.CircuitFailureThreshold,
            OpenDuration: p.CircuitCooldown, IgnoreRetryAfter: true},
        Classify: execution.TransientClassification,
    }
},
```

## Anonymous providers

`anonymous.Orchestrator` runs the automation that connects the reviewed
anonymous providers. For each profile it enrolls the provider, claims its
daily check, discovers the admitted catalog, probes every model with
`Reply with: ok`, and publishes only the targets that answered.

```go
orchestrator, err := anonymous.New(anonymous.Options{
    Catalog: catalog, // Discover(ctx, caller, instance): the admitted models, read now
    Invoker: rt,      // a runtime.Runtime: probes run under each instance's policy
    Hooks:   hooks,   // Enabled, Enroll, Claim, Generation, Record
})
stop := orchestrator.Start(ctx, time.Hour, wake) // at once, hourly, and on wake
defer stop()
```

- **Hooks** keep policy and persistence in the product: the gate, enrolling
  a provider in its configuration, the durable claim, the evidence
  generation, and recording each `Result`.
- **Probes** cover every discovered model, or with `ProbeVerificationModel`
  the reviewed default. A probe keeps its status and Retry-After, so a 401
  reads as a rejection and a 429 as retryable.
- **Results** carry the gateway's report: status, authentication state,
  catalog and completion evidence, counts, failure code, retryability and the
  targets. `ConnectAll` is the manual run, which consults neither the gate
  nor the claim.

`providers.AutoConnectAnonymousProviders` and
`providers.NewAnonymousOpenAICompatibleAdapter` are deprecated in its
favour.

## OAuth flows

`oauthflow.Service` runs the server-side sign-in flows both products expose:
browser authorization codes with PKCE, device authorization, and codes the
user pastes back. Products keep their HTTP routes and map each onto one call:

```go
svc, err := oauthflow.New(oauthflow.Options{
    // Or a shared store. The memory store can cap each caller's pending flows.
    Store:         oauthflow.NewMemoryFlowStore(oauthflow.MemoryFlowStoreOptions{MaxFlowsPerCaller: 5}),
    Credentials:   credentialStore, // core.CredentialStore
    Drivers:       driverFor,       // (instance, method) -> Driver
    CredentialKey: keyFor,          // product policy
})
view, err := svc.Start(ctx, caller, "antigravity", oauthflow.MethodBrowser,
    oauthflow.WithRedirectURI(callbackURL))
view, err = svc.Callback(ctx, oauthflow.CompleteInput{Code: code, State: state})
```

- A `FlowStore` binds each flow to the whole `Caller` that started it. Another
  caller gets `ErrFlowNotFound`, as if the flow did not exist.
- `Consume` is atomic and single-use, and an expired flow is never consumed.
  Run `oauthflowtest.Run` against every implementation.
- `Poll` never asks the provider more often than the flow's interval, even
  across processes, and honours `slow_down`.
- `Complete` and `Callback` consume the flow before the code exchange, so a
  code is used at most once even when the exchange fails. A pasted redirect
  URL whose state does not match spends nothing, so the user can paste again.
- Verifiers, OAuth states, device codes and tokens stay on the server. The
  returned `View` never holds them.
- Credentials are saved through `CredentialStore.Save` under a key the product
  chooses.
- `BrowserPKCE` is a ready driver over llm-provider-auth's `browseroauth`.

The package documentation maps each gateway and Facet Studio route onto the
Service.

## License

MIT
