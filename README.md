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
- **`EvidenceSink`** receives `AccountEvidence`: which credential served which
  operation, and the classified outcome. Reference: `MemoryEvidenceSink`.

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
  gets `/responses`.
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
