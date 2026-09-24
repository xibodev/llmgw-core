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

Request losses are checked against the `LossPolicy` before the provider is
called. All losses, including the provider's own, are returned in
`Response.Losses` or through the stream's `LossReporter`.

## License

MIT
