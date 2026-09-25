# OpenCode Zen anonymous provider

Package `providers/zen` contains the reusable, storage-neutral OpenCode Zen
anonymous admission and provider behavior.

## Public API

`providers.Zen`, the `core.Provider` for OpenCode Zen, is built on this
package; see the module README. The methods marked deprecated below are
superseded by it and keep working as before.

- `New(Config) (*Client, error)` accepts injectable HTTP transport, endpoints,
  clock, ID source, response bound, capability TTL, and user agent.
- `Normalize(metadata, live, NormalizeOptions)` derives the catalog from the
  `models.dev` document and, when given, Zen's live `/models`, without
  fetching either. A `*DocumentError` names the document it cannot read.
- `(*Client).Discover(context.Context)` (deprecated) exposes the
  OpenCode-compatible `models.dev` snapshot: zero input cost, normal status
  filtering, and the model/provider npm surface as typed
  `core.ModelCapabilities`.
- `(*Client).DiscoverVerified(context.Context)` (deprecated) intersects that
  snapshot with the live Zen catalog and enforces strict all-zero pricing.
- `(*Client).Connect(context.Context, core.ProviderConnectRequest)` implements
  `core.ProviderConnector`, performs one native-surface inference probe, and
  returns standard catalog, probe, health, and exact-target evidence.
- `(*Client).CompleteNative(...)` (deprecated) sends an already
  surface-correct Chat or Responses payload with the required anonymous
  headers.
- `NativeSurface(core.ModelInfo)` reads the catalog-derived Chat versus
  Responses surface without model-name heuristics.
- `InvocationIdentity`, `WithInvocationIdentity`, and
  `ApplyInvocationHeaders` keep project/session/request/client identity stable
  across transport retries without coupling it to authentication.
- `ClassifyChat`, `ClassifyResponses`, `AdmitChat`, and `AdmitResponses` expose
  request classification while preserving exact caller tools and tool choice.

The package performs no persistence, scheduling, auditing, or wire translation.
Callers own catalog refresh policy and any evidence storage.

## Gateway migration

1. Serve OpenCode Zen with `providers.Zen`, which shapes requests, admits
   anonymous ones and reads the catalog as the gateway's OpenAI-compatible
   transport does today.
2. Create one invocation identity at the API boundary with
   `NewInvocationIdentity`, carry it in context with `WithInvocationIdentity`,
   and every retry of the invocation reuses it.
3. Persist the rows `ListModels` returns only in the gateway's existing catalog
   store, retaining the core freshness fields, and hand `providers.Zen` a
   lookup over them. It routes each row by the endpoints it lists; the Muse
   prefix only routes a model the catalog lacks, as the gateway's cold
   catalog always has.
4. Register the client directly as the `core.ProviderConnector` when standard
   discovery/probe/publication evidence is desired. The generic
   `ProviderOrchestrator` adapter is intentionally not used because its fixed
   probe payload is Chat-shaped and cannot probe Responses-native models
   losslessly.

Changing these admission rules changes the meaning of persisted Zen catalog
rows. A gateway with schema-stamped catalog persistence should bump that schema
and force one fresh discovery rather than serving older rows.
