# OpenCode Zen anonymous provider

Package `providers/zen` contains the reusable, storage-neutral OpenCode Zen
anonymous admission and provider behavior.

## Public API

- `New(Config) (*Client, error)` accepts injectable HTTP transport, endpoints,
  clock, ID source, response bound, capability TTL, and user agent.
- `(*Client).Discover(context.Context)` fetches the live Zen catalog and
  `models.dev`, returning their strictly admitted intersection as
  `core.CatalogEvidence` with typed `core.ModelCapabilities`.
- `(*Client).Connect(context.Context, core.ProviderConnectRequest)` implements
  `core.ProviderConnector`, performs one native-surface inference probe, and
  returns standard catalog, probe, health, and exact-target evidence.
- `(*Client).CompleteNative(...)` sends an already surface-correct Chat or
  Responses payload with the required anonymous headers and admission tools.
- `NativeSurface(core.ModelInfo)` reads the catalog-derived Chat versus
  Responses surface without model-name heuristics.
- `ClassifyChat`, `ClassifyResponses`, `AdmitChat`, and `AdmitResponses` expose
  request classification and minimal `bash`/`read` admission independently of
  transport.

The package performs no persistence, scheduling, auditing, or wire translation.
Callers own catalog refresh policy and any evidence storage.

## Gateway migration

1. Construct one `zen.Client` from the provider's configured base URL. Leave
   `MetadataURL` empty for the public `models.dev` endpoint or inject it in tests.
2. Replace gateway-local anonymous header generation with
   `client.AnonymousHeaders`, or route requests through `CompleteNative`.
3. Replace the gateway's live-catalog filtering with `client.Discover`. Persist
   the returned rows only in the gateway's existing catalog store, retaining
   the core freshness fields.
4. Route each admitted row using `NativeSurface`; do not retain Muse/model-name
   prefix checks.
5. Apply `AdmitChat` or `AdmitResponses` only for anonymous Zen. Ordinary
   prompts remain unchanged; title behavior requires the explicit canonical
   title-generator instruction.
6. Register the client directly as the `core.ProviderConnector` when standard
   discovery/probe/publication evidence is desired. The generic
   `ProviderOrchestrator` adapter is intentionally not used because its fixed
   probe payload is Chat-shaped and cannot probe Responses-native models
   losslessly.

Changing these admission rules changes the meaning of persisted Zen catalog
rows. A gateway with schema-stamped catalog persistence should bump that schema
and force one fresh discovery rather than serving older rows.
