package runtime_test

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
)

type settings struct{ Endpoint string }

// counters are shared by every provider a factory builds, so they survive
// rebuilds after a settings change.
type counters struct {
	builds, invokes, lists atomic.Int32
	mu                     sync.Mutex
	seen                   []*core.Credential
}

func (c *counters) record(credential *core.Credential) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.seen = append(c.seen, credential)
}

func (c *counters) credentials() []*core.Credential {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]*core.Credential(nil), c.seen...)
}

type fakeProvider struct {
	endpoint string
	counters *counters
	reject   func(*core.Credential) bool
	failWith error
}

func (p *fakeProvider) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

func (p *fakeProvider) answer(ctx context.Context, credential *core.Credential) error {
	p.counters.record(credential)
	if err := ctx.Err(); err != nil {
		return err
	}
	if p.failWith != nil {
		return p.failWith
	}
	if p.reject != nil && p.reject(credential) {
		return &core.ProviderError{Message: "unauthorized", Class: core.ProviderErrorAuth,
			Classification: core.ProviderErrorClassification{StatusCode: 401}}
	}
	return nil
}

func (p *fakeProvider) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	p.counters.invokes.Add(1)
	if err := p.answer(ctx, request.Credential); err != nil {
		return core.Response{}, err
	}
	return core.Response{Body: []byte(`{"endpoint":"` + p.endpoint + `"}`), ContentType: core.ContentTypeJSON}, nil
}

func (p *fakeProvider) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	if err := p.answer(ctx, request.Credential); err != nil {
		return nil, err
	}
	return emptyStream{}, nil
}

func (p *fakeProvider) ListModels(ctx context.Context, credential *core.Credential) ([]core.ModelInfo, error) {
	p.counters.lists.Add(1)
	if err := p.answer(ctx, credential); err != nil {
		return nil, err
	}
	return []core.ModelInfo{{ID: "model-a", Object: "model"}}, nil
}

type emptyStream struct{}

func (emptyStream) Next() ([]byte, error) { return nil, io.EOF }
func (emptyStream) Close() error          { return nil }

func factory(c *counters, configure func(*fakeProvider)) coreruntime.ProviderFactory[settings] {
	return func(s settings, _ string) (core.Provider, error) {
		c.builds.Add(1)
		provider := &fakeProvider{endpoint: s.Endpoint, counters: c}
		if configure != nil {
			configure(provider)
		}
		return provider, nil
	}
}

var chat = core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m", Body: []byte(`{}`), ContentType: core.ContentTypeJSON}

func mustRuntime(t *testing.T, options coreruntime.Options[settings]) *coreruntime.Runtime[settings] {
	t.Helper()
	if options.Settings == nil {
		options.Settings = coreruntime.NewMemorySettings(settings{Endpoint: "one"})
	}
	runtime, err := coreruntime.New(options)
	if err != nil {
		t.Fatal(err)
	}
	return runtime
}

func TestNewRequiresSettingsAndProvidersAndDefaultsTheRest(t *testing.T) {
	t.Parallel()
	if _, err := coreruntime.New(coreruntime.Options[settings]{}); err == nil {
		t.Fatal("a Runtime without settings or providers was created")
	}
	runtime := mustRuntime(t, coreruntime.Options[settings]{Providers: factory(&counters{}, nil)})
	if runtime.Registry() != providers.DefaultRegistry() {
		t.Fatal("a nil registry must default to the core registry")
	}
	if health := runtime.Health("openai"); health.Status != core.ProviderHealthUnknown {
		t.Fatalf("unobserved health=%+v", health)
	}
}

func TestInvokeResolvesEachCallersCredential(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCredentialStore()
	user := core.Caller{ID: "user-1", Kind: core.CallerHuman}
	if _, err := store.Save(ctx, "system-key", core.APIKeyRecord("static-api-key")); err != nil {
		t.Fatal(err)
	}
	personal, err := store.Save(ctx, "user-1-key", tokenstore.Record{AccessToken: "user-token", TokenType: "Bearer", AccountID: "account-1"})
	if err != nil {
		t.Fatal(err)
	}
	store.BindShared("openai", "system-key")
	store.Bind(user, "openai", "user-1-key")
	calls := &counters{}
	evidence := &core.MemoryEvidenceSink{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{Providers: factory(calls, nil), Credentials: store, Evidence: evidence})

	if _, err := runtime.Invoke(ctx, user, "openai", chat); err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.Invoke(ctx, core.Caller{ID: "user-2", Kind: core.CallerHuman}, "openai", chat); err != nil {
		t.Fatal(err)
	}
	seen := calls.credentials()
	if seen[0].Token != "user-token" || seen[0].APIKey != "" || seen[0].ConnectionID != "user-1-key" {
		t.Fatalf("personal credential=%+v", seen[0])
	}
	if seen[1].APIKey != "static-api-key" || seen[1].Token != "" {
		t.Fatalf("shared credential=%+v", seen[1])
	}
	records := evidence.Records()
	if len(records) != 2 || records[0].CredentialKey != "user-1-key" || records[0].CredentialRevision != personal.Revision ||
		records[0].AccountID != "account-1" || records[0].Operation != core.EvidenceOperationInvoke ||
		records[0].Surface != core.ModelSurfaceChatCompletions || records[0].Model != "m" ||
		records[0].Outcome.Status != core.ProviderHealthHealthy || records[0].Caller != user {
		t.Fatalf("evidence=%+v", records)
	}
	if health := runtime.Health("openai"); health.Status != core.ProviderHealthHealthy {
		t.Fatalf("health=%+v", health)
	}
}

func TestInvokeWithoutAResolvableCredentialSendsNone(t *testing.T) {
	t.Parallel()
	calls := &counters{}
	anonymous := mustRuntime(t, coreruntime.Options[settings]{Providers: factory(calls, nil)})
	if _, err := anonymous.Invoke(context.Background(), core.LocalCaller(), "local", chat); err != nil {
		t.Fatal(err)
	}
	unbound := mustRuntime(t, coreruntime.Options[settings]{Providers: factory(calls, nil), Credentials: core.NewMemoryCredentialStore()})
	if _, err := unbound.Invoke(context.Background(), core.LocalCaller(), "local", chat); err != nil {
		t.Fatal(err)
	}
	for _, credential := range calls.credentials() {
		if credential != nil {
			t.Fatalf("a request without a credential carried %+v", credential)
		}
	}
}

func oauthStore(t *testing.T, now time.Time) *core.MemoryCredentialStore {
	t.Helper()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(context.Background(), "user-1-codex", tokenstore.Record{
		AccessToken: "stale", RefreshToken: "refresh-1", TokenType: "Bearer", AccountID: "account-1", Expiry: now.Add(time.Hour),
	}); err != nil {
		t.Fatal(err)
	}
	store.Bind(core.Caller{ID: "user-1", Kind: core.CallerHuman}, "codex", "user-1-codex")
	return store
}

func countingRefresh(refreshes *atomic.Int32, now time.Time) coreruntime.RefreshFactory[settings] {
	return func(settings, string) tokenstore.RefreshFunc {
		return func(_ context.Context, current tokenstore.Record) (tokenstore.Record, error) {
			refreshes.Add(1)
			return tokenstore.Record{AccessToken: "fresh", RefreshToken: "refresh-2", AccountID: current.AccountID, Expiry: now.Add(time.Hour)}, nil
		}
	}
}

func TestRejectedOAuthCredentialIsRefreshedAndReplayedOnce(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Now()
	store := oauthStore(t, now)
	calls := &counters{}
	refreshes := &atomic.Int32{}
	evidence := &core.MemoryEvidenceSink{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{
		Providers: factory(calls, func(p *fakeProvider) {
			p.reject = func(c *core.Credential) bool { return c != nil && c.Token == "stale" }
		}),
		Credentials: store, Refresh: countingRefresh(refreshes, now), Evidence: evidence,
	})

	if _, err := runtime.Invoke(ctx, core.Caller{ID: "user-1", Kind: core.CallerHuman}, "codex", chat); err != nil {
		t.Fatal(err)
	}
	seen := calls.credentials()
	if len(seen) != 2 || seen[0].Token != "stale" || seen[1].Token != "fresh" || refreshes.Load() != 1 {
		t.Fatalf("seen=%+v refreshes=%d, want stale then fresh after one refresh", seen, refreshes.Load())
	}
	stored, err := store.Load(ctx, "user-1-codex")
	if err != nil || stored.AccessToken != "fresh" {
		t.Fatalf("stored=%+v err=%v", stored, err)
	}
	if records := evidence.Records(); records[0].CredentialRevision != stored.Revision {
		t.Fatalf("evidence revision=%q, want the refreshed %q", records[0].CredentialRevision, stored.Revision)
	}
}

func TestRejectedAPIKeyIsNotReplayed(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(ctx, "system-key", core.APIKeyRecord("static-api-key")); err != nil {
		t.Fatal(err)
	}
	store.BindShared("openai", "system-key")
	calls := &counters{}
	refreshes := &atomic.Int32{}
	evidence := &core.MemoryEvidenceSink{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{
		Providers: factory(calls, func(p *fakeProvider) {
			p.reject = func(c *core.Credential) bool { return c != nil && c.APIKey == "static-api-key" }
		}),
		Credentials: store, Refresh: countingRefresh(refreshes, time.Now()), Evidence: evidence,
	})
	_, err := runtime.Invoke(ctx, core.LocalCaller(), "openai", chat)
	if core.ClassifyError(err).StatusCode != 401 || calls.invokes.Load() != 1 || refreshes.Load() != 0 {
		t.Fatalf("err=%v invokes=%d refreshes=%d", err, calls.invokes.Load(), refreshes.Load())
	}
	if health := runtime.Health("openai"); health.Status != core.ProviderHealthUnhealthy || health.ErrorClass != core.ProviderErrorAuth {
		t.Fatalf("health=%+v", health)
	}
	if records := evidence.Records(); len(records) != 1 || records[0].Outcome.ErrorClass != core.ProviderErrorAuth {
		t.Fatalf("evidence=%+v", records)
	}
}

func TestSettingsGenerationRebuildsProviders(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	source := coreruntime.NewMemorySettings(settings{Endpoint: "one"})
	calls := &counters{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{Settings: source, Providers: factory(calls, nil)})
	for range 2 {
		response, err := runtime.Invoke(ctx, core.LocalCaller(), "p", chat)
		if err != nil || !strings.Contains(string(response.Body), `"one"`) {
			t.Fatalf("response=%s err=%v", response.Body, err)
		}
	}
	if calls.builds.Load() != 1 {
		t.Fatalf("builds=%d, want one per generation", calls.builds.Load())
	}
	source.Update(settings{Endpoint: "two"})
	response, err := runtime.Invoke(ctx, core.LocalCaller(), "p", chat)
	if err != nil || !strings.Contains(string(response.Body), `"two"`) || calls.builds.Load() != 2 {
		t.Fatalf("response=%s err=%v builds=%d, want a rebuild with the new settings", response.Body, err, calls.builds.Load())
	}
}

func TestListModelsServesFreshCatalogsAndRediscovers(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Unix(1_800_000_000, 0).UTC()
	clock := func() time.Time { return now }
	source := coreruntime.NewMemorySettings(settings{Endpoint: "one"})
	catalogs := core.NewMemoryCatalogStore()
	calls := &counters{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{
		Settings: source, Providers: factory(calls, nil), Catalogs: catalogs, CatalogTTL: time.Minute, Now: clock,
	})

	record, err := runtime.ListModels(ctx, core.LocalCaller(), "p")
	if err != nil || record.Evidence.Status != core.CatalogDiscovered || !record.Evidence.ObservedAt.Equal(now) || record.Revision == "" {
		t.Fatalf("record=%+v err=%v", record, err)
	}
	if _, err := runtime.ListModels(ctx, core.LocalCaller(), "p"); err != nil || calls.lists.Load() != 1 {
		t.Fatalf("err=%v lists=%d, want the fresh catalog served", err, calls.lists.Load())
	}

	now = now.Add(time.Minute + time.Second)
	if _, err := runtime.ListModels(ctx, core.LocalCaller(), "p"); err != nil || calls.lists.Load() != 2 {
		t.Fatalf("err=%v lists=%d, want rediscovery after the TTL", err, calls.lists.Load())
	}

	source.Update(settings{Endpoint: "two"})
	now = now.Add(time.Second)
	if _, err := runtime.ListModels(ctx, core.LocalCaller(), "p"); err != nil || calls.lists.Load() != 3 {
		t.Fatalf("err=%v lists=%d, want rediscovery after a settings change", err, calls.lists.Load())
	}

	// Another process shares the catalog store and serves the fresh catalog
	// without discovering it again.
	other := &counters{}
	second := mustRuntime(t, coreruntime.Options[settings]{Providers: factory(other, nil), Catalogs: catalogs, CatalogTTL: time.Minute, Now: clock})
	if _, err := second.ListModels(ctx, core.LocalCaller(), "p"); err != nil || other.lists.Load() != 0 {
		t.Fatalf("err=%v lists=%d, want the shared catalog", err, other.lists.Load())
	}
}

func TestCatalogsAreKeyedByTheDiscoveringCredential(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := core.NewMemoryCredentialStore()
	for _, key := range []string{"shared-key", "user-1-key", "user-2-key"} {
		if _, err := store.Save(ctx, key, core.APIKeyRecord(key+"-value")); err != nil {
			t.Fatal(err)
		}
	}
	store.BindShared("p", "shared-key")
	store.Bind(core.Caller{ID: "user-1", Kind: core.CallerHuman}, "p", "user-1-key")
	store.Bind(core.Caller{ID: "user-2", Kind: core.CallerHuman}, "p", "user-2-key")
	calls := &counters{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{Providers: factory(calls, nil), Credentials: store})
	callers := []core.Caller{
		{ID: "user-1", Kind: core.CallerHuman}, {ID: "user-2", Kind: core.CallerHuman},
		{ID: "user-3", Kind: core.CallerHuman}, {Kind: core.CallerAnonymous},
	}
	for _, caller := range callers {
		if _, err := runtime.ListModels(ctx, caller, "p"); err != nil {
			t.Fatal(err)
		}
	}
	if calls.lists.Load() != 3 {
		t.Fatalf("lists=%d, want one per credential: two personal and one shared", calls.lists.Load())
	}
}

func TestHealthIgnoresNonUpstreamErrors(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	cases := map[string]struct {
		err        error
		wantStatus core.ProviderHealthStatus
		wantClass  core.ProviderErrorClass
	}{
		"configuration": {core.NewConfigurationError("no key", nil), core.ProviderHealthUnknown, core.ProviderErrorConfiguration},
		"surface":       {&core.SurfaceError{Surface: core.ModelSurfaceResponses, Model: "m"}, core.ProviderHealthUnknown, core.ProviderErrorNone},
		"loss policy":   {&core.LossPolicyError{}, core.ProviderHealthUnknown, core.ProviderErrorNone},
		"transport": {&providers.InvocationError{Msg: "connection reset", Retryable: true, CircuitFailure: true},
			core.ProviderHealthUnhealthy, core.ProviderErrorTransport},
		"rate limited": {&providers.InvocationError{Msg: "slow down", Status: 429, RetryAfter: 5 * time.Second},
			core.ProviderHealthDegraded, core.ProviderErrorRateLimited},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			evidence := &core.MemoryEvidenceSink{}
			runtime := mustRuntime(t, coreruntime.Options[settings]{
				Providers: factory(&counters{}, func(p *fakeProvider) { p.failWith = tc.err }), Evidence: evidence,
			})
			if _, err := runtime.Invoke(ctx, core.LocalCaller(), "p", chat); !errors.Is(err, tc.err) {
				t.Fatalf("err=%v, want %v", err, tc.err)
			}
			if health := runtime.Health("p"); health.Status != tc.wantStatus {
				t.Fatalf("health=%+v, want status %q", health, tc.wantStatus)
			}
			records := evidence.Records()
			if len(records) != 1 || records[0].Outcome.ErrorClass != tc.wantClass {
				t.Fatalf("evidence=%+v, want class %q", records, tc.wantClass)
			}
			if name == "rate limited" && records[0].Outcome.RetryAfter != 5*time.Second {
				t.Fatalf("retry after=%v", records[0].Outcome.RetryAfter)
			}
		})
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	evidence := &core.MemoryEvidenceSink{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{Providers: factory(&counters{}, nil), Evidence: evidence})
	if _, err := runtime.Invoke(canceled, core.LocalCaller(), "p", chat); !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v", err)
	}
	if len(evidence.Records()) != 0 || runtime.Health("p").Status != core.ProviderHealthUnknown {
		t.Fatal("a caller that gave up changed evidence or health")
	}
}

func TestInvalidRequestsNeverReachAProvider(t *testing.T) {
	t.Parallel()
	calls := &counters{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{Providers: factory(calls, nil)})
	_, err := runtime.Invoke(context.Background(), core.LocalCaller(), "p", core.Request{Surface: "telepathy", Model: "m"})
	var providerErr *core.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Class != core.ProviderErrorInvalidRequest || core.ClassifyError(err).Disposition() != core.DispositionTerminal {
		t.Fatalf("err=%v", err)
	}
	if _, err := runtime.Stream(context.Background(), core.LocalCaller(), "p", core.Request{Surface: core.ModelSurfaceChatCompletions}); err == nil {
		t.Fatal("a stream without a model was opened")
	}
	if calls.builds.Load() != 0 {
		t.Fatal("an invalid request built a provider")
	}
}

func TestStreamReplaysARejectedOAuthCredentialBeforeAnyFrame(t *testing.T) {
	t.Parallel()
	now := time.Now()
	calls := &counters{}
	refreshes := &atomic.Int32{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{
		Providers: factory(calls, func(p *fakeProvider) {
			p.reject = func(c *core.Credential) bool { return c != nil && c.Token == "stale" }
		}),
		Credentials: oauthStore(t, now), Refresh: countingRefresh(refreshes, now),
	})
	stream, err := runtime.Stream(context.Background(), core.Caller{ID: "user-1", Kind: core.CallerHuman}, "codex", chat)
	if err != nil || stream == nil || refreshes.Load() != 1 {
		t.Fatalf("stream=%v err=%v refreshes=%d", stream, err, refreshes.Load())
	}
	if seen := calls.credentials(); len(seen) != 2 || seen[1].Token != "fresh" {
		t.Fatalf("seen=%+v", seen)
	}
}

func TestFactoryFailuresSurface(t *testing.T) {
	t.Parallel()
	missing := mustRuntime(t, coreruntime.Options[settings]{Providers: func(settings, string) (core.Provider, error) { return nil, nil }})
	_, err := missing.Invoke(context.Background(), core.LocalCaller(), "p", chat)
	var providerErr *core.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Class != core.ProviderErrorConfiguration {
		t.Fatalf("nil provider err=%v", err)
	}
	broken := errors.New("provider p is misconfigured")
	failing := mustRuntime(t, coreruntime.Options[settings]{Providers: func(settings, string) (core.Provider, error) { return nil, broken }})
	if _, err := failing.ListModels(context.Background(), core.LocalCaller(), "p"); !errors.Is(err, broken) {
		t.Fatalf("factory err=%v", err)
	}
}

func TestConcurrentInvokesShareOneProvider(t *testing.T) {
	t.Parallel()
	calls := &counters{}
	runtime := mustRuntime(t, coreruntime.Options[settings]{Providers: factory(calls, nil), Credentials: core.NewMemoryCredentialStore()})
	var group sync.WaitGroup
	for range 32 {
		group.Add(1)
		go func() {
			defer group.Done()
			if _, err := runtime.Invoke(context.Background(), core.LocalCaller(), "p", chat); err != nil {
				t.Error(err)
			}
		}()
	}
	group.Wait()
	if calls.builds.Load() != 1 || calls.invokes.Load() != 32 {
		t.Fatalf("builds=%d invokes=%d", calls.builds.Load(), calls.invokes.Load())
	}
}
