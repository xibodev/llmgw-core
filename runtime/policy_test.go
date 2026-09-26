package runtime_test

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
)

// script fails every guarded operation with err while it is set. Every
// provider a factory builds shares one, so it survives rebuilds.
type script struct {
	mu    sync.Mutex
	err   error
	tries int
}

func (s *script) set(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = err
}

func (s *script) try() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tries++
	return s.err
}

func (s *script) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.tries
}

type scriptedProvider struct{ script *script }

func (p scriptedProvider) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

func (p scriptedProvider) Invoke(context.Context, core.Request) (core.Response, error) {
	if err := p.script.try(); err != nil {
		return core.Response{}, err
	}
	return core.Response{Body: []byte(`{}`), ContentType: core.ContentTypeJSON}, nil
}

func (p scriptedProvider) Stream(context.Context, core.Request) (core.StreamIter, error) {
	if err := p.script.try(); err != nil {
		return nil, err
	}
	return emptyStream{}, nil
}

func (p scriptedProvider) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	return []core.ModelInfo{{ID: "model-a"}}, nil
}

// guardedRuntime is a Runtime whose instance "guarded" repeats three
// times and opens its circuit at the first failure, as the gateway's
// policy maps, and whose other instances are left alone.
func guardedRuntime(t *testing.T, s *script, source *coreruntime.MemorySettings[settings], now func() time.Time) *coreruntime.Runtime[settings] {
	return mustRuntime(t, coreruntime.Options[settings]{
		Settings:  source,
		Providers: func(settings, string) (core.Provider, error) { return scriptedProvider{s}, nil },
		Policy: func(_ settings, instance string) coreruntime.Policy {
			if instance != "guarded" {
				return coreruntime.Policy{}
			}
			return coreruntime.Policy{
				Retry:    execution.Retry{Attempts: 3},
				Circuit:  execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute, IgnoreRetryAfter: true},
				Classify: execution.TransientClassification,
			}
		},
		Now: now,
	})
}

func TestPolicyGuardsEachInstanceAcrossSettingsChanges(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Unix(1_800_000_000, 0).UTC()
	s := &script{err: &providers.InvocationError{Msg: "fixture", Status: http.StatusTooManyRequests}}
	source := coreruntime.NewMemorySettings(settings{Endpoint: "one"})
	runtime := guardedRuntime(t, s, source, func() time.Time { return now })

	if _, err := runtime.Invoke(ctx, core.LocalCaller(), "guarded", chat); err == nil || s.count() != 3 {
		t.Fatalf("err=%v tries=%d, want three tries", err, s.count())
	}
	source.Update(settings{Endpoint: "two"})
	_, err := runtime.Invoke(ctx, core.LocalCaller(), "guarded", chat)
	var open *execution.CircuitOpenError
	if !errors.As(err, &open) || core.ClassifyError(err).StatusCode != http.StatusServiceUnavailable || s.count() != 3 {
		t.Fatalf("err=%v tries=%d, want the open circuit to outlive the settings change", err, s.count())
	}
	if health := runtime.Health("guarded"); health.Status != core.ProviderHealthUnhealthy {
		t.Fatalf("health=%+v", health)
	}
	if _, err := runtime.ListModels(ctx, core.LocalCaller(), "guarded"); err != nil {
		t.Fatalf("an open circuit blocked the catalog: %v", err)
	}
	if _, err := runtime.Stream(ctx, core.LocalCaller(), "other", chat); err == nil || s.count() != 4 {
		t.Fatalf("an unguarded instance: err=%v tries=%d, want one try", err, s.count())
	}

	now = now.Add(time.Minute)
	s.set(nil)
	if stream, err := runtime.Stream(ctx, core.LocalCaller(), "guarded", chat); err != nil || stream == nil {
		t.Fatalf("the half-open circuit refused: %v", err)
	}
}

func TestPolicyWithoutACircuitNeverRefuses(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	for name, tc := range map[string]struct {
		policy coreruntime.PolicyFactory[settings]
		tries  int
	}{
		"no policy": {nil, 10},
		"retry only": {func(settings, string) coreruntime.Policy {
			return coreruntime.Policy{Retry: execution.Retry{Attempts: 2}}
		}, 20},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			s := &script{err: &providers.InvocationError{Msg: "fixture", Status: http.StatusServiceUnavailable, RetryAfter: time.Hour}}
			runtime := mustRuntime(t, coreruntime.Options[settings]{
				Providers: func(settings, string) (core.Provider, error) { return scriptedProvider{s}, nil },
				Policy:    tc.policy,
			})
			for range 10 {
				if _, err := runtime.Invoke(ctx, core.LocalCaller(), "p", chat); errors.As(err, new(*execution.CircuitOpenError)) {
					t.Fatalf("refused without a circuit: %v", err)
				}
			}
			if s.count() != tc.tries {
				t.Fatalf("tries=%d want=%d", s.count(), tc.tries)
			}
		})
	}
}
