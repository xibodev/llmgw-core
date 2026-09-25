package anonymous_test

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/anonymous"
	"github.com/xibodev/llmgw-core/execution"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
)

// refusing is a provider that refuses every operation with err.
type refusing struct {
	err   error
	tries *atomic.Int32
}

func (p refusing) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

func (p refusing) Invoke(context.Context, core.Request) (core.Response, error) {
	p.tries.Add(1)
	return core.Response{}, p.err
}

func (p refusing) Stream(context.Context, core.Request) (core.StreamIter, error) { return nil, p.err }

func (p refusing) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	return []core.ModelInfo{{ID: "codestral-latest"}}, nil
}

// A Runtime is an Invoker, so probes run under each instance's resilience
// policy: the port of the gateway's verification tests, whose providers
// retry three times, against a Runtime.
func TestProbesRunUnderTheRuntimesPolicy(t *testing.T) {
	t.Parallel()
	for name, tc := range map[string]struct {
		err   error
		tries int32
		check func(anonymous.Result) bool
	}{
		"a rejection is never repeated": {rejected(401), 1, func(r anonymous.Result) bool {
			return r.FailureCode == anonymous.FailureAuthenticationRejected && !r.Retryable
		}},
		"a throttled probe keeps Retry-After": {rateLimited(), 3, func(r anonymous.Result) bool {
			return r.FailureCode == anonymous.FailureVerificationFailed && r.Retryable && r.RetryAfter == 17*time.Second
		}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			tries := &atomic.Int32{}
			runtime, err := coreruntime.New(coreruntime.Options[struct{}]{
				Settings:  coreruntime.NewMemorySettings(struct{}{}),
				Providers: func(struct{}, string) (core.Provider, error) { return refusing{tc.err, tries}, nil },
				Policy: func(struct{}, string) coreruntime.Policy {
					return coreruntime.Policy{Retry: execution.Retry{Attempts: 3}, Classify: execution.TransientClassification}
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			catalog := anonymous.CatalogFunc(func(ctx context.Context, caller core.Caller, instance string) ([]core.ModelInfo, error) {
				record, err := runtime.ListModels(ctx, caller, instance)
				return record.Evidence.Models, err
			})
			o, err := anonymous.New(anonymous.Options{Catalog: catalog, Invoker: runtime, Hooks: &product{}})
			if err != nil {
				t.Fatal(err)
			}
			results := o.RunOnce(context.Background())
			if len(results) != 5 || tries.Load() != 5*tc.tries || !tc.check(results[0]) {
				t.Fatalf("tries=%d results=%+v", tries.Load(), results)
			}
		})
	}
}
