package anonymous_test

import (
	"context"
	"slices"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/anonymous"
)

// wakeful counts runs through Enabled, which every run consults first, and
// reports each one.
type wakeful struct {
	product
	runs chan struct{}
}

func (w *wakeful) Enabled(ctx context.Context) bool {
	w.runs <- struct{}{}
	return false
}

func TestStartRunsAtOnceThenOnWakeUntilStopped(t *testing.T) {
	t.Parallel()
	hooks := &wakeful{runs: make(chan struct{}, 8)}
	wake := make(chan struct{}, 1)
	stop := orchestrator(t, hooks, &upstream{}, anonymous.Options{}).Start(context.Background(), time.Hour, wake)
	awaitRun(t, hooks.runs)
	wake <- struct{}{}
	awaitRun(t, hooks.runs)
	stop()
	stop()
	select {
	case <-hooks.runs:
		t.Fatal("a run started after stop returned")
	default:
	}
	ctx, cancel := context.WithCancel(context.Background())
	stopped := orchestrator(t, hooks, &upstream{}, anonymous.Options{}).Start(ctx, 0, nil)
	awaitRun(t, hooks.runs)
	cancel()
	stopped()
}

func awaitRun(t *testing.T, runs <-chan struct{}) {
	t.Helper()
	select {
	case <-runs:
	case <-time.After(10 * time.Second):
		t.Fatal("no run started")
	}
}

// The ports of TestAutoConnectFreeProvidersAPI and
// TestGatewayProviderOrchestratorKeepsBespokeTransportsInApplication:
// connecting every reviewed profile consults neither the gate nor the
// claim, and every probe, Zen's and Pollinations' included, goes through
// the product's Invoker to the provider's own vertical.
func TestConnectAllChecksEveryReviewedProfileThroughTheInvoker(t *testing.T) {
	t.Parallel()
	hooks := &product{disabled: true}
	u := &upstream{models: []core.ModelInfo{{ID: "shared-free-model"}}}
	o, err := anonymous.New(anonymous.Options{Catalog: u, Invoker: u, Hooks: hooks})
	if err != nil {
		t.Fatal(err)
	}
	results := o.ConnectAll(context.Background())
	instances := make([]string, 0, len(u.probes))
	for _, probe := range u.probes {
		instances = append(instances, probe.instance)
	}
	slices.Sort(instances)
	want := []string{"kilo-code", "llm7", "opencode-zen", "ovh-ai", "pollinations"}
	if len(results) != 5 || !slices.Equal(instances, want) || len(hooks.claims) != 0 || hooks.enabledCalls != 0 {
		t.Fatalf("results=%d probes=%v claims=%v", len(results), instances, hooks.claims)
	}
	for _, result := range results {
		if !result.Success || result.Targets[0].Provider != result.ProviderID {
			t.Fatalf("result=%+v", result)
		}
	}
}

// A run stopped during a check records nothing: the failure it saw is the
// stop, not the provider.
func TestCheckStoppedByItsCallerIsNotRecorded(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	hooks := &product{}
	catalog := anonymous.CatalogFunc(func(context.Context, core.Caller, string) ([]core.ModelInfo, error) {
		cancel()
		return nil, ctx.Err()
	})
	o, err := anonymous.New(anonymous.Options{Catalog: catalog, Invoker: &upstream{}, Hooks: hooks})
	if err != nil {
		t.Fatal(err)
	}
	results := o.RunOnce(ctx)
	if len(results) != 1 || results[0].RecordErr != context.Canceled || len(hooks.recordedResults()) != 0 {
		t.Fatalf("results=%+v records=%+v", results, hooks.recordedResults())
	}
}
