package anonymous_test

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/anonymous"
	"github.com/xibodev/llmgw-core/providers"
)

// product is a fake product behind the hooks. Enroll manages every
// provider unless statuses says otherwise, Claim grants one check per
// interval, and Enabled turns off after enabledFor calls when it is set.
type product struct {
	mu            sync.Mutex
	disabled      bool
	enabledFor    int
	enabledCalls  int
	statuses      map[string]string
	claims        map[string]time.Time
	generationErr error
	recordErr     error
	enrolled      []string
	records       []recorded
}

type recorded struct {
	providerID string
	generation int64
	result     anonymous.Result
}

func (p *product) Enabled(context.Context) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.enabledCalls++
	return !p.disabled && (p.enabledFor == 0 || p.enabledCalls <= p.enabledFor)
}

func (p *product) Enroll(_ context.Context, profile providers.AnonymousProviderProfile) (string, string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.enrolled = append(p.enrolled, profile.ProviderID)
	if status := p.statuses[profile.ProviderID]; status != "" {
		return profile.ProviderID, status
	}
	return profile.ProviderID, anonymous.StatusManaged
}

func (p *product) Claim(_ context.Context, providerID string, at time.Time, every time.Duration) (bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if last, ok := p.claims[providerID]; ok && at.Sub(last) < every {
		return false, nil
	}
	if p.claims == nil {
		p.claims = map[string]time.Time{}
	}
	p.claims[providerID] = at
	return true, nil
}

func (p *product) Generation(context.Context, string) (int64, error) { return 7, p.generationErr }

func (p *product) Record(_ context.Context, providerID string, generation int64, result anonymous.Result) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.records = append(p.records, recorded{providerID, generation, result})
	return p.recordErr
}

func (p *product) recordedResults() []recorded {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]recorded(nil), p.records...)
}

// upstream is a fake Catalog and Invoker. Each instance lists models, and
// a probe of a model fails with failures[model] when set.
type upstream struct {
	mu         sync.Mutex
	models     []core.ModelInfo
	catalogErr error
	failures   map[string]error
	discovers  int
	probes     []probe
}

type probe struct {
	caller          core.Caller
	instance, model string
	body            map[string]any
}

func (u *upstream) Discover(_ context.Context, _ core.Caller, _ string) ([]core.ModelInfo, error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.discovers++
	return u.models, u.catalogErr
}

func (u *upstream) Invoke(_ context.Context, caller core.Caller, instance string, request core.Request) (core.Response, error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	var body map[string]any
	_ = json.Unmarshal(request.Body, &body)
	u.probes = append(u.probes, probe{caller, instance, request.Model, body})
	if err := u.failures[request.Model]; err != nil {
		return core.Response{}, err
	}
	return core.Response{Body: []byte(`{"choices":[{"message":{"content":"ok"}}]}`), ContentType: core.ContentTypeJSON}, nil
}

func (u *upstream) calls() (discovers, probes int) {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.discovers, len(u.probes)
}
