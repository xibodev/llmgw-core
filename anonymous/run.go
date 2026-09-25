package anonymous

import (
	"context"
	"sync"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

// RunOnce runs the automation once, as the gateway's scheduler does: for
// each profile, while Enabled holds and ctx lasts, it enrolls the provider
// and, when the automation manages it and the claim of its daily check
// succeeds, checks it. It returns a result for every provider it enrolled
// but leaves alone and every provider it checked, and nil when the
// automation is off.
func (o *Orchestrator) RunOnce(ctx context.Context) []Result {
	if ctx.Err() != nil || !o.hooks.Enabled(ctx) {
		return nil
	}
	results := []Result{}
	for _, profile := range o.profiles {
		if ctx.Err() != nil || !o.hooks.Enabled(ctx) {
			break
		}
		providerID, status := o.hooks.Enroll(ctx, profile)
		if status != StatusManaged {
			results = append(results, Result{ProviderID: providerID, RegistryID: profile.RegistryID, Status: status})
			continue
		}
		claimed, err := o.hooks.Claim(ctx, providerID, o.now(), o.every)
		if err != nil || !claimed {
			continue
		}
		if !o.hooks.Enabled(ctx) {
			break
		}
		results = append(results, o.check(ctx, profile, providerID))
	}
	return results
}

// ConnectAll enrolls and checks every profile now, as the gateway's manual
// "connect free providers" does: it consults neither Enabled nor Claim.
func (o *Orchestrator) ConnectAll(ctx context.Context) []Result {
	results := make([]Result, 0, len(o.profiles))
	for _, profile := range o.profiles {
		if ctx.Err() != nil {
			break
		}
		providerID, status := o.hooks.Enroll(ctx, profile)
		if status != StatusManaged {
			results = append(results, Result{ProviderID: providerID, RegistryID: profile.RegistryID, Status: status})
			continue
		}
		results = append(results, o.check(ctx, profile, providerID))
	}
	return results
}

// Start runs the automation at once, then every interval and whenever wake
// delivers, until ctx ends or the returned stop runs. stop waits for a run
// in progress to end and may be called more than once. An interval of zero
// or less uses DefaultRunInterval, and a nil wake never delivers.
func (o *Orchestrator) Start(ctx context.Context, interval time.Duration, wake <-chan struct{}) (stop func()) {
	if interval <= 0 {
		interval = DefaultRunInterval
	}
	ctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			o.RunOnce(ctx)
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			case <-wake:
			}
		}
	}()
	var once sync.Once
	return func() { once.Do(func() { cancel(); <-done }) }
}

// check connects one managed provider with an anonymous connection and
// records the result under a fresh evidence generation.
func (o *Orchestrator) check(ctx context.Context, profile providers.AnonymousProviderProfile, providerID string) Result {
	generation, err := o.hooks.Generation(ctx, providerID)
	if err != nil {
		return Result{
			ProviderID: providerID, RegistryID: profile.RegistryID, Operation: OperationVerify,
			Status: StatusFailed, FailureCode: FailureEvidenceUnavailable,
			CatalogEvidence: core.CatalogNotProbed, CompletionEvidence: core.CompletionNotProbed,
		}
	}
	connector := core.NewProviderOrchestrator()
	connection := core.ProviderConnection{
		ProviderID: providerID, Kind: core.ProviderConnectionAnonymous, AuthKind: core.ProviderAuthAnonymous,
	}
	var connected core.ProviderConnectResult
	err = connector.Register(providerID, o.adapter(profile, providerID))
	if err == nil {
		connected, err = connector.Connect(ctx, core.ProviderConnectRequest{Connection: connection, PublicationPolicy: o.publish})
	}
	result := resultOf(profile, providerID, connected, err)
	if result.RecordErr = ctx.Err(); result.RecordErr == nil {
		result.RecordErr = o.hooks.Record(ctx, providerID, generation, result)
	}
	return result
}
