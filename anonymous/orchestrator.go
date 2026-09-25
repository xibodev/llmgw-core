package anonymous

import (
	"errors"
	"fmt"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

// DefaultCheckInterval is how often one provider is checked when
// Options.CheckInterval is zero: once a day, as the gateway checks.
const DefaultCheckInterval = 24 * time.Hour

// DefaultRunInterval is how often Start runs when it is given no
// interval: hourly, as the gateway runs.
const DefaultRunInterval = time.Hour

// ProbePolicy chooses which discovered models a check probes.
type ProbePolicy int

const (
	// ProbeAllDiscovered probes every model the catalog lists once, as the
	// gateway's automation does, so each model publishes on its own
	// evidence and no model is verified by a sibling's answer.
	ProbeAllDiscovered ProbePolicy = iota
	// ProbeVerificationModel probes only the model
	// providers.AnonymousVerificationModel selects: the first reviewed
	// default the catalog lists, or the lexically first model.
	ProbeVerificationModel
)

// Options configures an Orchestrator. Catalog, Invoker and Hooks are
// required.
type Options struct {
	// Profiles are the providers the automation connects. Nil connects
	// every anonymous profile of Registry.
	Profiles []providers.AnonymousProviderProfile
	// Registry is the product's effective registry, which vets every
	// profile. Nil uses providers.DefaultRegistry.
	Registry *providers.Registry
	Catalog  Catalog
	Invoker  Invoker
	Hooks    Hooks
	// Caller is who the checks act for, for Catalog and Invoker. The zero
	// Caller is anonymous.
	Caller core.Caller
	// Probe chooses the models a check probes.
	Probe ProbePolicy
	// Publish decides which targets a check publishes. Empty publishes
	// only targets whose probe verified inference, as the gateway does.
	Publish core.TargetPublicationPolicy
	// CheckInterval is how often one provider is checked: the every of
	// Claim. Zero uses DefaultCheckInterval.
	CheckInterval time.Duration
	// Now returns the current time, for Claim. Nil uses time.Now.
	Now func() time.Time
}

// Orchestrator runs the anonymous-provider automation. It keeps no state
// between runs, so it is safe for concurrent use.
type Orchestrator struct {
	profiles []providers.AnonymousProviderProfile
	catalog  Catalog
	invoker  Invoker
	hooks    Hooks
	caller   core.Caller
	probe    ProbePolicy
	publish  core.TargetPublicationPolicy
	every    time.Duration
	now      func() time.Time
}

// New returns an Orchestrator, or an error when a required option is
// missing or a profile is not one the registry curates for anonymous
// automation.
func New(options Options) (*Orchestrator, error) {
	if options.Catalog == nil || options.Invoker == nil || options.Hooks == nil {
		return nil, errors.New("anonymous: Catalog, Invoker and Hooks are required")
	}
	registry := options.Registry
	if registry == nil {
		registry = providers.DefaultRegistry()
	}
	profiles := options.Profiles
	if profiles == nil {
		profiles = registry.AnonymousProfiles()
	}
	for _, profile := range profiles {
		if err := eligible(registry, profile); err != nil {
			return nil, err
		}
	}
	o := &Orchestrator{
		profiles: append([]providers.AnonymousProviderProfile(nil), profiles...),
		catalog:  options.Catalog, invoker: options.Invoker, hooks: options.Hooks,
		caller: options.Caller, probe: options.Probe, publish: options.Publish,
		every: options.CheckInterval, now: options.Now,
	}
	if o.caller.Kind == "" {
		o.caller = core.Caller{Kind: core.CallerAnonymous}
	}
	if o.publish == "" {
		o.publish = core.PublishVerifiedTargets
	}
	if o.publish != core.PublishVerifiedTargets && o.publish != core.PublishDiscoveredTargets {
		return nil, fmt.Errorf("anonymous: unsupported target publication policy %q", o.publish)
	}
	if o.every <= 0 {
		o.every = DefaultCheckInterval
	}
	if o.now == nil {
		o.now = time.Now
	}
	return o, nil
}

// eligible admits only a profile of a registry entry curated for anonymous
// automation, over the OpenAI-compatible runtime, and never Codex, whose
// personal subscription must not fall through to anonymous access.
func eligible(registry *providers.Registry, profile providers.AnonymousProviderProfile) error {
	entry, ok := registry.ByID(profile.RegistryID)
	if !ok || !entry.AnonymousAutomation || entry.ID == "openai_codex" ||
		!strings.EqualFold(profile.RuntimeType, "openai_compatible") || strings.TrimSpace(profile.ProviderID) == "" {
		return fmt.Errorf("anonymous: provider profile %q is not eligible for anonymous automation", profile.RegistryID)
	}
	return nil
}
