package anonymous

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"strconv"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

// adapter connects one enrolled provider through the product's Catalog and
// Invoker, where providers.NewAnonymousOpenAICompatibleAdapter used its
// own HTTP client.
func (o *Orchestrator) adapter(profile providers.AnonymousProviderProfile, providerID string) core.ProviderAdapter {
	return core.ProviderAdapter{
		ValidateAuthentication: func(_ context.Context, connection core.ProviderConnection) error {
			if connection.Kind != core.ProviderConnectionAnonymous || connection.AuthKind != core.ProviderAuthAnonymous {
				return errors.New("anonymous: the automation connects anonymous connections only")
			}
			return nil
		},
		DiscoverModels: func(ctx context.Context, _ core.ProviderConnection) ([]core.ModelInfo, error) {
			models, err := o.catalog.Discover(ctx, o.caller, providerID)
			if err != nil {
				return nil, operationError("provider catalog", err)
			}
			return models, nil
		},
		SelectProbeTargets: func(_ core.ProviderConnection, models []core.ModelInfo) []core.Target {
			return o.probeTargets(profile, models)
		},
		Complete: func(ctx context.Context, _ core.ProviderConnection, target core.Target, payload map[string]any) (map[string]any, error) {
			return o.complete(ctx, providerID, target, payload)
		},
	}
}

// probeTargets selects the models a check probes. Each target leaves its
// provider to the connector, which fills in the id it publishes under.
func (o *Orchestrator) probeTargets(profile providers.AnonymousProviderProfile, models []core.ModelInfo) []core.Target {
	if o.probe == ProbeVerificationModel {
		if model := providers.AnonymousVerificationModel(profile.RegistryID, models); model != "" {
			return []core.Target{{Model: model}}
		}
		return nil
	}
	seen := make(map[string]bool, len(models))
	targets := make([]core.Target, 0, len(models))
	for _, model := range models {
		id := strings.TrimSpace(model.ID)
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		targets = append(targets, core.Target{Model: id})
	}
	return targets
}

// complete sends one probe as a non-streaming Chat Completions request:
// the connector's payload with the model set and streaming off.
func (o *Orchestrator) complete(ctx context.Context, providerID string, target core.Target, payload map[string]any) (map[string]any, error) {
	body := make(map[string]any, len(payload)+2)
	maps.Copy(body, payload)
	body["model"], body["stream"] = target.Model, false
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, core.NewConfigurationError("anonymous: the probe could not be encoded", err)
	}
	response, err := o.invoker.Invoke(ctx, o.caller, providerID, core.Request{
		Surface: core.ModelSurfaceChatCompletions, Model: target.Model, Body: encoded, ContentType: core.ContentTypeJSON,
	})
	if err != nil {
		return nil, operationError("provider completion", err)
	}
	var answer map[string]any
	if err := json.Unmarshal(response.Body, &answer); err != nil {
		return nil, core.NewProviderOperationError("provider response decode", http.StatusOK, "", err)
	}
	return answer, nil
}

// operationError keeps what health classification reads of err, its
// status and Retry-After, in the form core.ProviderOrchestrator reads. Its
// message names the operation only, so a result's details never quote an
// upstream.
func operationError(op string, err error) error {
	classification := core.ClassifyError(err)
	retryAfter := ""
	if classification.RetryAfter > 0 {
		seconds := (classification.RetryAfter + time.Second - 1) / time.Second
		retryAfter = strconv.FormatInt(int64(seconds), 10)
	}
	return core.NewProviderOperationError(op, classification.StatusCode, retryAfter, err)
}
