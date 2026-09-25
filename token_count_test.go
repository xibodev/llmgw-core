package core_test

import (
	"context"
	"errors"
	"net/http"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// countingProvider counts the tokens of Messages requests and nothing else.
type countingProvider struct {
	surfaceProvider
	count    core.TokenCount
	received []core.TokenCountRequest
}

func (p *countingProvider) CountTokens(_ context.Context, request core.TokenCountRequest) (core.TokenCount, error) {
	if request.Surface != core.ModelSurfaceMessages {
		return core.TokenCount{}, core.ErrTokenCountUnsupported
	}
	p.received = append(p.received, request)
	return p.count, nil
}

// retryingDecorator applies a policy of its own to counts, so it counts
// through the provider it wraps itself.
type retryingDecorator struct {
	decorator
	calls *int
}

func (d retryingDecorator) CountTokens(ctx context.Context, request core.TokenCountRequest) (core.TokenCount, error) {
	*d.calls++
	return core.CountTokens(ctx, d.Provider, request)
}

func TestCountTokensUsesTheNearestCounter(t *testing.T) {
	ctx := context.Background()
	request := core.TokenCountRequest{
		Request: core.Request{
			Surface: core.ModelSurfaceMessages, Model: "m", ContentType: core.ContentTypeJSON,
			Body: []byte(`{"messages":[{"role":"user","content":"hi"}]}`),
		},
		Header: http.Header{"Anthropic-Version": {"2023-06-01"}},
	}
	counter := &countingProvider{count: core.TokenCount{InputTokens: 42}}
	calls := 0
	for name, provider := range map[string]core.Provider{
		"counter":             counter,
		"through a decorator": decorator{Provider: counter},
		"decorator with its own policy": retryingDecorator{
			decorator: decorator{Provider: counter}, calls: &calls,
		},
	} {
		t.Run(name, func(t *testing.T) {
			before := len(counter.received)
			count, err := core.CountTokens(ctx, provider, request)
			if err != nil || count.InputTokens != 42 || len(counter.received) != before+1 {
				t.Fatalf("count = %+v, err = %v, counter reached %d times", count, err, len(counter.received)-before)
			}
			if got := counter.received[before]; got.Model != "m" || got.Header.Get("anthropic-version") != "2023-06-01" {
				t.Fatalf("the counter received %+v", got)
			}
		})
	}
	if calls != 1 {
		t.Fatalf("the decorator with its own policy counted %d times, want 1", calls)
	}
}

func TestCountTokensWithoutACounterIsUnsupported(t *testing.T) {
	chat := core.TokenCountRequest{Request: core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m"}}
	for name, check := range map[string]struct {
		provider core.Provider
		request  core.TokenCountRequest
	}{
		"no counter":                   {surfaceProvider{}, chat},
		"decorator over no counter":    {decorator{Provider: surfaceProvider{}}, chat},
		"a surface it cannot count":    {&countingProvider{}, chat},
		"the same through a decorator": {decorator{Provider: &countingProvider{}}, chat},
		"no provider":                  {nil, chat},
	} {
		t.Run(name, func(t *testing.T) {
			count, err := core.CountTokens(context.Background(), check.provider, check.request)
			if !errors.Is(err, core.ErrTokenCountUnsupported) || count != (core.TokenCount{}) {
				t.Fatalf("count = %+v, err = %v, want ErrTokenCountUnsupported", count, err)
			}
			if disposition := core.ClassifyError(err).Disposition(); disposition != core.DispositionTerminal {
				t.Fatalf("disposition = %s, want terminal: the product estimates instead", disposition)
			}
		})
	}
}
