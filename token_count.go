package core

import (
	"context"
	"errors"
	"net/http"
)

// ErrTokenCountUnsupported reports a provider that cannot count a request's
// tokens natively. A product may estimate the count instead, as the gateway
// does for a target that cannot count.
var ErrTokenCountUnsupported = errors.New("core: the provider cannot count tokens natively")

// TokenCountRequest asks how many input tokens a request would use.
//
// Request is the operation as it would be invoked, credential included;
// Anthropic counts a Messages body. Header holds the protocol headers the
// client sent that select how the upstream counts, such as
// anthropic-version and anthropic-beta. A provider forwards only the
// headers its protocol defines, after checking their values, and never
// authenticates with one.
type TokenCountRequest struct {
	Request
	Header http.Header
}

// TokenCount is an upstream's count of a request's input tokens.
type TokenCount struct {
	InputTokens int64
}

// TokenCounter is an optional Provider interface for the token_count
// operation: counting a request's input tokens without running it, as
// Anthropic's /v1/messages/count_tokens does. A request it cannot count,
// such as one for another surface, fails with ErrTokenCountUnsupported.
type TokenCounter interface {
	CountTokens(ctx context.Context, request TokenCountRequest) (TokenCount, error)
}

// CountTokens counts request's input tokens through the nearest
// TokenCounter, looking through decorators that unwrap as PreservesWire
// does, and fails with ErrTokenCountUnsupported when there is none. A
// decorator that applies its own policy to counts, such as retries,
// implements TokenCounter itself and forwards through CountTokens.
func CountTokens(ctx context.Context, provider Provider, request TokenCountRequest) (TokenCount, error) {
	for provider != nil {
		if counter, ok := provider.(TokenCounter); ok {
			return counter.CountTokens(ctx, request)
		}
		provider = unwrapProvider(provider)
	}
	return TokenCount{}, ErrTokenCountUnsupported
}
