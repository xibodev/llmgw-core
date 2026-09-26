package providers

import (
	"context"
	"errors"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// InvocationError represents an upstream provider call failure.
type InvocationError struct {
	Msg               string
	Status            int
	Retryable         bool
	FailoverEligible  bool
	CircuitFailure    bool
	RetryAfter        time.Duration
	Cause             error
	upstreamTransport bool
	// class says what failed, for a provider that reports the canonical
	// *core.ProviderError. A status class comes from Status instead.
	class core.ProviderErrorClass
}

func (e *InvocationError) Unwrap() error { return e.Cause }

func (e *InvocationError) Error() string {
	return e.Msg
}

func (e *InvocationError) ProviderErrorClassification() core.ProviderErrorClassification {
	if e == nil {
		return core.ProviderErrorClassification{}
	}
	if callerCancellation(e) && !e.upstreamTransport {
		return core.ProviderErrorClassification{StatusCode: e.Status}
	}
	retryable := e.Retryable
	failoverEligible := e.FailoverEligible
	circuitFailure := e.CircuitFailure
	if e.Status != 0 {
		health := core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: e.Status})
		retryable = retryable || health.Retryable
		failoverEligible = failoverEligible || health.Retryable
		circuitFailure = circuitFailure || (e.Status >= 500 && e.Status < 600 && health.Retryable)
	}
	if e.upstreamTransport {
		retryable = true
		failoverEligible = true
	}
	return core.ProviderErrorClassification{
		StatusCode:       e.Status,
		Retryable:        retryable,
		FailoverEligible: failoverEligible,
		CircuitFailure:   circuitFailure,
		RetryAfter:       e.RetryAfter,
	}
}

func invocationError(ctx context.Context, msg string, status int, cause error) error {
	err := &InvocationError{Msg: msg, Status: status, Cause: cause}
	if status == 0 && cause != nil && ctx.Err() == nil {
		err.upstreamTransport = true
		err.CircuitFailure = true
	}
	return err
}

// CatalogError represents a failure discovering or reading a provider catalog.
type CatalogError struct {
	Code       string
	Detail     string
	Status     int
	RetryAfter time.Duration
	Cause      error
}

func (e *CatalogError) Unwrap() error { return e.Cause }

func (e *CatalogError) Error() string {
	return e.Detail
}

// IsThrottle inspects an error message or status for rate limit signals.
func IsThrottle(err error) bool {
	if err == nil {
		return false
	}
	var ie *InvocationError
	if errors.As(err, &ie) && ie.Status == 429 {
		return true
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "429") || strings.Contains(s, "throttl") ||
		strings.Contains(s, "rate limit") || strings.Contains(s, "too many")
}

// InvocationRetryable returns true if an invocation error is safe to retry on the same target.
func InvocationRetryable(err error) bool {
	classification, ok := providerErrorClassification(err)
	return ok && classification.Retryable
}

// InvocationFailoverEligible selects provider failures that an ordered endpoint may advance past.
func InvocationFailoverEligible(err error) bool {
	classification, ok := providerErrorClassification(err)
	return ok && classification.FailoverEligible
}

// InvocationCircuitFailure reports failures that should count against a target circuit.
func InvocationCircuitFailure(err error) bool {
	classification, ok := providerErrorClassification(err)
	return ok && classification.CircuitFailure
}

func providerErrorClassification(err error) (core.ProviderErrorClassification, bool) {
	var classified core.ProviderErrorClassifier
	if !errors.As(err, &classified) {
		return core.ProviderErrorClassification{}, false
	}
	return classified.ProviderErrorClassification(), true
}

func callerCancellation(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}
