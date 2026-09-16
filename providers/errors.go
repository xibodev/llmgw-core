package providers

import (
	"errors"
	"strings"
)

// InvocationError represents an upstream provider call failure.
type InvocationError struct {
	Msg              string
	Status           int
	Retryable        bool
	FailoverEligible bool
	CircuitFailure   bool
}

func (e *InvocationError) Error() string {
	return e.Msg
}

// CatalogError represents a failure discovering or reading a provider catalog.
type CatalogError struct {
	Code   string
	Detail string
	Status int
}

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
	var ie *InvocationError
	if !errors.As(err, &ie) {
		return false
	}
	if ie.Status == 0 {
		return ie.Retryable
	}
	switch ie.Status {
	case 408, 429, 500, 502, 503, 504:
		return true
	default:
		return false
	}
}

// InvocationFailoverEligible selects provider failures that an ordered endpoint may advance past.
func InvocationFailoverEligible(err error) bool {
	var ie *InvocationError
	if !errors.As(err, &ie) {
		return false
	}
	if ie.FailoverEligible || ie.Status == 0 {
		return true
	}
	return InvocationRetryable(ie)
}
