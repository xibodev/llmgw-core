package core

import (
	"errors"
)

// Disposition is what a failure permits a router to do next.
type Disposition string

const (
	// DispositionRetryable permits retrying the same target, and failing over.
	DispositionRetryable Disposition = "retryable"
	// DispositionFailover permits trying the next target but not repeating
	// this one.
	DispositionFailover Disposition = "failover"
	// DispositionTerminal ends the request: no target would serve it.
	DispositionTerminal Disposition = "terminal"
)

// Disposition summarizes the classification.
func (c ProviderErrorClassification) Disposition() Disposition {
	switch {
	case c.Retryable:
		return DispositionRetryable
	case c.FailoverEligible:
		return DispositionFailover
	}
	return DispositionTerminal
}

// Error classes beyond the upstream causes that health classification reports.
const (
	// ProviderErrorConfiguration is a provider that cannot run as configured.
	ProviderErrorConfiguration ProviderErrorClass = "configuration"
	// ProviderErrorInvalidRequest is a request the provider rejects as
	// malformed; other targets would reject it too.
	ProviderErrorInvalidRequest ProviderErrorClass = "invalid_request"
	// ProviderErrorUnsupported is a request this target cannot serve, such as
	// a surface it lacks or a translation the loss policy rejects.
	ProviderErrorUnsupported ProviderErrorClass = "unsupported"
)

// ProviderError is the canonical provider failure.
//
// Message is safe to show a client: it never contains credentials, private
// endpoints or upstream response bodies. Cause keeps the underlying error for
// errors.Is and errors.As and is never rendered.
type ProviderError struct {
	Message        string
	Class          ProviderErrorClass
	Classification ProviderErrorClassification
	Cause          error
}

func (e *ProviderError) Error() string {
	if e == nil || e.Message == "" {
		return "provider operation failed"
	}
	return e.Message
}

func (e *ProviderError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

// ProviderErrorClassification returns the routing metadata.
func (e *ProviderError) ProviderErrorClassification() ProviderErrorClassification {
	if e == nil {
		return ProviderErrorClassification{}
	}
	return e.Classification
}

// NewConfigurationError reports a provider that cannot run as configured. It
// permits failover, because another target may be configured correctly, but
// never marks the provider unhealthy.
func NewConfigurationError(message string, cause error) *ProviderError {
	return &ProviderError{
		Message:        message,
		Class:          ProviderErrorConfiguration,
		Classification: ProviderErrorClassification{FailoverEligible: true},
		Cause:          cause,
	}
}

// ClassifyError returns the routing metadata of any error. An error that does
// not classify itself, including a canceled context, is terminal.
func ClassifyError(err error) ProviderErrorClassification {
	var classifier ProviderErrorClassifier
	if errors.As(err, &classifier) {
		return classifier.ProviderErrorClassification()
	}
	return ProviderErrorClassification{}
}
