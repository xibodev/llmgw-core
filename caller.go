package core

import (
	"fmt"
	"strings"
)

// CallerKind classifies who is making a request.
type CallerKind string

const (
	// CallerHuman is a signed-in person.
	CallerHuman CallerKind = "human"
	// CallerService is an automation identity, such as an API key.
	CallerService CallerKind = "service"
	// CallerAnonymous is an unauthenticated request.
	CallerAnonymous CallerKind = "anonymous"
	// CallerLocal is the single user of a local, single-user application.
	CallerLocal CallerKind = "local"
)

// Caller identifies who a request acts for. Products map their own identity
// types onto it; core never needs more than these fields. ProjectID is
// optional.
type Caller struct {
	ID        string     `json:"id"`
	Kind      CallerKind `json:"kind"`
	ProjectID string     `json:"project_id,omitempty"`
}

// LocalCaller is the caller of a single-user application. It is a complete,
// valid Caller.
func LocalCaller() Caller { return Caller{ID: "local", Kind: CallerLocal} }

// Validate reports a caller with no identity or an unknown kind. An anonymous
// caller may have an empty ID.
func (c Caller) Validate() error {
	switch c.Kind {
	case CallerHuman, CallerService, CallerLocal:
		if strings.TrimSpace(c.ID) == "" {
			return fmt.Errorf("caller of kind %q has no id", c.Kind)
		}
	case CallerAnonymous:
	default:
		return fmt.Errorf("caller has unknown kind %q", c.Kind)
	}
	return nil
}

// Caller converts a Principal to its Caller.
//
// Deprecated: Principal is replaced by Caller and will be removed in the next
// minor release. Its Type maps onto a CallerKind: "user" is human, "api_key"
// and "service" are service, and an empty or "anonymous" type is anonymous.
func (p *Principal) Caller() Caller {
	if p == nil {
		return Caller{Kind: CallerAnonymous}
	}
	kind := CallerKind(strings.ToLower(strings.TrimSpace(p.Type)))
	switch kind {
	case "user":
		kind = CallerHuman
	case "api_key":
		kind = CallerService
	case "":
		kind = CallerAnonymous
	}
	return Caller{ID: p.ID, Kind: kind, ProjectID: p.ProjectID}
}
