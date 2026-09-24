package oauthflow

import (
	"context"
	"crypto/subtle"
	"errors"
	"fmt"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// CompleteInput is what came back from the provider: a browser callback's
// query parameters, or what the owner pasted.
type CompleteInput struct {
	// Code is the authorization code.
	Code string
	// State is the OAuth state that came back with the code. A callback
	// always carries it and a pasted redirect URL does too; a pasted bare
	// code has none. When set, it must match the flow's.
	State string
	// Error is the provider's error code, such as access_denied, when the
	// owner refused. It ends the flow.
	Error string
	// RedirectURI is where a callback arrived. Callback refuses, without
	// consuming the flow, a callback that arrived anywhere but the flow's
	// redirect URI.
	RedirectURI string
}

// Complete finishes caller's browser or manual flow with a pasted code, a
// pasted redirect URL, or a callback the product routed itself.
//
// It consumes the flow before anything else can fail, so the code is used at
// most once even when the exchange fails: a mismatched state, a denial or a
// failed exchange ends the flow, and the owner starts again. Only an input
// with neither a code nor an error leaves the flow untouched.
func (s *Service) Complete(ctx context.Context, caller core.Caller, id string, input CompleteInput) (View, error) {
	input = trimInput(input)
	if input.Code == "" && input.Error == "" {
		return View{}, ErrCodeRequired
	}
	flow, err := s.store.Get(ctx, caller, id)
	if errors.Is(err, ErrFlowExpired) {
		return expiredView(id), err
	}
	if err != nil {
		return View{}, err
	}
	return s.complete(ctx, flow, input)
}

// Callback finishes the browser flow a provider redirect names by its OAuth
// state. It is the one operation without a caller: the redirect carries no
// session, and the state, which only ever appeared inside its owner's
// authorization URL, identifies the flow and its owner instead.
func (s *Service) Callback(ctx context.Context, input CompleteInput) (View, error) {
	input = trimInput(input)
	if input.State == "" {
		return View{}, ErrFlowNotFound
	}
	if input.Code == "" && input.Error == "" {
		return View{}, ErrCodeRequired
	}
	owner, id, err := s.store.ResolveState(ctx, input.State)
	if errors.Is(err, ErrFlowExpired) {
		return expiredView(id), err
	}
	if err != nil {
		return View{}, err
	}
	flow, err := s.store.Get(ctx, owner, id)
	if errors.Is(err, ErrFlowExpired) {
		return expiredView(id), err
	}
	if err != nil {
		return View{}, err
	}
	if flow.Method != MethodBrowser {
		return View{}, ErrWrongMethod
	}
	if input.RedirectURI != "" && input.RedirectURI != flow.Secrets.RedirectURI {
		return View{}, ErrFlowNotFound
	}
	return s.complete(ctx, flow, input)
}

func (s *Service) complete(ctx context.Context, flow Flow, input CompleteInput) (View, error) {
	if flow.Method == MethodDevice {
		return View{}, ErrWrongMethod
	}
	if flow.Consumed() {
		return View{}, ErrFlowNotFound
	}
	// Resolve the driver before consuming, so a product whose settings no
	// longer offer the instance does not spend the flow.
	driver, err := s.driver(flow.Instance, flow.Method)
	if err != nil {
		return View{}, err
	}
	consumed, err := s.store.Consume(ctx, flow.Caller, flow.ID)
	if errors.Is(err, ErrFlowExpired) {
		return expiredView(flow.ID), err
	}
	if err != nil {
		return View{}, err
	}
	if input.Error != "" {
		return s.finish(ctx, consumed, OutcomeFailed, ""), ErrAccessDenied
	}
	if input.State != "" && subtle.ConstantTimeCompare([]byte(consumed.Secrets.State), []byte(input.State)) != 1 {
		return s.finish(ctx, consumed, OutcomeFailed, ""), ErrStateMismatch
	}
	record, err := driver.(CodeDriver).Exchange(ctx, consumed, input.Code)
	if err != nil {
		return s.finish(ctx, consumed, OutcomeFailed, ""), fmt.Errorf("oauthflow: exchange %s: %w", consumed.Instance, err)
	}
	return s.save(ctx, consumed, record)
}

func trimInput(input CompleteInput) CompleteInput {
	input.Code = strings.TrimSpace(input.Code)
	input.State = strings.TrimSpace(input.State)
	input.Error = strings.TrimSpace(input.Error)
	input.RedirectURI = strings.TrimSpace(input.RedirectURI)
	return input
}
