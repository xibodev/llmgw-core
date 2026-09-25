package oauthflow

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strings"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
)

// Start begins a flow for caller on instance and stores it. The returned
// view shows the owner where to authorize; the flow's secrets stay in the
// store.
func (s *Service) Start(ctx context.Context, caller core.Caller, instance string, method Method, options ...StartOption) (View, error) {
	if err := caller.Validate(); err != nil {
		return View{}, err
	}
	instance = strings.TrimSpace(instance)
	if instance == "" {
		return View{}, errors.New("oauthflow: instance is required")
	}
	driver, err := s.driver(instance, method)
	if err != nil {
		return View{}, err
	}
	request := StartRequest{Caller: caller, Instance: instance, Method: method}
	for _, option := range options {
		option(&request)
	}
	authorization, err := driver.Start(ctx, request)
	if err != nil {
		return View{}, err
	}
	if err := checkAuthorization(method, authorization); err != nil {
		return View{}, err
	}
	id, err := NewID()
	if err != nil {
		return View{}, err
	}
	now := s.now()
	ttl := authorization.ExpiresIn
	if ttl <= 0 {
		ttl = DefaultTTL
	}
	secrets := authorization.Secrets
	secrets.DriverData = maps.Clone(secrets.DriverData)
	secrets.Params = request.Params
	if secrets.RedirectURI == "" {
		secrets.RedirectURI = request.RedirectURI
	}
	flow := Flow{
		ID: id, Caller: caller, Instance: instance, Method: method,
		CreatedAt: now, ExpiresAt: now.Add(ttl),
		AuthorizationURL: authorization.AuthorizationURL, UserCode: authorization.UserCode,
		VerificationURI: authorization.VerificationURI, VerificationURIComplete: authorization.VerificationURIComplete,
		Secrets: secrets,
	}
	if method == MethodDevice {
		interval := authorization.Interval
		if interval <= 0 {
			interval = defaultInterval
		}
		// The first poll waits a full interval, as the gateway enforces.
		flow.Progress = Progress{Interval: interval, NextPollAt: now.Add(interval)}
	}
	if err := s.store.Create(ctx, flow); err != nil {
		return View{}, err
	}
	return viewOf(flow), nil
}

// Get returns caller's flow. An expired flow returns ErrFlowExpired with a
// view whose status is StatusExpired.
func (s *Service) Get(ctx context.Context, caller core.Caller, id string) (View, error) {
	flow, err := s.store.Get(ctx, caller, id)
	if errors.Is(err, ErrFlowExpired) {
		return expiredView(id), err
	}
	if err != nil {
		return View{}, err
	}
	return viewOf(flow), nil
}

// driver resolves the driver for a method and checks it can run it.
func (s *Service) driver(instance string, method Method) (Driver, error) {
	switch method {
	case MethodBrowser, MethodDevice, MethodManual:
	default:
		return nil, fmt.Errorf("oauthflow: unknown method %q", method)
	}
	driver, err := s.drivers(instance, method)
	if err != nil {
		return nil, err
	}
	var ok bool
	if method == MethodDevice {
		_, ok = driver.(DeviceDriver)
	} else {
		_, ok = driver.(CodeDriver)
	}
	if driver == nil || !ok {
		return nil, fmt.Errorf("oauthflow: instance %q has no %s driver", instance, method)
	}
	return driver, nil
}

func checkAuthorization(method Method, authorization Authorization) error {
	secrets := authorization.Secrets
	if method == MethodDevice {
		if secrets.DeviceCode == "" || authorization.UserCode == "" || authorization.VerificationURI == "" {
			return errors.New("oauthflow: the device driver returned incomplete authorization data")
		}
		return nil
	}
	if authorization.AuthorizationURL == "" || secrets.State == "" || secrets.Verifier == "" {
		return errors.New("oauthflow: the driver returned incomplete authorization data")
	}
	return nil
}

// save stores a consumed flow's credential under the product's key and
// records the outcome.
func (s *Service) save(ctx context.Context, flow Flow, record tokenstore.Record) (View, error) {
	key, err := s.key(ctx, Completion{
		Caller: flow.Caller, Instance: flow.Instance, Method: flow.Method,
		Params: maps.Clone(flow.Secrets.Params), Record: record.Clone(),
	})
	if err == nil && strings.TrimSpace(key) == "" {
		err = errors.New("oauthflow: the credential key hook returned no key")
	}
	if err == nil {
		_, err = s.credentials.Save(ctx, key, record)
	}
	if err != nil {
		return s.finish(ctx, flow, OutcomeFailed, ""), fmt.Errorf("oauthflow: save credential: %w", err)
	}
	return s.finish(ctx, flow, OutcomeComplete, key), nil
}

// finish records how a consumed flow ended, so its owner can read it later.
// The flow is spent either way, so failing to record is not an error; the
// write outlives a canceled request.
func (s *Service) finish(ctx context.Context, flow Flow, outcome Outcome, key string) View {
	flow.Progress.Outcome, flow.Progress.CredentialKey = outcome, key
	flow.Secrets = Secrets{}
	if updated, err := s.store.Update(context.WithoutCancel(ctx), flow.Caller, flow); err == nil {
		flow = updated
	}
	return viewOf(flow)
}
