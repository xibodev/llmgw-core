package oauthflow_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	"github.com/xibodev/llmgw-core/oauthflow"
)

// Fixture secrets. A view must never show any of them, except the state
// inside AuthorizationURL.
const (
	secretVerifier = "fixture-verifier-value"
	secretState    = "fixture-state-value"
	secretDevice   = "fixture-device-code"
	secretDriver   = "fixture-driver-data"
	secretParam    = "fixture-client-secret"
	secretAccess   = "test-access-token"
	secretRefresh  = "test-refresh-token"
	secretIDToken  = "test-id-token"
	fixtureCode    = "fixture-authorization-code"
)

func allSecrets() []string {
	return []string{secretVerifier, secretState, secretDevice, secretDriver, secretParam, secretAccess, secretRefresh, secretIDToken, fixtureCode}
}

// fakeDriver scripts a provider. Poll returns the queued results in order,
// then pending; each start gets a distinct state.
type fakeDriver struct {
	mu          sync.Mutex
	starts      []oauthflow.StartRequest
	polls       []pollStep
	pollCalls   int
	exchanges   int
	exchangeErr error
	seen        []oauthflow.Flow
}

type pollStep struct {
	result oauthflow.PollResult
	err    error
}

func (d *fakeDriver) Start(_ context.Context, request oauthflow.StartRequest) (oauthflow.Authorization, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.starts = append(d.starts, request)
	n := len(d.starts)
	if request.Method == oauthflow.MethodDevice {
		return oauthflow.Authorization{
			UserCode: "WXYZ-1234", VerificationURI: "https://verify.example.test/device",
			VerificationURIComplete: "https://verify.example.test/device?user_code=WXYZ-1234",
			Interval:                5 * time.Second, ExpiresIn: 15 * time.Minute,
			Secrets: oauthflow.Secrets{DeviceCode: fmt.Sprintf("%s-%d", secretDevice, n), DriverData: map[string]string{"token": secretDriver}},
		}, nil
	}
	state := fmt.Sprintf("%s-%d", secretState, n)
	return oauthflow.Authorization{
		AuthorizationURL: "https://auth.example.test/authorize?code_challenge=fixture-challenge&state=" + state,
		Secrets: oauthflow.Secrets{
			Verifier: fmt.Sprintf("%s-%d", secretVerifier, n), State: state,
			DriverData: map[string]string{"token": secretDriver},
		},
	}, nil
}

func (d *fakeDriver) Poll(_ context.Context, flow oauthflow.Flow) (oauthflow.PollResult, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.pollCalls++
	d.seen = append(d.seen, flow)
	if len(d.polls) == 0 {
		return oauthflow.PollResult{Status: oauthflow.PollPending}, nil
	}
	step := d.polls[0]
	d.polls = d.polls[1:]
	return step.result, step.err
}

func (d *fakeDriver) Exchange(_ context.Context, flow oauthflow.Flow, code string) (tokenstore.Record, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.exchanges++
	d.seen = append(d.seen, flow)
	if d.exchangeErr != nil {
		return tokenstore.Record{}, d.exchangeErr
	}
	if code != fixtureCode || !strings.HasPrefix(flow.Secrets.Verifier, secretVerifier) {
		return tokenstore.Record{}, errors.New("fixture provider: invalid_grant")
	}
	return approvedRecord(), nil
}

func (d *fakeDriver) queue(steps ...pollStep) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.polls = append(d.polls, steps...)
}

func (d *fakeDriver) counts() (polls, exchanges int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.pollCalls, d.exchanges
}

func approvedRecord() tokenstore.Record {
	return tokenstore.Record{
		AccessToken: secretAccess, RefreshToken: secretRefresh, IDToken: secretIDToken,
		TokenType: "Bearer", AccountID: "fixture-account", Expiry: time.Unix(1_900_000_000, 0),
	}
}

func approved() pollStep {
	return pollStep{result: oauthflow.PollResult{Status: oauthflow.PollApproved, Record: approvedRecord()}}
}

func status(s oauthflow.PollStatus) pollStep {
	return pollStep{result: oauthflow.PollResult{Status: s}}
}
