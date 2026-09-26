package extension

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	"github.com/xibodev/llmgw-core/oauthflow"
)

// OAuthDriver runs one daemon provider's sign-in for oauthflow.Service. It
// is a device driver and a code driver; which methods the provider accepts
// is in its ProviderInfo.OAuthMethods. Its errors carry the daemon's reason,
// which never holds token material.
type OAuthDriver struct {
	client   *Client
	provider string
}

var (
	_ oauthflow.DeviceDriver = (*OAuthDriver)(nil)
	_ oauthflow.CodeDriver   = (*OAuthDriver)(nil)
)

// OAuthDriver returns the sign-in driver of provider.
func (c *Client) OAuthDriver(provider string) *OAuthDriver {
	return &OAuthDriver{client: c, provider: provider}
}

// Start begins a sign-in with the request's method, redirect URI and
// parameters.
func (d *OAuthDriver) Start(ctx context.Context, request oauthflow.StartRequest) (oauthflow.Authorization, error) {
	var answer OAuthStartResponse
	err := d.call(ctx, "start", OAuthStartRequest{
		Method:      request.Method,
		RedirectURI: request.RedirectURI,
		Params:      request.Params,
	}, &answer, func() string { return answer.Error })
	if err != nil {
		return oauthflow.Authorization{}, err
	}
	return answer.Authorization, nil
}

// Poll asks once whether the owner approved a device sign-in. The flow,
// secrets included, travels to the daemon, which keeps none.
func (d *OAuthDriver) Poll(ctx context.Context, flow oauthflow.Flow) (oauthflow.PollResult, error) {
	var answer OAuthPollResponse
	if err := d.call(ctx, "poll", OAuthPollRequest{Flow: flow}, &answer, func() string { return answer.Error }); err != nil {
		return oauthflow.PollResult{}, err
	}
	return answer.Result, nil
}

// Exchange trades the authorization code of a browser or manual sign-in for
// a credential.
func (d *OAuthDriver) Exchange(ctx context.Context, flow oauthflow.Flow, code string) (tokenstore.Record, error) {
	var answer OAuthExchangeResponse
	if err := d.call(ctx, "exchange", OAuthExchangeRequest{Code: code, Flow: flow}, &answer, func() string { return answer.Error }); err != nil {
		return tokenstore.Record{}, err
	}
	return answer.Record, nil
}

// call posts one oauth step and decodes its answer into out. reason reads
// the answer's error after decoding.
func (d *OAuthDriver) call(ctx context.Context, step string, in, out any, reason func() string) error {
	status, body, err := d.client.postJSON(ctx, d.provider, "oauth/"+step, in)
	if err != nil {
		return err
	}
	decodeErr := json.Unmarshal(body, out)
	succeededStatus := status >= 200 && status <= 299
	if succeededStatus && decodeErr == nil && reason() == "" {
		return nil
	}
	message := ""
	if decodeErr == nil {
		message = reason()
	}
	if message == "" {
		message = errorMessage(body)
	}
	text := "extension " + d.provider + " sign-in " + step + " failed"
	if message = safeMessage(message); message != "" {
		return errors.New(text + ": " + message)
	}
	if !succeededStatus {
		return fmt.Errorf("%s (HTTP %d)", text, status)
	}
	return errors.New(text + ": the extension's answer is invalid")
}
