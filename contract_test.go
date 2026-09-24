package core_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"testing"
	"time"

	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
)

func TestCallerValidation(t *testing.T) {
	t.Parallel()
	if err := core.LocalCaller().Validate(); err != nil {
		t.Fatalf("the local caller is invalid: %v", err)
	}
	cases := map[string]struct {
		caller core.Caller
		valid  bool
	}{
		"human":                {core.Caller{ID: "user-1", Kind: core.CallerHuman}, true},
		"service in project":   {core.Caller{ID: "key-1", Kind: core.CallerService, ProjectID: "p"}, true},
		"anonymous without id": {core.Caller{Kind: core.CallerAnonymous}, true},
		"human without id":     {core.Caller{Kind: core.CallerHuman}, false},
		"local without id":     {core.Caller{Kind: core.CallerLocal}, false},
		"unknown kind":         {core.Caller{ID: "x", Kind: "robot"}, false},
		"empty kind":           {core.Caller{ID: "x"}, false},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if err := tc.caller.Validate(); (err == nil) != tc.valid {
				t.Fatalf("Validate()=%v, want valid=%v", err, tc.valid)
			}
		})
	}
}

func TestPrincipalConvertsToCaller(t *testing.T) {
	t.Parallel()
	for principalType, kind := range map[string]core.CallerKind{
		"user": core.CallerHuman, "api_key": core.CallerService, "service": core.CallerService,
		"anonymous": core.CallerAnonymous, "": core.CallerAnonymous, "HUMAN": core.CallerHuman,
	} {
		caller := (&core.Principal{ID: "id", Type: principalType, ProjectID: "project"}).Caller()
		if caller.Kind != kind || caller.ID != "id" || caller.ProjectID != "project" {
			t.Fatalf("type %q converted to %+v, want kind %q", principalType, caller, kind)
		}
	}
	var missing *core.Principal
	if caller := missing.Caller(); caller.Kind != core.CallerAnonymous {
		t.Fatalf("nil principal converted to %+v", caller)
	}
}

func TestModelSurfacesAreKnownAndCapabilitiesStayUnknown(t *testing.T) {
	t.Parallel()
	surfaces := []core.ModelSurface{
		core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses, core.ModelSurfaceMessages,
		core.ModelSurfaceEmbeddings, core.ModelSurfaceAudioTranscriptions, core.ModelSurfaceAudioSpeech,
		core.ModelSurfaceImages, core.ModelSurfaceVideos,
	}
	for _, surface := range surfaces {
		if !core.KnownModelSurface(surface) {
			t.Fatalf("%s is not known", surface)
		}
	}
	if core.KnownModelSurface("telepathy") {
		t.Fatal("an undefined surface is known")
	}
	if support := (core.ModelCapabilities{}).SurfaceCompatibility(core.ModelSurfaceImages); support != core.SupportUnknown {
		t.Fatalf("a surface outside the capabilities schema reported %q", support)
	}
}

func TestRequestValidation(t *testing.T) {
	t.Parallel()
	cases := map[string]struct {
		request core.Request
		valid   bool
	}{
		"json chat": {core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m", Body: []byte(`{}`), ContentType: core.ContentTypeJSON}, true},
		"multipart transcription": {core.Request{Surface: core.ModelSurfaceAudioTranscriptions, Model: "whisper",
			Body: []byte("--b\r\n"), ContentType: "multipart/form-data; boundary=b"}, true},
		"empty body":        {core.Request{Surface: core.ModelSurfaceEmbeddings, Model: "m"}, true},
		"unknown surface":   {core.Request{Surface: "telepathy", Model: "m"}, false},
		"missing model":     {core.Request{Surface: core.ModelSurfaceResponses, Model: " "}, false},
		"body without type": {core.Request{Surface: core.ModelSurfaceMessages, Model: "m", Body: []byte(`{}`)}, false},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if err := tc.request.Validate(); (err == nil) != tc.valid {
				t.Fatalf("Validate()=%v, want valid=%v", err, tc.valid)
			}
		})
	}
}

// chatOnlyProvider serves Chat Completions natively and rejects every other
// surface, as the contract requires.
type chatOnlyProvider struct{}

func (chatOnlyProvider) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

func (p chatOnlyProvider) Invoke(_ context.Context, request core.Request) (core.Response, error) {
	if !core.ServesNatively(p, request.Model, request.Surface) {
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	return core.Response{Body: []byte(`{"choices":[]}`), ContentType: core.ContentTypeJSON}, nil
}

func (p chatOnlyProvider) Stream(ctx context.Context, request core.Request) (core.StreamIter, error) {
	if _, err := p.Invoke(ctx, request); err != nil {
		return nil, err
	}
	return &lossyStream{}, nil
}

func (chatOnlyProvider) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	return []core.ModelInfo{{ID: "m"}}, nil
}

type lossyStream struct{ losses []core.Loss }

func (s *lossyStream) Next() ([]byte, error) { return nil, io.EOF }
func (s *lossyStream) Close() error          { return nil }
func (s *lossyStream) Losses() []core.Loss   { return s.losses }

type plainStream struct{}

func (plainStream) Next() ([]byte, error) { return nil, io.EOF }
func (plainStream) Close() error          { return nil }

func TestProviderContractRejectsNonNativeSurfaces(t *testing.T) {
	t.Parallel()
	var provider core.Provider = chatOnlyProvider{}
	if !core.ServesNatively(provider, "m", core.ModelSurfaceChatCompletions) || core.ServesNatively(provider, "m", core.ModelSurfaceResponses) {
		t.Fatal("ServesNatively disagrees with NativeSurfaces")
	}
	_, err := provider.Invoke(context.Background(), core.Request{Surface: core.ModelSurfaceResponses, Model: "m"})
	var surfaceErr *core.SurfaceError
	if !errors.As(err, &surfaceErr) || surfaceErr.Surface != core.ModelSurfaceResponses {
		t.Fatalf("error=%v, want a SurfaceError", err)
	}
	if got := core.ClassifyError(fmt.Errorf("wrapped: %w", err)); got.Disposition() != core.DispositionFailover || got.CircuitFailure {
		t.Fatalf("surface error classification=%+v, want failover without a circuit failure", got)
	}
}

func TestStreamLosses(t *testing.T) {
	t.Parallel()
	losses := []core.Loss{{Path: "temperature", Class: translate.LossDropped, Severity: translate.LossMaterial}}
	if got := core.StreamLosses(&lossyStream{losses: losses}); !slices.Equal(got, losses) {
		t.Fatalf("losses=%v", got)
	}
	if got := core.StreamLosses(plainStream{}); got != nil {
		t.Fatalf("a stream that does not translate reported %v", got)
	}
}

func TestDispositionAndClassifyError(t *testing.T) {
	t.Parallel()
	cases := map[string]struct {
		err  error
		want core.Disposition
	}{
		"retryable":     {&core.ProviderError{Classification: core.ProviderErrorClassification{Retryable: true, FailoverEligible: true}}, core.DispositionRetryable},
		"failover":      {&core.ProviderError{Classification: core.ProviderErrorClassification{FailoverEligible: true}}, core.DispositionFailover},
		"terminal":      {&core.ProviderError{Class: core.ProviderErrorInvalidRequest}, core.DispositionTerminal},
		"plain error":   {errors.New("boom"), core.DispositionTerminal},
		"canceled":      {context.Canceled, core.DispositionTerminal},
		"configuration": {core.NewConfigurationError("provider 'x' has no API key", nil), core.DispositionFailover},
		"loss policy":   {&core.LossPolicyError{}, core.DispositionFailover},
		"upstream 503":  {core.NewProviderOperationError("chat", 503, "", nil), core.DispositionRetryable},
		"upstream 400":  {core.NewProviderOperationError("chat", 400, "", nil), core.DispositionTerminal},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if got := core.ClassifyError(tc.err).Disposition(); got != tc.want {
				t.Fatalf("disposition=%q, want %q", got, tc.want)
			}
		})
	}
}

func TestProviderErrorKeepsCauseOutOfItsMessage(t *testing.T) {
	t.Parallel()
	cause := errors.New("dial https://private.example.test/?key=secret")
	err := &core.ProviderError{Message: "provider unavailable", Class: core.ProviderErrorTransport, Cause: cause}
	if err.Error() != "provider unavailable" || !errors.Is(err, cause) {
		t.Fatalf("message=%q is=%v", err.Error(), errors.Is(err, cause))
	}
	configuration := core.NewConfigurationError("provider 'x' has no API key", cause)
	if configuration.Class != core.ProviderErrorConfiguration || configuration.ProviderErrorClassification().CircuitFailure {
		t.Fatalf("configuration error=%+v", configuration)
	}
	var missing *core.ProviderError
	if missing.Error() == "" || missing.Unwrap() != nil || missing.ProviderErrorClassification() != (core.ProviderErrorClassification{}) {
		t.Fatal("a nil ProviderError is not safe to use")
	}
}

func TestInvocationErrorReportsRetryAfter(t *testing.T) {
	t.Parallel()
	classification := core.ClassifyError(&providers.InvocationError{Msg: "rate limited", Status: 429, RetryAfter: 3 * time.Second})
	if classification.RetryAfter != 3*time.Second || classification.Disposition() != core.DispositionRetryable {
		t.Fatalf("classification=%+v", classification)
	}
}

func TestLossPolicyDefaults(t *testing.T) {
	t.Parallel()
	var policy core.LossPolicy
	cases := map[string]struct {
		loss core.Loss
		want core.LossAction
	}{
		"material":         {core.Loss{Path: "tools", Severity: translate.LossMaterial}, core.LossReject},
		"advisory":         {core.Loss{Path: "max_tokens", Severity: translate.LossAdvisory}, core.LossAllow},
		"unknown severity": {core.Loss{Path: "x"}, core.LossReject},
	}
	for name, tc := range cases {
		if got := policy.Decide(tc.loss); got != tc.want {
			t.Fatalf("%s: %q, want %q", name, got, tc.want)
		}
	}
}

func TestLossPolicyPathGlobs(t *testing.T) {
	t.Parallel()
	cases := []struct {
		pattern, path string
		match         bool
	}{
		{"temperature", "temperature", true},
		{"temperature", "temperature.value", false},
		{"messages.*.tool_calls", "messages.3.tool_calls", true},
		{"messages.*.tool_calls", "messages.tool_calls", false},
		{"messages.*.tool_calls.**", "messages.3.tool_calls", true},
		{"messages.*.tool_calls.**", "messages.3.tool_calls.0.function", true},
		{"**.thought_signature", "messages.3.tool_calls.0.extra_content.google.thought_signature", true},
		{"**.thought_signature", "thought_signature", true},
		{"**.thought_signature", "messages.3.thought_signature.x", false},
		{"**", "", true},
		{"**", "output.2", true},
		{"output.*", "output", false},
	}
	for _, tc := range cases {
		policy := core.LossPolicy{Rules: []core.LossRule{{Path: tc.pattern, Action: core.LossAllow}}}
		got := policy.Decide(core.Loss{Path: tc.path, Severity: translate.LossMaterial}) == core.LossAllow
		if got != tc.match {
			t.Fatalf("pattern %q path %q: match=%v, want %v", tc.pattern, tc.path, got, tc.match)
		}
	}
}

// TestLossPolicyExpressesPerFieldPreferences mirrors Facet Studio's Codex
// routing: dropped sampling parameters are acceptable, while tool calls,
// thought signatures and reasoning never are, whatever their severity.
func TestLossPolicyExpressesPerFieldPreferences(t *testing.T) {
	t.Parallel()
	policy := core.LossPolicy{Rules: []core.LossRule{
		{Path: "**.thought_signature", Action: core.LossReject},
		{Path: "**.tool_calls.**", Action: core.LossReject},
		{Path: "**.reasoning", Action: core.LossReject},
		{Path: "temperature", Class: translate.LossDropped, Action: core.LossAllow},
		{Path: "top_p", Class: translate.LossDropped, Action: core.LossAllow},
	}}
	if err := policy.Validate(); err != nil {
		t.Fatal(err)
	}
	losses := []core.Loss{
		{Path: "messages.1.tool_calls.0.extra_content.google.thought_signature", Class: translate.LossDropped, Severity: translate.LossAdvisory},
		{Path: "messages.1.tool_calls.0", Class: translate.LossApproximated, Severity: translate.LossAdvisory},
		{Path: "output.2.reasoning", Class: translate.LossDropped, Severity: translate.LossAdvisory},
		{Path: "temperature", Class: translate.LossDropped, Severity: translate.LossMaterial},
		{Path: "top_p", Class: translate.LossDropped, Severity: translate.LossMaterial},
		{Path: "max_tokens", Class: translate.LossRenamed, Severity: translate.LossAdvisory},
		{Path: "seed", Class: translate.LossDropped, Severity: translate.LossMaterial},
	}
	rejected := []string{}
	for _, loss := range policy.Rejected(losses) {
		rejected = append(rejected, loss.Path)
	}
	want := []string{
		"messages.1.tool_calls.0.extra_content.google.thought_signature", "messages.1.tool_calls.0",
		"output.2.reasoning", "seed",
	}
	if !slices.Equal(rejected, want) {
		t.Fatalf("rejected=%v, want %v", rejected, want)
	}
	err := policy.Check(losses)
	var policyErr *core.LossPolicyError
	if !errors.As(err, &policyErr) || len(policyErr.Losses) != 4 || err.Error() == "" {
		t.Fatalf("Check()=%v", err)
	}
	if policy.Check(losses[3:6]) != nil {
		t.Fatal("allowed losses were rejected")
	}
}

func TestLossPolicyFirstMatchWinsAndValidates(t *testing.T) {
	t.Parallel()
	policy := core.LossPolicy{Rules: []core.LossRule{
		{Path: "tools", Action: core.LossReject},
		{Path: "tools", Action: core.LossAllow},
	}}
	if policy.Decide(core.Loss{Path: "tools", Severity: translate.LossAdvisory}) != core.LossReject {
		t.Fatal("a later rule overrode an earlier match")
	}
	for name, rule := range map[string]core.LossRule{
		"unknown action":   {Path: "x", Action: "ignore"},
		"empty segment":    {Path: "messages..tools", Action: core.LossAllow},
		"unknown class":    {Class: "vanished", Action: core.LossAllow},
		"unknown severity": {Severity: "fatal", Action: core.LossAllow},
	} {
		if err := (core.LossPolicy{Rules: []core.LossRule{rule}}).Validate(); err == nil {
			t.Fatalf("%s was accepted", name)
		}
	}
}
