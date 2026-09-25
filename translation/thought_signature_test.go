package translation_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	translate "github.com/xibodev/llm-translate"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/translation"
)

// A Chat history in which Gemini called tools, with the signature at each
// location a Gemini provider puts it.
const signedChatRequest = `{"model":"m","messages":[
{"role":"user","content":"look it up"},
{"role":"assistant","tool_calls":[
{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"},"extra_content":{"google":{"thought_signature":"sig-a"}}},
{"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{}","thought_signature":"sig-b"}},
{"id":"call_3","type":"function","function":{"name":"lookup","arguments":"{}"},"thought_signature":"sig-c"}]},
{"role":"tool","tool_call_id":"call_1","content":"found"},
{"role":"tool","tool_call_id":"call_2","content":"found"},
{"role":"tool","tool_call_id":"call_3","content":"found"}]}`

// A Gemini Chat completion that calls a tool.
const signedChatCompletion = `{"id":"chatcmpl-1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"},"extra_content":{"google":{"thought_signature":"sig-a"}}}]},"finish_reason":"tool_calls"}]}`

type signatureCase struct {
	native   core.ModelSurface
	response string
	request  core.Request
	paths    []string // the signature losses, in report order
	calls    int32    // provider calls before the policy decides
}

// signatureCases are the translations that drop a Gemini thought signature.
// A request loss is decided before the provider is called, a response loss
// after it answered.
func signatureCases() map[string]signatureCase {
	responsePath := []string{"choices.0.message.tool_calls.0.extra_content.google.thought_signature"}
	return map[string]signatureCase{
		"Chat request over Responses": {
			native: core.ModelSurfaceResponses, response: responsesObject,
			request: jsonRequest(core.ModelSurfaceChatCompletions, signedChatRequest),
			paths: []string{
				"messages.1.tool_calls.0.extra_content.google.thought_signature",
				"messages.1.tool_calls.1.function.thought_signature",
				"messages.1.tool_calls.2.thought_signature",
			},
		},
		"Chat response for a Messages client": {
			native: core.ModelSurfaceChatCompletions, response: signedChatCompletion,
			request: jsonRequest(core.ModelSurfaceMessages, `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"look it up"}]}`),
			paths:   responsePath, calls: 1,
		},
		"Chat response for a Responses client": {
			native: core.ModelSurfaceChatCompletions, response: signedChatCompletion,
			request: jsonRequest(core.ModelSurfaceResponses, `{"model":"m","input":"look it up"}`),
			paths:   responsePath, calls: 1,
		},
	}
}

// hasSignatureLosses reports whether losses hold every path as a material
// dropped thought signature.
func hasSignatureLosses(losses []core.Loss, paths []string) bool {
	for _, path := range paths {
		if !slices.ContainsFunc(losses, func(loss core.Loss) bool {
			return loss.Path == path && loss.Class == translate.LossDropped && loss.Severity == translate.LossMaterial
		}) {
			return false
		}
	}
	return true
}

func lossPaths(losses []core.Loss) []string {
	paths := make([]string, len(losses))
	for index, loss := range losses {
		paths[index] = loss.Path
	}
	return paths
}

// The default policy rejects an unmatched material loss, so an adapter that
// would drop a thought signature refuses, and another target may serve.
func TestDroppedThoughtSignaturesAreRejectedByDefault(t *testing.T) {
	t.Parallel()
	for name, tc := range signatureCases() {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			provider := &fakeProvider{native: tc.native, response: tc.response}
			response, err := translation.Adapter{Provider: provider}.Invoke(context.Background(), tc.request)
			var policyErr *core.LossPolicyError
			if !errors.As(err, &policyErr) || !slices.Equal(lossPaths(policyErr.Losses), tc.paths) {
				t.Fatalf("err=%v, want exactly the thought signatures %v rejected", err, tc.paths)
			}
			if disposition := core.ClassifyError(err).Disposition(); disposition != core.DispositionFailover || provider.calls.Load() != tc.calls {
				t.Fatalf("disposition=%v provider calls=%d, want failover after %d calls", disposition, provider.calls.Load(), tc.calls)
			}
			if response.Body != nil || !hasSignatureLosses(response.Losses, tc.paths) {
				t.Fatalf("body=%s losses=%v, want no body and the losses reported", response.Body, response.Losses)
			}
		})
	}
}

// An allow rule on **.thought_signature, the rule a product serving Gemini
// through the adapter considers, restores serving and still reports the loss.
func TestAllowingThoughtSignaturesServesAndStillReports(t *testing.T) {
	t.Parallel()
	policy := core.LossPolicy{Rules: []core.LossRule{
		{Path: "**.thought_signature", Class: translate.LossDropped, Action: core.LossAllow},
	}}
	if err := policy.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, tc := range signatureCases() {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			provider := &fakeProvider{native: tc.native, response: tc.response}
			response, err := translation.Adapter{Provider: provider, Policy: policy}.Invoke(context.Background(), tc.request)
			if err != nil || len(response.Body) == 0 || provider.calls.Load() != 1 {
				t.Fatalf("err=%v body=%s provider calls=%d, want the translation served", err, response.Body, provider.calls.Load())
			}
			if !hasSignatureLosses(response.Losses, tc.paths) {
				t.Fatalf("losses=%v, want the allowed losses %v reported", response.Losses, tc.paths)
			}
		})
	}
}

// A stream's response losses are reported, never enforced, because its
// frames are already delivered, so a Messages stream over Chat serves
// Gemini's signed tool calls under the default policy.
func TestMessagesStreamReportsDroppedThoughtSignatures(t *testing.T) {
	t.Parallel()
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, frames: []string{
		`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"},"extra_content":{"google":{"thought_signature":"sig-a"}}}]}}]}` + "\n\n",
		`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n",
		"data: [DONE]\n\n",
	}}
	stream, err := translation.Adapter{Provider: provider}.Stream(context.Background(), jsonRequest(core.ModelSurfaceMessages,
		`{"model":"m","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"look it up"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	frames, err := collect(t, stream)
	if joined := strings.Join(frames, ""); err != nil || !strings.Contains(joined, `"tool_use"`) || !strings.Contains(joined, "message_stop") {
		t.Fatalf("err=%v stream:\n%s", err, joined)
	}
	path := []string{"choices.0.delta.tool_calls.0.extra_content.google.thought_signature"}
	if losses := core.StreamLosses(stream); !hasSignatureLosses(losses, path) {
		t.Fatalf("losses=%v, want the signature loss reported", losses)
	}
}
