package providers

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"slices"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// ollamaStream converts Ollama's NDJSON stream to Chat chunks as the
// gateway does, each an SSE frame, and ends it with [DONE] once Ollama
// reports it done or closes it. An event with neither content, tool calls
// nor done is skipped, as is a line that is not a JSON object. The event
// that reports done ends the stream: with content it is the last content
// chunk, and without it a chunk that finishes with stop. A line is bounded
// to maxStreamRecordWireSize.
type ollamaStream struct {
	ctx                  context.Context
	body                 io.ReadCloser
	reader               *bufio.Reader
	model                string
	losses               []core.Loss
	emitted, done, ended bool
	err                  error
}

var _ core.LossReporter = (*ollamaStream)(nil)

func newOllamaStream(ctx context.Context, body io.ReadCloser, model string, losses []core.Loss) *ollamaStream {
	return &ollamaStream{ctx: ctx, body: body, reader: bufio.NewReader(body), model: model, losses: losses}
}

func (s *ollamaStream) Next() ([]byte, error) {
	for s.err == nil && !s.done {
		line, err := readBoundedLine(s.reader, maxStreamRecordWireSize, "NDJSON")
		var tooLarge *streamRecordTooLargeError
		if errors.As(err, &tooLarge) {
			s.err = streamFailure(s.ctx, "Ollama", err)
			break
		}
		if frame := s.frame(line); frame != nil {
			return frame, nil
		}
		switch {
		case err == io.EOF && s.emitted:
			s.done = true
		case err == io.EOF:
			// The gateway reports a stream that ends before any content,
			// and before done, as an empty answer another target may serve.
			s.err = &core.ProviderError{
				Message: "Ollama returned an empty response", Class: core.ProviderErrorUpstream,
				Classification: core.ProviderErrorClassification{FailoverEligible: true},
			}
		case err != nil:
			// Unlike a relayed SSE stream, the gateway lets a broken Ollama
			// stream count against the provider and be repeated.
			s.err = transportFailure(s.ctx, "the Ollama stream broke off", err)
		}
	}
	switch {
	case s.err != nil:
		return nil, s.err
	case s.ended:
		return nil, io.EOF
	}
	s.ended = true
	return []byte("data: [DONE]\n\n"), nil
}

// frame converts one NDJSON line, or returns nil for a line with nothing to
// relay.
func (s *ollamaStream) frame(line []byte) []byte {
	text := strings.TrimSpace(string(line))
	var event map[string]any
	if text == "" || json.Unmarshal([]byte(text), &event) != nil {
		return nil
	}
	message, _ := event["message"].(map[string]any)
	content, _ := message["content"].(string)
	toolCalls := ollamaToolCalls(message["tool_calls"])
	done, _ := event["done"].(bool)
	s.done = s.done || done
	switch {
	case content != "" || toolCalls != nil:
		delta := map[string]any{}
		if content != "" {
			delta["content"] = content
		}
		if toolCalls != nil {
			delta["tool_calls"] = toolCalls
		}
		s.emitted = true
		return ollamaChunk(s.model, delta, nil)
	case done:
		return ollamaChunk(s.model, map[string]any{}, "stop")
	}
	return nil
}

// ollamaChunk is one Chat chunk as the gateway encodes it, framed as SSE.
func ollamaChunk(model string, delta map[string]any, finish any) []byte {
	chunk, _ := json.Marshal(map[string]any{
		"id": "chatcmpl-ollama", "object": "chat.completion.chunk", "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
	})
	return []byte("data: " + string(chunk) + "\n\n")
}

func (s *ollamaStream) Close() error { return s.body.Close() }

// Losses implements core.LossReporter. They are known before the first
// frame.
func (s *ollamaStream) Losses() []core.Loss { return slices.Clone(s.losses) }
