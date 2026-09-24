package execution_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/execution"
)

func ExampleExecute() {
	health := execution.NewHealthTracker(execution.HealthOptions{})
	executor := execution.Executor[string]{
		Health: health,
		Key:    func(instance string) string { return instance },
	}
	invoke := func(_ context.Context, instance string) (string, error) {
		if instance == "primary" {
			return "", &core.ProviderError{Message: "primary is overloaded", Classification: core.ProviderErrorClassification{
				StatusCode: 503, Retryable: true, FailoverEligible: true, CircuitFailure: true,
			}}
		}
		return "hello from " + instance, nil
	}

	result, err := execution.Execute(context.Background(), executor, []string{"primary", "secondary"}, invoke)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Value)
	for _, attempt := range result.Attempts {
		if attempt.Err != nil {
			fmt.Println(attempt.Candidate, attempt.Disposition, attempt.Classification.StatusCode)
		}
	}
	fmt.Println("primary's streak:", health.State("primary").Streak)
	// Output:
	// hello from secondary
	// primary retryable 503
	// primary's streak: 1
}

// sseFrames yields fixed SSE frames, then fails with err or ends.
type sseFrames struct {
	frames []string
	err    error
}

func (s *sseFrames) Next() ([]byte, error) {
	if len(s.frames) == 0 {
		if s.err != nil {
			return nil, s.err
		}
		return nil, io.EOF
	}
	frame := s.frames[0]
	s.frames = s.frames[1:]
	return []byte(frame), nil
}

func (s *sseFrames) Close() error { return nil }

func ExampleExecuteStream() {
	dropped := &core.ProviderError{Message: "connection reset", Classification: core.ProviderErrorClassification{
		Retryable: true, FailoverEligible: true, CircuitFailure: true,
	}}
	open := func(_ context.Context, instance string) (core.StreamIter, error) {
		if instance == "primary" {
			// A role frame carries no output, so the caller never sees it.
			return &sseFrames{frames: []string{"data: {\"role\":\"assistant\"}\n\n"}, err: dropped}, nil
		}
		return &sseFrames{frames: []string{
			"data: {\"role\":\"assistant\"}\n\n",
			"data: {\"content\":\"hello\"}\n\n",
			"data: [DONE]\n\n",
		}}, nil
	}
	carriesOutput := func(frame []byte) bool { return bytes.Contains(frame, []byte(`"content"`)) }

	result, err := execution.ExecuteStream(context.Background(), execution.Executor[string]{}, []string{"primary", "secondary"}, open, carriesOutput)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer result.Value.Close()
	fmt.Println("served by", result.Candidate)
	for {
		frame, err := result.Value.Next()
		if err != nil {
			// io.EOF, or an *AfterOutputError once output reached the caller.
			break
		}
		fmt.Println(strings.TrimSpace(string(frame)))
	}
	// Output:
	// served by secondary
	// data: {"role":"assistant"}
	// data: {"content":"hello"}
	// data: [DONE]
}
