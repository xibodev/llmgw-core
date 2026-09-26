package execution

import (
	"encoding/json"
	"math"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// ExponentialBackoff returns a Retry.Delay that waits initial after the
// first failed try and multiplier times longer after each later one, at
// most maxDelay, as the gateway's resilience wrapper waits. When the
// upstream asked callers to wait longer, through Retry-After, it waits
// that long instead.
func ExponentialBackoff(initial time.Duration, multiplier float64, maxDelay time.Duration) func(failed int, err error) time.Duration {
	return func(failed int, err error) time.Duration {
		backoff := float64(initial) * math.Pow(multiplier, float64(max(failed, 1)-1))
		if !(backoff > 0) {
			// Also a NaN, which no Duration can hold.
			backoff = 0
		}
		delay := time.Duration(math.Min(backoff, float64(maxDelay)))
		return max(delay, core.ClassifyError(err).RetryAfter)
	}
}

// RepeatableRequest reports whether sending request again is as safe as
// sending it once, as the gateway's resilience wrapper decides:
//
//   - Chat Completions, Messages and speech repeat.
//   - Responses repeats unless the request is stateful: it stores the
//     response, continues a previous response or a conversation, uses a
//     stored prompt, or carries a tool or an input item that acts on
//     upstream state, such as a hosted tool, a vector store or an item
//     reference. A body that is no JSON object is not repeated.
//   - Embeddings, images, video and transcription never repeat. The
//     gateway repeats none of them, since a repeat may pay for a second
//     result.
func RepeatableRequest(request core.Request) bool {
	switch request.Surface {
	case core.ModelSurfaceChatCompletions, core.ModelSurfaceMessages, core.ModelSurfaceAudioSpeech:
		return true
	case core.ModelSurfaceResponses:
		var payload map[string]any
		if json.Unmarshal(request.Body, &payload) != nil || payload == nil {
			return false
		}
		return !statefulResponses(payload)
	}
	return false
}

// statefulResponses is the gateway's ResponsesPayloadIsStateful, verbatim.
func statefulResponses(payload map[string]any) bool {
	if store, _ := payload["store"].(bool); store {
		return true
	}
	if value := payload["previous_response_id"]; value != nil && value != "" {
		return true
	}
	if payload["conversation"] != nil {
		return true
	}
	if prompt, ok := payload["prompt"].(map[string]any); ok && prompt["id"] != nil {
		return true
	}
	if tools, ok := payload["tools"].([]any); ok {
		for _, raw := range tools {
			tool, _ := raw.(map[string]any)
			toolType, _ := tool["type"].(string)
			if toolType != "" && toolType != "function" {
				return true
			}
			if ids, ok := tool["vector_store_ids"].([]any); ok && len(ids) > 0 {
				return true
			}
			if tool["container"] != nil {
				return true
			}
		}
	}
	return statefulResponsesValue(payload["input"])
}

func statefulResponsesValue(value any) bool {
	switch current := value.(type) {
	case []any:
		for _, nested := range current {
			if statefulResponsesValue(nested) {
				return true
			}
		}
	case map[string]any:
		itemType, _ := current["type"].(string)
		switch itemType {
		case "item_reference", "reasoning", "compaction", "computer_call",
			"computer_call_output", "local_shell_call", "local_shell_call_output",
			"mcp_approval_response", "mcp_call", "mcp_list_tools":
			return true
		}
		if current["encrypted_content"] != nil || current["file_id"] != nil {
			return true
		}
		if current["id"] != nil && itemType != "" && itemType != "message" {
			return true
		}
		for _, nested := range current {
			if statefulResponsesValue(nested) {
				return true
			}
		}
	}
	return false
}
