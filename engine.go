package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/xibodev/llm-translate"
)

// Engine is the core LLM Gateway routing and proxy engine.
// It implements http.Handler and can be mounted directly on any router.
type Engine struct {
	config    Config
	mu        sync.RWMutex
	providers map[string]any // provider instances
	routes    map[string]RouteConfig
	auth      Authenticator
	policy    PolicyGate
	resolver  CredentialResolver
	usageHook UsageHook
}

func NewEngine(cfg Config) *Engine {
	if cfg.PolicyGate == nil {
		cfg.PolicyGate = AllowAllPolicy{}
	}
	if cfg.Routes == nil {
		cfg.Routes = make(map[string]RouteConfig)
	}

	return &Engine{
		config:    cfg,
		providers: make(map[string]any),
		routes:    cfg.Routes,
		auth:      cfg.Authenticator,
		policy:    cfg.PolicyGate,
		resolver:  cfg.CredentialResolver,
		usageHook: cfg.UsageHook,
	}
}

// RegisterProvider registers an active provider backend.
func (e *Engine) RegisterProvider(id string, provider any) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.providers[id] = provider
}

// ServeHTTP implements http.Handler.
func (e *Engine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	// Strip optional trailing slashes or normalize /v1 prefix
	normalized := strings.TrimSuffix(path, "/")

	switch {
	case strings.HasSuffix(normalized, "/v1/models") || normalized == "/models":
		if r.Method == http.MethodGet {
			e.handleModels(w, r)
			return
		}
	case strings.HasSuffix(normalized, "/v1/chat/completions") || normalized == "/chat/completions":
		if r.Method == http.MethodPost {
			e.handleChatCompletions(w, r)
			return
		}
	case strings.HasSuffix(normalized, "/v1/messages") || normalized == "/messages":
		if r.Method == http.MethodPost {
			e.handleMessages(w, r)
			return
		}
	case strings.HasSuffix(normalized, "/healthz") || strings.HasSuffix(normalized, "/health"):
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"status": "ok", "version": "core-1.0"})
		return
	}

	http.NotFound(w, r)
}

func (e *Engine) authenticate(r *http.Request) (*Principal, error) {
	if e.auth != nil {
		return e.auth.Authenticate(r)
	}
	return &Principal{ID: "anonymous", Type: "anonymous"}, nil
}

func (e *Engine) handleModels(w http.ResponseWriter, r *http.Request) {
	principal, err := e.authenticate(r)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":%q,"type":"auth_error"}}`, err.Error()), http.StatusUnauthorized)
		return
	}

	e.mu.RLock()
	routes := make(map[string]RouteConfig, len(e.routes))
	for k, v := range e.routes {
		routes[k] = v
	}
	e.mu.RUnlock()

	var models []ModelInfo
	now := time.Now().Unix()

	// 1. Add route aliases
	for name, route := range routes {
		desc := route.Description
		if desc == "" {
			desc = fmt.Sprintf("Route alias across %d targets", len(route.Targets))
		}
		models = append(models, ModelInfo{
			ID:          name,
			Object:      "model",
			Created:     now,
			OwnedBy:     "route",
			Description: desc,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   models,
	})
	_ = principal
}

func (e *Engine) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	principal, err := e.authenticate(r)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": err.Error(), "type": "auth_error"}})
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":{"message":"read request body failed"}}`, http.StatusBadRequest)
		return
	}

	var payload map[string]any
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		http.Error(w, `{"error":{"message":"malformed json body"}}`, http.StatusBadRequest)
		return
	}

	requestedModel, _ := payload["model"].(string)
	if requestedModel == "" {
		http.Error(w, `{"error":{"message":"'model' field is required"}}`, http.StatusBadRequest)
		return
	}

	stream, _ := payload["stream"].(bool)
	targets := e.resolveTargets(r.Context(), principal, requestedModel)
	if len(targets) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": fmt.Sprintf("no enabled provider available for requested model %q", requestedModel),
				"type":    "model_not_found",
			},
		})
		return
	}

	var lastErr error
	var lastTarget Target
	start := time.Now()

	for _, target := range targets {
		lastTarget = target
		var cred *Credential
		if e.resolver != nil {
			cred, err = e.resolver.Resolve(r.Context(), principal, target.Provider)
			if err != nil {
				lastErr = err
				if requestCanceled(r.Context(), err) || !failoverEligible(err) {
					break
				}
				continue
			}
		}

		provider := e.getProvider(target.Provider)
		if provider == nil {
			lastErr = fmt.Errorf("provider %q not registered", target.Provider)
			continue
		}

		// Try streaming or non-streaming via provider interface
		type streamableProvider interface {
			Stream(ctx context.Context, model string, payload map[string]any, cred *Credential) (StreamIter, error)
		}
		type completableProvider interface {
			Complete(ctx context.Context, model string, payload map[string]any, cred *Credential) (map[string]any, error)
		}

		if stream {
			sp, ok := provider.(streamableProvider)
			if !ok {
				lastErr = fmt.Errorf("provider %q does not support streaming", target.Provider)
				continue
			}
			iter, err := sp.Stream(r.Context(), target.Model, payload, cred)
			if err != nil {
				lastErr = err
				if requestCanceled(r.Context(), err) || !failoverEligible(err) {
					break
				}
				continue
			}
			streamErr, wrote := writeSSEStream(w, iter)
			if streamErr != nil && !wrote {
				lastErr = streamErr
				if requestCanceled(r.Context(), streamErr) || !failoverEligible(streamErr) {
					break
				}
				continue
			}
			status := http.StatusOK
			if streamErr != nil {
				status = http.StatusBadGateway
			}
			e.recordTelemetry(r.Context(), principal, target, time.Since(start), true, status, streamErr)
			return
		}

		if cp, ok := provider.(completableProvider); ok {
			resp, err := cp.Complete(r.Context(), target.Model, payload, cred)
			if err != nil {
				lastErr = err
				if requestCanceled(r.Context(), err) || !failoverEligible(err) {
					break
				}
				continue
			}

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
			e.recordTelemetry(r.Context(), principal, target, time.Since(start), false, 200, nil)
			return
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadGateway)
	errMsg := "all targets failed"
	if lastErr != nil {
		errMsg = lastErr.Error()
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": errMsg,
			"type":    "upstream_error",
		},
	})
	e.recordTelemetry(r.Context(), principal, lastTarget, time.Since(start), stream, 502, lastErr)
}

func (e *Engine) handleMessages(w http.ResponseWriter, r *http.Request) {
	principal, err := e.authenticate(r)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(map[string]any{"type": "error", "error": map[string]any{"type": "authentication_error", "message": err.Error()}})
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"type":"error","error":{"type":"invalid_request_error","message":"read request failed"}}`, http.StatusBadRequest)
		return
	}

	var anthropicReq map[string]any
	if err := json.Unmarshal(bodyBytes, &anthropicReq); err != nil {
		http.Error(w, `{"type":"error","error":{"type":"invalid_request_error","message":"malformed json"}}`, http.StatusBadRequest)
		return
	}

	requestedModel, _ := anthropicReq["model"].(string)
	stream, _ := anthropicReq["stream"].(bool)

	// Translate Anthropic request to OpenAI format
	openaiMsgs, extra, _ := translate.AnthropicRequestToOpenAI(anthropicReq)
	openaiPayload := map[string]any{
		"model":    requestedModel,
		"messages": openaiMsgs,
		"stream":   stream,
	}
	for k, v := range extra {
		openaiPayload[k] = v
	}

	targets := e.resolveTargets(r.Context(), principal, requestedModel)
	if len(targets) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "not_found_error",
				"message": fmt.Sprintf("model %q not found", requestedModel),
			},
		})
		return
	}

	target := targets[0]
	provider := e.getProvider(target.Provider)
	if provider == nil {
		http.Error(w, fmt.Sprintf(`{"type":"error","error":{"message":"provider %s unavailable"}}`, target.Provider), http.StatusBadGateway)
		return
	}

	var cred *Credential
	if e.resolver != nil {
		cred, _ = e.resolver.Resolve(r.Context(), principal, target.Provider)
	}

	if stream {
		type streamableProvider interface {
			Stream(ctx context.Context, model string, payload map[string]any, cred *Credential) (StreamIter, error)
		}
		if sp, ok := provider.(streamableProvider); ok {
			iter, err := sp.Stream(r.Context(), target.Model, openaiPayload, cred)
			if err != nil {
				http.Error(w, fmt.Sprintf(`{"type":"error","error":{"message":%q}}`, err.Error()), http.StatusBadGateway)
				return
			}

			defer iter.Close()
			flusher, _ := w.(http.Flusher)
			var streamErr error
			var pendingErr error
			terminal := false
			wrote := false
			chunksIter := func() (string, bool) {
				for {
					if pendingErr != nil {
						streamErr = pendingErr
						return "", false
					}
					b, err := iter.Next()
					if len(b) == 0 && err != nil {
						if err == io.EOF {
							streamErr = errors.New("provider stream ended before data: [DONE]")
						} else {
							streamErr = err
						}
						return "", false
					}
					if len(b) == 0 {
						continue
					}
					pendingErr = err
					payload, done, ok := sseDataPayload(b)
					if done {
						pendingErr = nil
						terminal = true
						return "", false
					}
					if ok {
						return payload, true
					}
				}
			}

			translate.OpenAIStreamToAnthropicSSE(chunksIter, target.Model, func(sseEvent string) {
				if streamErr != nil {
					return
				}
				if !wrote {
					w.Header().Set("Content-Type", "text/event-stream")
					w.Header().Set("Cache-Control", "no-cache")
					w.Header().Set("Connection", "keep-alive")
				}
				wrote = true
				_, _ = w.Write([]byte(sseEvent))
				if flusher != nil {
					flusher.Flush()
				}
			})
			if streamErr != nil {
				if !wrote {
					http.Error(w, fmt.Sprintf(`{"type":"error","error":{"message":%q}}`, streamErr.Error()), http.StatusBadGateway)
				} else if !requestCanceled(r.Context(), streamErr) {
					_, _ = io.WriteString(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"upstream stream failed\"}}\n\n")
					if flusher != nil {
						flusher.Flush()
					}
				}
				return
			}
			if !terminal {
				http.Error(w, `{"type":"error","error":{"message":"provider stream ended without a terminal event"}}`, http.StatusBadGateway)
			}
			return
		}
	}

	// Non-streaming completion translated back to Anthropic
	type completableProvider interface {
		Complete(ctx context.Context, model string, payload map[string]any, cred *Credential) (map[string]any, error)
	}
	if cp, ok := provider.(completableProvider); ok {
		oaiResp, err := cp.Complete(r.Context(), target.Model, openaiPayload, cred)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"type":"error","error":{"message":%q}}`, err.Error()), http.StatusBadGateway)
			return
		}

		anthropicResp := translate.OpenAIResponseToAnthropic(oaiResp, target.Model)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(anthropicResp)
		return
	}

	http.Error(w, `{"type":"error","error":{"message":"unsupported provider"}}`, http.StatusNotImplemented)
}

func (e *Engine) resolveTargets(ctx context.Context, principal *Principal, requested string) []Target {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if route, ok := e.routes[requested]; ok {
		var valid []Target
		for _, t := range route.Targets {
			if e.policy == nil {
				valid = append(valid, t)
				continue
			}
			if allowed, _ := e.policy.Allows(ctx, principal, t); allowed {
				valid = append(valid, t)
			}
		}
		return valid
	}

	// Format "provider/model"
	if idx := strings.IndexByte(requested, '/'); idx != -1 {
		t := Target{
			Provider: requested[:idx],
			Model:    requested[idx+1:],
		}
		if e.policy == nil {
			return []Target{t}
		}
		if allowed, _ := e.policy.Allows(ctx, principal, t); allowed {
			return []Target{t}
		}
		return nil
	}

	return nil
}

func (e *Engine) getProvider(id string) any {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.providers[id]
}

func (e *Engine) recordTelemetry(ctx context.Context, principal *Principal, target Target, d time.Duration, stream bool, status int, err error) {
	if e.usageHook == nil {
		return
	}
	errStr := ""
	if err != nil {
		errStr = err.Error()
	}
	pID := ""
	projID := ""
	if principal != nil {
		pID = principal.ID
		projID = principal.ProjectID
	}
	e.usageHook.RecordUsage(ctx, UsageRecord{
		PrincipalID: pID,
		ProjectID:   projID,
		Provider:    target.Provider,
		Model:       target.Model,
		Duration:    d,
		Stream:      stream,
		StatusCode:  status,
		Error:       errStr,
	})
}

func failoverEligible(err error) bool {
	var classified ProviderErrorClassifier
	return errors.As(err, &classified) && classified.ProviderErrorClassification().FailoverEligible
}

func requestCanceled(ctx context.Context, err error) bool {
	return ctx.Err() != nil
}

func writeSSEStream(w http.ResponseWriter, iter StreamIter) (streamErr error, wrote bool) {
	defer func() {
		if err := iter.Close(); streamErr == nil && err != nil {
			streamErr = err
		}
	}()
	flusher, _ := w.(http.Flusher)
	terminal := false
	for {
		chunk, err := iter.Next()
		if len(chunk) > 0 {
			if terminal {
				return errors.New("provider stream emitted an SSE event after data: [DONE]"), wrote
			}
			if !wrote {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				w.Header().Set("Connection", "keep-alive")
			}
			wrote = true
			_, _ = w.Write(chunk)
			if flusher != nil {
				flusher.Flush()
			}
			if isSSEDone(chunk) {
				terminal = true
			}
		}
		if err == io.EOF {
			if !terminal {
				return errors.New("provider stream ended before data: [DONE]"), wrote
			}
			return nil, wrote
		}
		if err != nil {
			return err, wrote
		}
	}
}

func isSSEDone(frame []byte) bool {
	payload, ok := sseData(frame)
	return ok && payload == "[DONE]"
}

func sseDataPayload(frame []byte) (payload string, done bool, ok bool) {
	payload, ok = sseData(frame)
	if !ok {
		return "", false, false
	}
	if payload == "[DONE]" {
		return "", true, true
	}
	return payload, false, true
}

func sseData(frame []byte) (string, bool) {
	var values []string
	for _, line := range strings.Split(strings.ReplaceAll(string(frame), "\r\n", "\n"), "\n") {
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		value := strings.TrimPrefix(line, "data:")
		if strings.HasPrefix(value, " ") {
			value = value[1:]
		}
		values = append(values, value)
	}
	return strings.Join(values, "\n"), len(values) > 0
}
