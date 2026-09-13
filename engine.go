package core

import (
	"context"
	"encoding/json"
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
	config     Config
	mu         sync.RWMutex
	providers  map[string]any // provider instances
	routes     map[string]RouteConfig
	auth       Authenticator
	policy     PolicyGate
	resolver   CredentialResolver
	usageHook  UsageHook
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
	start := time.Now()

	for _, target := range targets {
		var cred *Credential
		if e.resolver != nil {
			cred, _ = e.resolver.Resolve(r.Context(), principal, target.Provider)
		}

		provider := e.getProvider(target.Provider)
		if provider == nil {
			lastErr = fmt.Errorf("provider %q not registered", target.Provider)
			continue
		}

		// Try streaming or non-streaming via provider interface
		type streamableProvider interface {
			Stream(ctx context.Context, model string, payload map[string]any, cred *Credential) (any, error)
		}
		type completableProvider interface {
			Complete(ctx context.Context, model string, payload map[string]any, cred *Credential) (map[string]any, error)
		}

		if stream {
			if sp, ok := provider.(streamableProvider); ok {
				iter, err := sp.Stream(r.Context(), target.Model, payload, cred)
				if err != nil {
					lastErr = err
					continue
				}

				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				w.Header().Set("Connection", "keep-alive")

				type nextCloser interface {
					Next() ([]byte, error)
					Close() error
				}
				if nc, ok := iter.(nextCloser); ok {
					defer nc.Close()
					flusher, _ := w.(http.Flusher)
					for {
						chunk, err := nc.Next()
						if len(chunk) > 0 {
							_, _ = w.Write(chunk)
							if flusher != nil {
								flusher.Flush()
							}
						}
						if err != nil {
							break
						}
					}
				}

				e.recordTelemetry(r.Context(), principal, target, time.Since(start), true, 200, nil)
				return
			}
		}

		if cp, ok := provider.(completableProvider); ok {
			resp, err := cp.Complete(r.Context(), target.Model, payload, cred)
			if err != nil {
				lastErr = err
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
	e.recordTelemetry(r.Context(), principal, Target{Provider: requestedModel}, time.Since(start), stream, 502, lastErr)
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
			Stream(ctx context.Context, model string, payload map[string]any, cred *Credential) (any, error)
		}
		if sp, ok := provider.(streamableProvider); ok {
			iter, err := sp.Stream(r.Context(), target.Model, openaiPayload, cred)
			if err != nil {
				http.Error(w, fmt.Sprintf(`{"type":"error","error":{"message":%q}}`, err.Error()), http.StatusBadGateway)
				return
			}

			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")

			type nextCloser interface {
				Next() ([]byte, error)
				Close() error
			}
			if nc, ok := iter.(nextCloser); ok {
				defer nc.Close()
				flusher, _ := w.(http.Flusher)
				chunksIter := func() (string, bool) {
					b, err := nc.Next()
					if err != nil || len(b) == 0 {
						return "", false
					}
					return string(b), true
				}

				translate.OpenAIStreamToAnthropicSSE(chunksIter, target.Model, func(sseEvent string) {
					_, _ = w.Write([]byte(sseEvent))
					if flusher != nil {
						flusher.Flush()
					}
				})
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
