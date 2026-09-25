package providers

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strings"
	"sync"
	"time"

	copilotauth "github.com/xibodev/llm-provider-auth/copilot"
	core "github.com/xibodev/llmgw-core"
)

// copilotSessionMargin is how long before its expiry a session is replaced,
// the margin llm-provider-auth applies to the sessions it caches.
const copilotSessionMargin = 60 * time.Second

// copilotSessions obtains Copilot session tokens.
//
// A caller's GitHub OAuth token is exchanged through Auth, and the session
// kept in memory, keyed by a digest of the token, until it nears expiry or
// Copilot rejects it. Concurrent requests for one credential share one
// exchange. The product's own token is left to Auth on every request, as the
// gateway leaves it: Auth resolves the token from its settings each time, so
// a sign-out or a new token takes effect at once, and caches the session on
// disk when it has a cache directory.
//
// Auth takes no context, so an exchange runs on its own and the request
// waits for it only while its context lasts. An abandoned exchange still
// completes, and a credential's session is kept for its next request.
type copilotSessions struct {
	auth *copilotauth.Client
	now  func() time.Time

	mu    sync.Mutex
	slots map[string]*copilotSessionSlot
}

// copilotSessionSlot is one credential's cached session and the exchange in
// flight for it.
type copilotSessionSlot struct {
	session *copilotauth.Session
	pending *copilotExchange
}

// copilotExchange is one session-token exchange. force bypasses Auth's disk
// cache, which may still hold a session Copilot rejected.
type copilotExchange struct {
	force   bool
	done    chan struct{}
	session *copilotauth.Session
	err     error
}

func newCopilotSessions(auth *copilotauth.Client, now func() time.Time) *copilotSessions {
	return &copilotSessions{auth: auth, now: now, slots: map[string]*copilotSessionSlot{}}
}

// session returns the session for credential, or for the product's own
// token when credential is nil. rejected is a session token Copilot refused,
// or empty: that session is never returned, and its replacement is
// exchanged afresh, unless another request has replaced it already.
func (s *copilotSessions) session(ctx context.Context, credential *core.Credential, rejected string) (*copilotauth.Session, error) {
	if err := s.auth.AssertProxyAllowed(); err != nil {
		return nil, copilotSessionFailure(ctx, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, copilotSessionFailure(ctx, err)
	}
	force := rejected != ""
	if credential == nil {
		return s.await(ctx, s.start(force, func(exchange *copilotExchange) (*copilotauth.Session, error) {
			return s.auth.GetSession(exchange.force)
		}))
	}
	token := strings.TrimSpace(credential.Token)
	if token == "" {
		return nil, core.NewConfigurationError("Copilot needs a credential with a GitHub OAuth token", nil)
	}
	digest := sha256.Sum256([]byte(token))
	key := hex.EncodeToString(digest[:])
	s.mu.Lock()
	slot := s.slots[key]
	if slot == nil {
		slot = &copilotSessionSlot{}
		s.slots[key] = slot
	}
	if force && slot.session != nil && slot.session.Token == rejected {
		slot.session = nil
	}
	if slot.session != nil && s.fresh(slot.session) {
		session := slot.session
		s.mu.Unlock()
		return session, nil
	}
	exchange := slot.pending
	if exchange == nil || (force && !exchange.force) {
		// The slot's lock is held until pending is set, so the exchange
		// cannot store its session before it is the slot's.
		exchange = s.start(force, func(exchange *copilotExchange) (*copilotauth.Session, error) {
			session, err := s.auth.GetSessionForOAuth(token, exchange.force)
			s.store(key, exchange, session, err)
			return session, err
		})
		slot.pending = exchange
	}
	s.mu.Unlock()
	return s.await(ctx, exchange)
}

// start runs one exchange. It completes only after run has returned, so a
// waiter never sees it before its session is stored.
func (s *copilotSessions) start(force bool, run func(*copilotExchange) (*copilotauth.Session, error)) *copilotExchange {
	exchange := &copilotExchange{force: force, done: make(chan struct{})}
	go func() {
		exchange.session, exchange.err = run(exchange)
		close(exchange.done)
	}()
	return exchange
}

// store keeps the session an exchange obtained, unless a forced exchange for
// the credential superseded it, and forgets every session that has expired.
func (s *copilotSessions) store(key string, exchange *copilotExchange, session *copilotauth.Session, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	slot := s.slots[key]
	if slot == nil || slot.pending != exchange {
		return
	}
	slot.pending = nil
	if err == nil {
		slot.session = session
	}
	for other, candidate := range s.slots {
		if candidate.pending == nil && (candidate.session == nil || !s.fresh(candidate.session)) {
			delete(s.slots, other)
		}
	}
}

// fresh reports a session that is not about to expire.
func (s *copilotSessions) fresh(session *copilotauth.Session) bool {
	return time.Unix(session.ExpiresAt, 0).Sub(s.now()) >= copilotSessionMargin
}

func (s *copilotSessions) await(ctx context.Context, exchange *copilotExchange) (*copilotauth.Session, error) {
	select {
	case <-exchange.done:
		if exchange.err != nil {
			return nil, copilotSessionFailure(ctx, exchange.err)
		}
		return exchange.session, nil
	case <-ctx.Done():
		return nil, copilotSessionFailure(ctx, ctx.Err())
	}
}

// copilotSessionFailure reports a session that could not be obtained. The
// *copilotauth.AuthError stays the cause, so a product matches its kind with
// errors.Is to add guidance that names its own settings, as the gateway
// does.
//
// Copilot being disabled and there being no token are configuration errors.
// GitHub rejecting the OAuth token keeps status 401, so a Runtime refreshes
// the credential or reports it. Other statuses classify as the gateway
// classifies them: they permit failover, but 403 does not, and a transient
// status also permits a retry and counts against the provider.
func copilotSessionFailure(ctx context.Context, err error) error {
	if callerCancellation(err) {
		return &core.ProviderError{Message: "the Copilot session request was canceled", Cause: err}
	}
	var auth *copilotauth.AuthError
	switch {
	case !errors.As(err, &auth):
		return &core.ProviderError{
			Message: "the Copilot session is unavailable", Class: core.ProviderErrorUpstream,
			Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
		}
	case errors.Is(err, copilotauth.ErrProxyDisabled), errors.Is(err, copilotauth.ErrNoOAuthToken):
		return core.NewConfigurationError(auth.Msg, err)
	case auth.Transport:
		return copilotTransportFailure(ctx, "the Copilot session-token exchange could not reach GitHub", err)
	case auth.StatusCode != 0:
		failure := copilotStatusError(auth.StatusCode, auth.Msg, 0, err)
		failure.Classification.FailoverEligible = auth.StatusCode != 401 && auth.StatusCode != 403
		return failure
	}
	// The exchange answered with something that is not a session.
	return &core.ProviderError{
		Message: auth.Msg, Class: core.ProviderErrorUpstream,
		Classification: core.ProviderErrorClassification{FailoverEligible: true}, Cause: err,
	}
}
