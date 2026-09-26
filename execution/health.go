package execution

import (
	"context"
	"errors"
	"maps"
	"sync"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// DefaultFailureThreshold and DefaultOpenDuration are the policy of every
// key of a HealthTracker without a Policy. They are router.CircuitBreaker's
// defaults.
const (
	DefaultFailureThreshold = 5
	DefaultOpenDuration     = 30 * time.Second
)

// Effect is what one outcome does to a key's health.
type Effect int

const (
	// EffectNeutral changes nothing: the operation never reached the
	// instance, or its failure says nothing about the instance.
	EffectNeutral Effect = iota
	// EffectSuccess shows the instance answers. It ends the streak and
	// closes the circuit. A RetryAfter cooldown still runs its course: an
	// operation admitted before the upstream asked callers to wait proves
	// nothing about the wait.
	EffectSuccess
	// EffectFailure is instability. It extends the streak toward opening the
	// circuit.
	EffectFailure
)

// Observation is what one outcome says about the instance that produced it.
type Observation struct {
	// Effect is what the outcome does to the instance's key.
	Effect Effect
	// Class names the kind of a failure, for a Backoff that treats kinds
	// differently, such as billing failures. The empty class is a kind too.
	Class string
	// RetryAfter is how long the upstream asked callers to wait. A positive
	// value starts a cooldown, which only a later RetryAfter can extend,
	// unless the policy ignores RetryAfter.
	RetryAfter time.Duration
}

// Observe is the default Observation of an operation's error, read from its
// classification:
//
//   - nil is a success.
//   - A circuit failure is a failure.
//   - Another failure with an upstream status is a success: the instance
//     answered, so a definitive rejection such as 400 or 401 ends a streak
//     of instability instead of extending it.
//   - Anything else is neutral, including a configuration error, a surface
//     the provider lacks and a canceled request.
//
// RetryAfter is the classification's.
func Observe(err error) Observation {
	if err == nil {
		return Observation{Effect: EffectSuccess}
	}
	classification := core.ClassifyError(err)
	observation := Observation{RetryAfter: classification.RetryAfter}
	switch {
	case classification.CircuitFailure:
		observation.Effect = EffectFailure
	case classification.StatusCode != 0 && !canceled(err):
		// Classifications of a canceled operation keep the status they saw,
		// which proves nothing about the instance.
		observation.Effect = EffectSuccess
	}
	return observation
}

func canceled(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// HealthPolicy is how failures make one key unavailable.
//
// When an open circuit's time has passed the circuit is half-open: it admits
// requests, and because the streak is kept the next failure reopens it at
// once. A success closes it.
type HealthPolicy struct {
	// FailureThreshold is the streak of consecutive circuit failures that
	// opens the circuit. Zero or less disables circuit breaking: failures
	// are not counted and the circuit never opens.
	FailureThreshold int
	// OpenDuration is how long an open circuit refuses requests.
	OpenDuration time.Duration
	// Backoff, when set, replaces OpenDuration for each failure at or past
	// the threshold, so the open time can grow with the streak or depend on
	// the failure's class. It runs under the tracker's lock and must be a
	// pure function of its argument.
	Backoff func(Failure) time.Duration
	// FailureWindow forgets a streak whose last failure is older than the
	// window, so the next failure starts a new streak. Zero never forgets.
	FailureWindow time.Duration
	// IgnoreRetryAfter keeps RetryAfter from starting cooldowns, for a
	// product whose failover predates them.
	IgnoreRetryAfter bool
}

// Failure is a failure at or past the threshold, as a Backoff sees it.
type Failure struct {
	// Key is the key that failed.
	Key string
	// Class is the failure's Observation.Class.
	Class string
	// Streak is the length of the streak, this failure included, and
	// ClassFailures the number of its failures of Class.
	Streak        int
	ClassFailures int
}

// HealthOptions configures a HealthTracker. Every field is optional.
type HealthOptions struct {
	// Policy returns the policy of one key, so keys can differ and follow
	// settings as they change. Nil gives every key DefaultFailureThreshold
	// and DefaultOpenDuration.
	Policy func(key string) HealthPolicy
	// Observe decides what an operation's error says about health. Nil uses
	// Observe.
	Observe func(error) Observation
	// Now returns the current time. Nil uses time.Now.
	Now func() time.Time
}

// HealthState is what a HealthTracker holds for one key.
type HealthState struct {
	// Streak counts consecutive circuit failures, and ClassFailures counts
	// them by Observation.Class.
	Streak        int
	ClassFailures map[string]int
	// LastFailure is when the streak last grew.
	LastFailure time.Time
	// OpenUntil is when the circuit stops refusing requests. Once it has
	// passed with the streak still at the threshold, the circuit is
	// half-open.
	OpenUntil time.Time
	// CooldownUntil is when the latest RetryAfter cooldown ends.
	CooldownUntil time.Time
}

// Health is what an Executor consults before trying a candidate and tells
// afterwards. HealthTracker implements it; a product may adapt its own.
type Health interface {
	// Available reports whether key may be tried now and, when it may not,
	// until when.
	Available(key string) (available bool, until time.Time)
	// Record reports the outcome of one operation on key: nil is a success.
	Record(key string, err error)
}

var _ Health = (*HealthTracker)(nil)

// HealthTracker tracks the health of keys, typically provider instances. It
// is safe for concurrent use.
type HealthTracker struct {
	policy  func(string) HealthPolicy
	observe func(error) Observation
	now     func() time.Time

	mu   sync.Mutex
	keys map[string]*health
}

// health is the state of one key. A key without state is healthy, so a
// success deletes it unless a cooldown is running.
type health struct {
	streak      int
	classes     map[string]int
	lastFailure time.Time
	openUntil   time.Time
	coolUntil   time.Time
}

// NewHealthTracker returns a tracker that has recorded nothing.
func NewHealthTracker(options HealthOptions) *HealthTracker {
	tracker := &HealthTracker{
		policy:  options.Policy,
		observe: options.Observe,
		now:     options.Now,
		keys:    map[string]*health{},
	}
	if tracker.policy == nil {
		tracker.policy = func(string) HealthPolicy {
			return HealthPolicy{FailureThreshold: DefaultFailureThreshold, OpenDuration: DefaultOpenDuration}
		}
	}
	if tracker.observe == nil {
		tracker.observe = Observe
	}
	if tracker.now == nil {
		tracker.now = time.Now
	}
	return tracker
}

// Record updates key's health with the outcome of one operation: nil is a
// success. Do not record an operation whose caller gave up: it says nothing
// about the instance, and Execute never records one.
func (t *HealthTracker) Record(key string, err error) {
	// Product code runs outside the lock, so it may take locks of its own.
	observation := t.observe(err)
	policy := t.policy(key)
	t.mu.Lock()
	defer t.mu.Unlock()
	now := t.now()
	switch observation.Effect {
	case EffectSuccess:
		t.succeed(key, now)
	case EffectFailure:
		if policy.FailureThreshold > 0 {
			t.fail(key, policy, observation.Class, now)
		}
	}
	if observation.RetryAfter > 0 && !policy.IgnoreRetryAfter {
		state := t.state(key)
		state.coolUntil = later(state.coolUntil, now.Add(observation.RetryAfter))
	}
}

func (t *HealthTracker) succeed(key string, now time.Time) {
	state := t.keys[key]
	if state == nil {
		return
	}
	if !now.Before(state.coolUntil) {
		delete(t.keys, key)
		return
	}
	*state = health{coolUntil: state.coolUntil}
}

func (t *HealthTracker) fail(key string, policy HealthPolicy, class string, now time.Time) {
	state := t.state(key)
	if policy.FailureWindow > 0 && !state.lastFailure.IsZero() && now.Sub(state.lastFailure) > policy.FailureWindow {
		state.streak = 0
		clear(state.classes)
	}
	state.streak++
	if state.classes == nil {
		state.classes = map[string]int{}
	}
	state.classes[class]++
	state.lastFailure = now
	if state.streak < policy.FailureThreshold {
		return
	}
	open := policy.OpenDuration
	if policy.Backoff != nil {
		open = policy.Backoff(Failure{Key: key, Class: class, Streak: state.streak, ClassFailures: state.classes[class]})
	}
	// A deadline only moves later. A shorter open time, such as a standard
	// failure while a billing failure holds the key, never cuts it short.
	if open > 0 {
		state.openUntil = later(state.openUntil, now.Add(open))
	}
}

func (t *HealthTracker) state(key string) *health {
	state := t.keys[key]
	if state == nil {
		state = &health{}
		t.keys[key] = state
	}
	return state
}

// Available reports whether key may be tried now and, when it may not,
// until when: the later of the open circuit's end and the cooldown's. It
// reads the key's current policy, so a key whose policy stopped breaking
// circuits or honouring RetryAfter is no longer held back by them.
func (t *HealthTracker) Available(key string) (bool, time.Time) {
	t.mu.Lock()
	state, tracked := t.keys[key]
	var open, cool time.Time
	if tracked {
		open, cool = state.openUntil, state.coolUntil
	}
	now := t.now()
	t.mu.Unlock()
	if !tracked {
		return true, time.Time{}
	}
	policy := t.policy(key)
	var until time.Time
	if policy.FailureThreshold > 0 {
		until = open
	}
	if !policy.IgnoreRetryAfter {
		until = later(until, cool)
	}
	if now.Before(until) {
		return false, until
	}
	return true, time.Time{}
}

// State returns what the tracker holds for key: the zero HealthState when it
// holds nothing.
func (t *HealthTracker) State(key string) HealthState {
	t.mu.Lock()
	defer t.mu.Unlock()
	state := t.keys[key]
	if state == nil {
		return HealthState{}
	}
	return HealthState{
		Streak:        state.streak,
		ClassFailures: maps.Clone(state.classes),
		LastFailure:   state.lastFailure,
		OpenUntil:     state.openUntil,
		CooldownUntil: state.coolUntil,
	}
}

// Reset forgets key, as if nothing had been recorded for it.
func (t *HealthTracker) Reset(key string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.keys, key)
}

func later(a, b time.Time) time.Time {
	if b.After(a) {
		return b
	}
	return a
}
