// Package execution provides the primitives failover is built from: a
// HealthTracker that breaks circuits and cools down instances, and
// executors that try candidates in order and never fail over once output
// has reached the caller.
//
// Routing stays product code. A product resolves its candidates, orders
// them, applies its policy gates and decides which stream frames carry
// output; the executors walk the list they are given.
//
// # Health
//
// A HealthTracker keeps one state per key, the instance a candidate runs on,
// behind a mutex and with an injectable clock. What an operation's outcome
// says about health is an Observation; the default, Observe, reads the
// error's classification through core.ClassifyError. A circuit failure
// extends the key's streak of consecutive failures, and once the streak
// reaches the policy's FailureThreshold the circuit opens for OpenDuration,
// or for what Backoff returns. When that time has passed the circuit is
// half-open: it admits requests and, because the streak is kept, the next
// failure reopens it at once. A success ends the streak and closes the
// circuit. A RetryAfter makes the key unavailable until then, as the
// upstream asked, whatever succeeds meanwhile.
//
// # Execution
//
// Execute tries candidates in order. It skips a candidate its Health reports
// unavailable, stops at a terminal disposition, and moves on after a
// failover or retryable one, repeating the candidate first only when Retry
// asks. It records the outcome of each candidate tried and returns the
// result with a trace of every candidate it reached. When the caller's
// context ends it stops with the context's error and records nothing.
//
// ExecuteStream does the same for streams. A candidate serves once one of
// its frames carries output, content or reasoning, as the product's
// predicate decides. Until then its frames are held back and a failure moves
// on to the next candidate; from then on nothing fails over, and a failure
// reaches the caller as the stream's *AfterOutputError.
//
// # Existing breakers
//
// The gateway's per-provider breaker is a HealthPolicy with the provider's
// circuit threshold and cooldown, no FailureWindow and IgnoreRetryAfter,
// read by an Observe that fails circuit failures, succeeds every other
// invocation error and ignores errors that never reached the upstream.
// Facet Studio's cooldown is a FailureThreshold of one with a 24-hour
// FailureWindow, IgnoreRetryAfter, and a Backoff of 1, 5 and 25 minutes and
// then an hour, or of 5, 10 and 20 hours and then a day for billing
// failures. The tests named after each product pin these mappings.
package execution
