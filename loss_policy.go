package core

import (
	"fmt"
	"strings"

	translate "github.com/xibodev/llm-translate"
)

// LossAction is what a loss policy does with one loss.
type LossAction string

const (
	LossAllow  LossAction = "allow"
	LossReject LossAction = "reject"
)

// LossRule matches losses and decides them.
type LossRule struct {
	// Path is a glob over the dot-separated loss path. "*" matches exactly one
	// segment, "**" matches any number of segments including none, and any
	// other segment matches literally. An empty Path matches every loss.
	Path string
	// Class and Severity narrow the rule. Empty matches any value.
	Class    translate.LossClass
	Severity translate.LossSeverity
	Action   LossAction
}

// LossPolicy decides which translation and adaptation losses a product
// accepts.
//
// Rules apply in order and the first match decides. A loss that no rule
// matches is rejected unless it is advisory, so an unknown severity counts as
// material. Allowed losses are still reported to the caller.
type LossPolicy struct {
	Rules []LossRule
}

// Validate reports a malformed rule.
func (p LossPolicy) Validate() error {
	for index, rule := range p.Rules {
		switch rule.Action {
		case LossAllow, LossReject:
		default:
			return fmt.Errorf("loss rule %d has unknown action %q", index+1, rule.Action)
		}
		if rule.Path != "" {
			for _, segment := range strings.Split(rule.Path, ".") {
				if segment == "" {
					return fmt.Errorf("loss rule %d path %q has an empty segment", index+1, rule.Path)
				}
			}
		}
		switch rule.Class {
		case "", translate.LossUnsupported, translate.LossDropped, translate.LossApproximated, translate.LossRenamed:
		default:
			return fmt.Errorf("loss rule %d has unknown class %q", index+1, rule.Class)
		}
		switch rule.Severity {
		case "", translate.LossMaterial, translate.LossAdvisory:
		default:
			return fmt.Errorf("loss rule %d has unknown severity %q", index+1, rule.Severity)
		}
	}
	return nil
}

// Decide returns the action for one loss.
func (p LossPolicy) Decide(loss Loss) LossAction {
	for _, rule := range p.Rules {
		if rule.matches(loss) {
			return rule.Action
		}
	}
	if loss.Severity == translate.LossAdvisory {
		return LossAllow
	}
	return LossReject
}

// Rejected returns the losses the policy rejects, in report order.
func (p LossPolicy) Rejected(losses []Loss) []Loss {
	var rejected []Loss
	for _, loss := range losses {
		if p.Decide(loss) == LossReject {
			rejected = append(rejected, loss)
		}
	}
	return rejected
}

// Check returns a *LossPolicyError when the policy rejects any loss.
func (p LossPolicy) Check(losses []Loss) error {
	if rejected := p.Rejected(losses); len(rejected) > 0 {
		return &LossPolicyError{Losses: rejected}
	}
	return nil
}

func (r LossRule) matches(loss Loss) bool {
	if r.Class != "" && r.Class != loss.Class {
		return false
	}
	if r.Severity != "" && r.Severity != loss.Severity {
		return false
	}
	if r.Path == "" {
		return true
	}
	return matchLossPath(strings.Split(r.Path, "."), strings.Split(loss.Path, "."))
}

func matchLossPath(pattern, path []string) bool {
	for len(pattern) > 0 {
		switch pattern[0] {
		case "**":
			for skip := 0; skip <= len(path); skip++ {
				if matchLossPath(pattern[1:], path[skip:]) {
					return true
				}
			}
			return false
		case "*":
			if len(path) == 0 {
				return false
			}
		default:
			if len(path) == 0 || path[0] != pattern[0] {
				return false
			}
		}
		pattern, path = pattern[1:], path[1:]
	}
	return len(path) == 0
}

// LossPolicyError reports losses a policy rejected. Another target may serve
// the surface natively, so it permits failover.
type LossPolicyError struct {
	Losses []Loss
}

func (e *LossPolicyError) Error() string {
	paths := make([]string, len(e.Losses))
	for index, loss := range e.Losses {
		paths[index] = loss.Path
	}
	return "translation would lose " + strings.Join(paths, ", ")
}

// ProviderErrorClassification permits failover without marking the provider
// unhealthy.
func (e *LossPolicyError) ProviderErrorClassification() ProviderErrorClassification {
	return ProviderErrorClassification{FailoverEligible: true}
}
