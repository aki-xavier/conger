module conger

// generic_expert_registry.v — domain-independent expert registry (V port of
// src/generic_expert_registry.py).
import mlx

// GenericExpert maps an observation to a StructuredHypothesis.
interface GenericExpert {
	estimate(observation mlx.Array) StructuredHypothesis
}

struct GenericExpertRegistry {
	experts map[string]GenericExpert
	gate    GenericStructureGate
mut:
	birth_controller   &StructureBirthController = unsafe { nil }
	last_birth_request ?StructureBirthRequest
}

// decide routes one observation through all experts, gates them, and (when a
// birth controller is attached) records the gate outcome.
fn (mut r GenericExpertRegistry) decide(observation mlx.Array) GenericStructureDecision {
	mut estimates := map[string]StructuredHypothesis{}
	for name, expert in r.experts {
		estimates[name] = expert.estimate(observation)
	}
	decision := r.gate.decide(estimates)
	r.last_birth_request = none
	if !isnil(r.birth_controller) {
		r.last_birth_request = r.birth_controller.observe(decision, observation, observation)
	}
	return decision
}
