module conger

// generic_expert_registry.v — domain-independent expert registry, generic
// over the scene payload `T`.
import mlx

// GenericExpert maps an observation to a StructuredHypothesis.
pub interface GenericExpert[T] {
	estimate(observation mlx.Array) StructuredHypothesis[T]
}

pub struct GenericExpertRegistry[T] {
pub:
	experts map[string]GenericExpert[T]
	gate    GenericStructureGate[T]
pub mut:
	birth_controller   ?&StructureBirthController
	last_birth_request ?StructureBirthRequest
}

// decide routes one observation through all experts, gates them, and (when a
// birth controller is attached) records the gate outcome.
pub fn (mut r GenericExpertRegistry[T]) decide(observation mlx.Array) GenericStructureDecision[T] {
	mut estimates := map[string]StructuredHypothesis[T]{}
	for name, expert in r.experts {
		estimates[name] = expert.estimate(observation)
	}
	decision := r.gate.decide(estimates)
	r.last_birth_request = none
	if mut bc := r.birth_controller {
		r.last_birth_request = bc.observe[T](decision, observation, observation)
	}
	return decision
}
