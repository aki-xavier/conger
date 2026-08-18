module conger

// causal_edge_test.v — structure-level causal discovery black-box tests.

fn causal_proposal(env int, ratio f64, lateral f64) TemplateProposal {
	mut delta := map[string]MetaValue{}
	delta['ratio'] = ratio
	delta['lateral_ratio'] = lateral
	mut metadata := map[string]MetaValue{}
	metadata['env'] = env
	return TemplateProposal{
		family: 'composite_attach_xyz'
		operation: 'attach'
		params: [0.0, 70.0, 70.0, 0.4, 3.0, 1.0, 0.0, 1.0]
		residual: 100.0
		complexity: 1.5
		score: 100.0
		parent_family: 'composite'
		delta: delta
		metadata: metadata
	}
}

fn learn_edges(proposals []TemplateProposal) map[string]CausalEdge {
	mut out := map[string]CausalEdge{}
	for e in CausalDeltaLearner{}.learn(proposals) {
		out[e.target] = e
	}
	return out
}

fn test_stable_delta_is_causal_edge() {
	proposals := [
		causal_proposal(0, 0.44, 0.0),
		causal_proposal(0, 0.46, 0.0),
		causal_proposal(1, 0.45, 0.0),
		causal_proposal(2, 0.45, 0.0),
	]
	edges := learn_edges(proposals)
	scale := edges['scale_ratio']
	assert scale.agreement == 1.0
	assert scale.is_causal()
}

fn test_drifting_delta_is_not_causal_edge() {
	proposals := [
		causal_proposal(0, 0.45, 0.0),
		causal_proposal(1, 0.45, 0.5),
		causal_proposal(2, 0.45, 1.0),
	]
	edges := learn_edges(proposals)
	lateral := edges['lateral_ratio']
	assert lateral.agreement == 0.0
	assert !lateral.is_causal()
}

fn test_single_env_is_not_causal_despite_trivial_agreement() {
	proposals := [causal_proposal(0, 0.45, 0.0), causal_proposal(0, 0.46, 0.0)]
	edges := learn_edges(proposals)
	scale := edges['scale_ratio']
	assert scale.agreement == 1.0
	assert !scale.is_causal()
}

fn observed_proposal(env int, observed_ratio f64) TemplateProposal {
	mut delta := map[string]MetaValue{}
	delta['ratio'] = 0.45
	delta['lateral_ratio'] = 0.0
	mut metadata := map[string]MetaValue{}
	metadata['case_index'] = env
	mut obs := map[string]f64{}
	obs['scale_ratio'] = observed_ratio
	obs['lateral_ratio'] = 0.0
	metadata['observed'] = obs
	return TemplateProposal{
		family: 'composite_attach_x'
		operation: 'attach'
		params: [0.0, 70.0, 70.0, 0.4, 3.0, 1.0, 0.0, 1.0]
		residual: 100.0
		complexity: 1.5
		score: 100.0
		parent_family: 'composite'
		delta: delta
		metadata: metadata
	}
}

fn test_observed_delta_overrides_grid_and_env_is_case_index() {
	proposals := [observed_proposal(0, 0.4), observed_proposal(1, 0.5), observed_proposal(2,
		0.6)]
	edges := learn_edges(proposals)
	assert edges['scale_ratio'].agreement < 0.5
	assert edges['scale_ratio'].n_envs == 3
	assert !edges['scale_ratio'].is_causal()
}

fn test_mirror_maps_lateral_to_period_ratio() {
	p := causal_proposal(0, 0.45, 0.6)
	mut delta := map[string]MetaValue{}
	delta['ratio'] = 0.45
	delta['lateral_ratio'] = 0.6
	mut metadata := map[string]MetaValue{}
	metadata['env'] = 0
	mirror := TemplateProposal{
		family: 'composite_mirror_x'
		operation: 'mirror'
		params: p.params
		residual: p.residual
		complexity: p.complexity
		score: p.score
		parent_family: 'composite'
		delta: delta
		metadata: metadata
	}
	edges := learn_edges([mirror, p])
	assert 'period_ratio' in edges
	assert 'lateral_ratio' in edges
}
