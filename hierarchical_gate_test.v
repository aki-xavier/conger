module conger

// hierarchical_gate_test.v — two-level hierarchical posterior gate tests.
import math

fn est(name string, residual f64, family string, complexity f64, geometry_cost f64) StructuredHypothesis[voidptr] {
	return StructuredHypothesis[voidptr]{
		structure_id:    name
		params:          [0.0]
		residual:        residual
		complexity:      complexity
		geometry_cost:   geometry_cost
		geometry_family: family
	}
}

fn map_sum(m map[string]f64) f64 {
	mut s := 0.0
	for _, v in m {
		s += v
	}
	return s
}

fn test_hierarchical_posterior_decomposes_and_normalizes() {
	mut estimates := map[string]StructuredHypothesis[voidptr]{}
	estimates['single'] = est('single', 0.4, 'single', 0.0, 0.0)
	estimates['composite'] = est('composite', 0.5, 'composite', 0.0, 0.0)
	estimates['composite_attach'] = est('composite_attach', 0.3, 'composite', 0.0, 0.0)
	out := GenericStructureGate[voidptr]{}.decide_hierarchical(estimates)
	assert math.abs(map_sum(out.posterior) - 1.0) < 1e-6
	for name, _ in estimates {
		fam := (estimates[name] or { continue }).geometry_family
		expected := out.family_posterior[fam] * out.family_conditional[fam][name]
		assert math.abs(out.posterior[name] - expected) < 1e-12
	}
	for _, cond in out.family_conditional {
		assert math.abs(map_sum(cond) - 1.0) < 1e-6
	}
	assert out.estimate.structure_id == 'composite_attach'
}

fn test_hierarchical_degenerates_to_flat_for_singletons() {
	mut estimates := map[string]StructuredHypothesis[voidptr]{}
	estimates['a'] = est('a', 0.2, '', 0.0, 0.0)
	estimates['b'] = est('b', 0.8, '', 0.0, 0.0)
	estimates['c'] = est('c', 1.5, '', 0.0, 0.0)
	gate := GenericStructureGate[voidptr]{}
	flat := gate.decide(estimates)
	hier := gate.decide_hierarchical(estimates)
	for name, _ in estimates {
		assert math.abs(flat.posterior[name] - hier.posterior[name]) < 1e-12
	}
}

fn test_temperature_scale_sharpens_posterior() {
	mut estimates := map[string]StructuredHypothesis[voidptr]{}
	estimates['a'] = est('a', 0.3, '', 0.0, 0.0)
	estimates['b'] = est('b', 0.5, '', 0.0, 0.0)
	sharp := GenericStructureGate[voidptr]{
		temperature_scale: 0.5
	}.decide_hierarchical(estimates)
	flat := GenericStructureGate[voidptr]{
		temperature_scale: 1.0
	}.decide_hierarchical(estimates)
	assert sharp.posterior['a'] > flat.posterior['a']
	assert sharp.posterior['b'] < flat.posterior['b']
}

fn test_negative_score_does_not_collapse_to_one_hot() {
	mut estimates := map[string]StructuredHypothesis[voidptr]{}
	estimates['a'] = est('a', 100.0, '', 0.0, -1.0)
	estimates['b'] = est('b', 200.0, '', 0.0, 0.0)
	out := GenericStructureGate[voidptr]{
		geometry_weight: 5000.0
	}.decide_hierarchical(estimates)
	assert out.estimate.structure_id == 'a'
	assert 0.5 < out.posterior['a'] && out.posterior['a'] < 0.99
}
