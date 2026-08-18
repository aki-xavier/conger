module conger

// structured_hypothesis.v — domain-independent structured hypothesis / posterior
// return object (V port of src/structured_hypothesis.py).
import mlx

pub struct HypothesisCandidate {
pub:
	params      []f64
	probability f64
	residual    f64
}

pub struct StructuredHypothesis {
pub:
	scene                 voidptr // opaque domain payload (conger-vision stores a heap-boxed cga.Scene); 0 = none
	params                []f64
	spn_posterior         ?mlx.Array
	structure_id          string = 'unknown'
	geometry_family       string
	template_delta        map[string]MetaValue
	representation        ?mlx.Array
	candidate_params      [][]f64
	candidate_scores      ?mlx.Array
	candidate_posterior   ?mlx.Array
	candidate_temperature f64
	hypotheses            []HypothesisCandidate
	factor_sizes          []int
	factor_indices        []int
	responsibility_max    f64
	posterior_entropy     f64
	residual              f64
	complexity            f64
	geometry_cost         f64
	novelty_score         f64
	structure_posterior   f64
	structure_posteriors  map[string]f64
	em_trajectory         []f64
}

// new_hypothesis returns a StructuredHypothesis with the visual factor defaults.
pub fn new_hypothesis() StructuredHypothesis {
	return StructuredHypothesis{
		factor_sizes:   [3, 6, 3, 3]
		factor_indices: [0, 5, 6, 7]
	}
}

// with_structure returns a copy with the structure-id/posterior fields replaced.
pub fn (est StructuredHypothesis) with_structure(id string, sp f64, sps map[string]f64) StructuredHypothesis {
	return StructuredHypothesis{
		scene:                 est.scene
		params:                est.params
		spn_posterior:         est.spn_posterior
		structure_id:          id
		geometry_family:       est.geometry_family
		template_delta:        est.template_delta
		representation:        est.representation
		candidate_params:      est.candidate_params
		candidate_scores:      est.candidate_scores
		candidate_posterior:   est.candidate_posterior
		candidate_temperature: est.candidate_temperature
		hypotheses:            est.hypotheses
		factor_sizes:          est.factor_sizes
		factor_indices:        est.factor_indices
		responsibility_max:    est.responsibility_max
		posterior_entropy:     est.posterior_entropy
		residual:              est.residual
		complexity:            est.complexity
		geometry_cost:         est.geometry_cost
		novelty_score:         est.novelty_score
		structure_posterior:   sp
		structure_posteriors:  sps
		em_trajectory:         est.em_trajectory
	}
}

// with_residual_geometry returns a copy with the residual / geometry-cost fields replaced.
pub fn (est StructuredHypothesis) with_residual_geometry(r f64, gc f64) StructuredHypothesis {
	return StructuredHypothesis{
		scene:                 est.scene
		params:                est.params
		spn_posterior:         est.spn_posterior
		structure_id:          est.structure_id
		geometry_family:       est.geometry_family
		template_delta:        est.template_delta
		representation:        est.representation
		candidate_params:      est.candidate_params
		candidate_scores:      est.candidate_scores
		candidate_posterior:   est.candidate_posterior
		candidate_temperature: est.candidate_temperature
		hypotheses:            est.hypotheses
		factor_sizes:          est.factor_sizes
		factor_indices:        est.factor_indices
		responsibility_max:    est.responsibility_max
		posterior_entropy:     est.posterior_entropy
		residual:              r
		complexity:            est.complexity
		geometry_cost:         gc
		novelty_score:         est.novelty_score
		structure_posterior:   est.structure_posterior
		structure_posteriors:  est.structure_posteriors
		em_trajectory:         est.em_trajectory
	}
}

// factor_marginals returns the per-factor marginal posteriors.
pub fn (h StructuredHypothesis) factor_marginals() []mlx.Array {
	sizes := if h.factor_sizes.len > 0 { h.factor_sizes } else { [3, 6, 3, 3] }
	indices := if h.factor_indices.len > 0 { h.factor_indices } else { [0, 5, 6, 7] }
	mut vals := [][]f64{len: sizes.len}
	for i, n in sizes {
		vals[i] = []f64{len: n}
	}
	if cp := h.candidate_posterior {
		if h.candidate_params.len > 0 {
			cvals := cp.data_f32()
			for pi, prm in h.candidate_params {
				for i, j in indices {
					vals[i][int(prm[j])] += f64(cvals[pi])
				}
			}
		}
	} else {
		for i, j in indices {
			vals[i][int(h.params[j])] = 1.0
		}
	}
	mut out := []mlx.Array{len: vals.len}
	for i, v in vals {
		out[i] = arr32(v, [v.len])
	}
	return out
}
