module conger

// structured_hypothesis.v — domain-independent structured hypothesis / posterior
// return object (V port of src/structured_hypothesis.py). Generic over the
// opaque scene payload `T` (conger-vision uses `cga.Scene`, the non-visual
// validation domain uses `voidptr`), so the generic core never type-erases the
// domain payload.
import mlx

pub struct HypothesisCandidate {
pub:
	params      []f64
	probability f64
	residual    f64
}

pub struct StructuredHypothesis[T] {
pub:
	scene                 T // domain payload (conger-vision: cga.Scene); zero value = none
	params                []f64
	spn_posterior         ?mlx.Array
	structure_id          string = 'unknown'
	geometry_family       string
	template_delta        TemplateConstraints
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
pub fn new_hypothesis[T]() StructuredHypothesis[T] {
	return StructuredHypothesis[T]{
		factor_sizes:   [3, 6, 3, 3]
		factor_indices: [0, 5, 6, 7]
	}
}

// with_structure returns a copy with the structure-id/posterior fields replaced.
pub fn (est StructuredHypothesis[T]) with_structure(id string, sp f64, sps map[string]f64) StructuredHypothesis[T] {
	return StructuredHypothesis[T]{
		...est
		structure_id:         id
		structure_posterior:  sp
		structure_posteriors: sps
	}
}

// with_residual_geometry returns a copy with the residual / geometry-cost fields replaced.
pub fn (est StructuredHypothesis[T]) with_residual_geometry(r f64, gc f64) StructuredHypothesis[T] {
	return StructuredHypothesis[T]{
		...est
		residual:      r
		geometry_cost: gc
	}
}

// factor_marginals returns the per-factor marginal posteriors.
pub fn (h StructuredHypothesis[T]) factor_marginals() []mlx.Array {
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
