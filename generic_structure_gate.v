module conger

// generic_structure_gate.v — domain-independent structural-posterior gate
// (V port of src/generic_structure_gate.py). Pure-f64 softmax (the score arrays
// are tiny; no MLX needed).
import math

pub struct GenericStructureDecision[T] {
pub:
	estimate            StructuredHypothesis[T]
	posterior           map[string]f64
	residuals           map[string]f64
	scores              map[string]f64
	needs_new_structure bool
	family_posterior    map[string]f64
	family_conditional  map[string]map[string]f64
}

pub struct GenericStructureGate[T] {
pub:
	birth_residual    f64 = 1.0
	posterior_floor   f64 = -1.0 // -1.0 = None (no floor)
	priors            map[string]f64
	complexity_weight f64 = 0.0
	geometry_weight   f64 = 0.0
	temperature_scale f64 = 1.0
}

// softmax_map normalises scores into a posterior (order-independent softmax).
pub fn softmax_map(scores map[string]f64, temperature f64, priors map[string]f64) map[string]f64 {
	mut logp := map[string]f64{}
	mut maxlog := -1e300
	for n, s in scores {
		pr := if n in priors { priors[n] } else { 1.0 }
		lp := -s / temperature + math.log(pr)
		logp[n] = lp
		if lp > maxlog {
			maxlog = lp
		}
	}
	mut sum := 0.0
	for _, lp in logp {
		sum += math.exp(lp - maxlog)
	}
	lse := maxlog + math.log(sum)
	mut probs := map[string]f64{}
	for n, lp in logp {
		probs[n] = math.exp(lp - lse)
	}
	return probs
}

// min_of returns the smallest value, or `none` for an empty map (no +inf sentinel).
pub fn min_of(m map[string]f64) ?f64 {
	if m.len == 0 {
		return none
	}
	mut best := 0.0
	mut first := true
	for _, v in m {
		if first || v < best {
			best = v
			first = false
		}
	}
	return best
}

// max_of returns the largest value, or `none` for an empty map (no -inf sentinel).
pub fn max_of(m map[string]f64) ?f64 {
	if m.len == 0 {
		return none
	}
	mut best := 0.0
	mut first := true
	for _, v in m {
		if first || v > best {
			best = v
			first = false
		}
	}
	return best
}

// argmin_of returns the key with the smallest value, or `none` for an empty map.
pub fn argmin_of(m map[string]f64) ?string {
	if m.len == 0 {
		return none
	}
	mut best_name := ''
	mut best := 0.0
	mut first := true
	for n, v in m {
		if first || v < best {
			best = v
			best_name = n
			first = false
		}
	}
	return best_name
}

// scores_map returns (residuals, scores) for each expert.
pub fn (g GenericStructureGate[T]) scores_map(estimates map[string]StructuredHypothesis[T]) (map[string]f64, map[string]f64) {
	mut residuals := map[string]f64{}
	mut scores := map[string]f64{}
	for name, est in estimates {
		residuals[name] = est.residual
		scores[name] = est.residual + g.complexity_weight * est.complexity +
			g.geometry_weight * est.geometry_cost
	}
	return residuals, scores
}

// decide returns the flat (single-level) gate decision.
pub fn (g GenericStructureGate[T]) decide(estimates map[string]StructuredHypothesis[T]) GenericStructureDecision[T] {
	if estimates.len == 0 {
		panic('GenericStructureGate.decide: cannot gate an empty set of experts')
	}
	residuals, scores := g.scores_map(estimates)
	best_raw := min_of(residuals) or { 0.0 }
	best_score := min_of(scores) or { 0.0 }
	temperature := math.max(2.0 * math.abs(best_score), 1e-8) * g.temperature_scale
	posterior := softmax_map(scores, temperature, g.priors)
	best_name := argmin_of(scores) or { '' }
	best := (estimates[best_name] or { panic('unknown expert') }).with_structure(best_name,
		posterior[best_name], posterior)
	needs_new := best_raw > g.birth_residual
		&& (g.posterior_floor < 0.0 || posterior[best_name] < g.posterior_floor)
	return GenericStructureDecision[T]{
		estimate:            best
		posterior:           posterior
		residuals:           residuals
		scores:              scores
		needs_new_structure: needs_new
	}
}

// decide_hierarchical returns the two-level (family → member) gate decision.
pub fn (g GenericStructureGate[T]) decide_hierarchical(estimates map[string]StructuredHypothesis[T]) GenericStructureDecision[T] {
	if estimates.len == 0 {
		panic('GenericStructureGate.decide_hierarchical: cannot gate an empty set of experts')
	}
	residuals, scores := g.scores_map(estimates)
	best_raw := min_of(residuals) or { 0.0 }
	mut groups := map[string][]string{}
	for name, est in estimates {
		fam := if est.geometry_family != '' { est.geometry_family } else { name }
		groups[fam] << name
	}
	mut family_scores := map[string]f64{}
	for fam, names in groups {
		mut m := 1e300
		for n in names {
			if scores[n] < m {
				m = scores[n]
			}
		}
		family_scores[fam] = m
	}
	fam_temp := math.max(2.0 * math.abs(min_of(family_scores) or { 0.0 }), 1e-8) * g.temperature_scale
	family_posterior := softmax_map(family_scores, fam_temp, g.priors)
	mut family_conditional := map[string]map[string]f64{}
	mut posterior := map[string]f64{}
	for fam, names in groups {
		mut member_scores := map[string]f64{}
		for n in names {
			member_scores[n] = scores[n]
		}
		mem_temp := math.max(2.0 * math.abs(min_of(member_scores) or { 0.0 }), 1e-8) * g.temperature_scale
		cond := softmax_map(member_scores, mem_temp, g.priors)
		family_conditional[fam] = cond.clone()
		for n in names {
			posterior[n] = family_posterior[fam] * cond[n]
		}
	}
	best_name := argmin_of(scores) or { '' }
	best := (estimates[best_name] or { panic('unknown expert') }).with_structure(best_name,
		posterior[best_name], posterior)
	needs_new := best_raw > g.birth_residual
		&& (g.posterior_floor < 0.0 || posterior[best_name] < g.posterior_floor)
	return GenericStructureDecision[T]{
		estimate:            best
		posterior:           posterior
		residuals:           residuals
		scores:              scores
		needs_new_structure: needs_new
		family_posterior:    family_posterior
		family_conditional:  family_conditional
	}
}
