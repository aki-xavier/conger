module conger

// template_delta_learner.v — estimate child-template constraints from birth
// proposals (V port of src/template_delta_learner.py).

import math

struct TemplateDeltaLearner {
	min_evidence int   = 2
	range_margin f64  = 0.10
}

fn tdl_isfinite(v f64) bool {
	return !math.is_nan(v) && !math.is_inf(v, 0)
}

// tdl_range returns a numeric evidence range with relative/absolute margin.
fn (tdl TemplateDeltaLearner) tdl_range(values []f64) []f64 {
	mut lo := values[0]
	mut hi := values[0]
	for v in values {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	pad := fmax2((hi - lo) * tdl.range_margin, 0.02)
	return [lo - pad, hi + pad]
}

// tdl_repr serialises a MetaValue for the spec-name digest.
fn tdl_repr(v MetaValue) string {
	match v {
		string { return 's:${v}' }
		int { return 'i:${v}' }
		f64 { return 'f:${v}' }
		[]f64 {
			mut parts := []string{}
			for x in v {
				parts << x.str()
			}
			return 'l:${parts.join(',')}'
		}
		else { return 'o' }
	}
}

// tdl_hash returns a short deterministic digest of the constraints.
fn tdl_hash(constraints map[string]MetaValue) string {
	mut keys := []string{}
	for k, _ in constraints {
		keys << k
	}
	keys.sort()
	mut text := ''
	for k in keys {
		text += k + '=' + tdl_repr(constraints[k]) + ';'
	}
	mut h := u32(2166136261)
	for b in text.bytes() {
		h ^= u32(b)
		h *= 16777619
	}
	return h.hex()
}

// tdl_spec builds one ChildTemplateSpec from a proposal group.
fn (tdl TemplateDeltaLearner) tdl_spec(parent string, operation string, proposals []TemplateProposal, lineages map[string]TemplateLineage) ChildTemplateSpec {
	mut ratios := []f64{}
	mut laterals := []f64{}
	mut depth_gaps := []f64{}
	mut depth_jitters := [][]f64{}
	mut part_kinds := map[int]bool{}
	mut part_hues := map[int]bool{}
	for p in proposals {
		if 'ratio' in p.delta {
			ratios << meta_f64(p.delta, 'ratio')
		}
		if 'lateral_ratio' in p.delta {
			laterals << meta_f64(p.delta, 'lateral_ratio')
		}
		if 'depth_gap' in p.delta {
			depth_gaps << meta_f64(p.delta, 'depth_gap')
		}
		if 'depth_jitter' in p.delta {
			depth_jitters << meta_list(p.delta, 'depth_jitter')
		}
		if 'part_kind' in p.delta {
			part_kinds[meta_int(p.delta, 'part_kind')] = true
		}
		if 'part_hue' in p.delta {
			part_hues[meta_int(p.delta, 'part_hue')] = true
		}
	}
	mut constraints := map[string]MetaValue{}
	constraints['relation'] = operation
	if ratios.len > 0 {
		constraints['scale_ratio'] = tdl.tdl_range(ratios)
	}
	if laterals.len > 0 {
		if operation == 'mirror' || operation == 'repeat' {
			mut absv := []f64{len: laterals.len}
			for i, v in laterals {
				absv[i] = math.abs(v)
			}
			constraints['period_ratio'] = tdl.tdl_range(absv)
		} else {
			constraints['lateral_ratio'] = tdl.tdl_range(laterals)
		}
	}
	if depth_gaps.len > 0 {
		constraints['depth_gap'] = tdl.tdl_range(depth_gaps)
	}
	if depth_jitters.len > 0 {
		mut lo := 1e18
		mut hi := -1e18
		for dj in depth_jitters {
			if dj[0] < lo {
				lo = dj[0]
			}
			if dj[1] > hi {
				hi = dj[1]
			}
		}
		constraints['depth_jitter'] = [lo, hi]
	}
	mut kinds := []f64{}
	for k, _ in part_kinds {
		kinds << f64(k)
	}
	kinds.sort()
	constraints['part_kinds'] = kinds
	mut hues := []f64{}
	for h, _ in part_hues {
		hues << f64(h)
	}
	hues.sort()
	constraints['part_hues'] = hues

	mut generation := 1
	if parent in lineages {
		generation = lineages[parent].generation + 1
	}
	mut complexity_sum := 0.0
	mut residual_sum := 0.0
	mut score_sum := 0.0
	for p in proposals {
		complexity_sum += p.complexity
		residual_sum += p.residual
		score_sum += p.score
	}
	n := f64(proposals.len)
	return ChildTemplateSpec{
		name: '${parent}_${operation}_${tdl_hash(constraints)}'
		family: proposals[0].family
		parent_family: parent
		operation: operation
		constraints: constraints
		complexity: complexity_sum / n
		generation: generation
		evidence_count: proposals.len
		residual_mean: residual_sum / n
		score_mean: score_sum / n
	}
}

// tdl_learn aggregates proposals → ChildTemplateSpec list.
fn (tdl TemplateDeltaLearner) tdl_learn(requests []StructureBirthRequest, lineages map[string]TemplateLineage) []ChildTemplateSpec {
	mut groups := map[string][]TemplateProposal{}
	mut order := []string{}
	for request in requests {
		for proposal in request.proposals {
			if proposal.parent_family == '' {
				continue
			}
			if !tdl_isfinite(proposal.residual) || !tdl_isfinite(proposal.score) {
				continue
			}
			key := proposal.parent_family + '|' + proposal.operation
			if key !in groups {
				groups[key] = []TemplateProposal{}
				order << key
			}
			groups[key] << proposal
		}
	}
	mut specs := []ChildTemplateSpec{}
	for _, key in order {
		props := groups[key]
		if props.len < tdl.min_evidence {
			continue
		}
		specs << tdl.tdl_spec(props[0].parent_family, props[0].operation, props, lineages)
	}
	// deterministic order: (score_mean asc, evidence_count desc, name asc)
	specs.sort_with_compare(fn (a &ChildTemplateSpec, b &ChildTemplateSpec) int {
		if a.score_mean != b.score_mean {
			return if a.score_mean < b.score_mean { -1 } else { 1 }
		}
		if a.evidence_count != b.evidence_count {
			return if a.evidence_count > b.evidence_count { -1 } else { 1 }
		}
		return if a.name < b.name { -1 } else { 1 }
	})
	return specs
}
