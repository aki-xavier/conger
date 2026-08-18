module conger

// causal_edge.v — structure-level causal discovery: upgrade template-delta edges
// into candidate causal edges (V port of src/causal_edge.py).

pub struct CausalEdge {
pub:
	parent_family string
	operation     string
	target        string
	env_midpoints []f64
	env_ranges    [][]f64
	pooled_range  []f64
	agreement     f64
	n_envs        int
}

// is_causal requires ≥2 environments of cross-env evidence and low drift.
pub fn (e CausalEdge) is_causal() bool {
	return e.n_envs >= 2 && e.agreement >= 0.5
}

pub struct CausalDeltaLearner {
pub:
	agreement_threshold f64 = 0.5
}

// targets extracts scalar targets from a proposal (observed delta overrides grid).
pub fn (l CausalDeltaLearner) targets(p TemplateProposal) map[string]f64 {
	observed := meta_map(p.metadata, 'observed')
	mut out := map[string]f64{}
	if 'scale_ratio' in observed {
		out['scale_ratio'] = observed['scale_ratio']
	} else if v := p.delta['ratio'] {
		out['scale_ratio'] = meta_as_f64(v)
	}
	if 'period_ratio' in observed {
		out['period_ratio'] = observed['period_ratio']
	} else if 'lateral_ratio' in observed {
		key := if p.operation == 'mirror' || p.operation == 'repeat' {
			'period_ratio'
		} else {
			'lateral_ratio'
		}
		out[key] = observed['lateral_ratio']
	} else if v := p.delta['lateral_ratio'] {
		key := if p.operation == 'mirror' || p.operation == 'repeat' {
			'period_ratio'
		} else {
			'lateral_ratio'
		}
		out[key] = meta_as_f64(v)
	}
	if 'depth_gap' in observed {
		out['depth_gap'] = observed['depth_gap']
	} else if v := p.delta['depth_gap'] {
		out['depth_gap'] = meta_as_f64(v)
	}
	return out
}

// default_env_key returns env → seed → case_index → "0".
pub fn default_env_key(p TemplateProposal) string {
	if v := p.metadata['env'] {
		return meta_value_str(v)
	}
	if v := p.metadata['seed'] {
		return meta_value_str(v)
	}
	if v := p.metadata['case_index'] {
		return meta_value_str(v)
	}
	return '0'
}

// agreement returns 1 − midpoint drift / pooled width (∈[0,1]).
pub fn (l CausalDeltaLearner) agreement(mids []f64, ranges [][]f64) f64 {
	if mids.len <= 1 {
		return 1.0
	}
	mut lo := 1e300
	mut hi := -1e300
	for r in ranges {
		if r[0] < lo {
			lo = r[0]
		}
		if r[1] > hi {
			hi = r[1]
		}
	}
	width := hi - lo
	if width < 1e-12 {
		return 1.0
	}
	mut min_mid := 1e300
	mut max_mid := -1e300
	for m in mids {
		if m < min_mid {
			min_mid = m
		}
		if m > max_mid {
			max_mid = m
		}
	}
	drift := max_mid - min_mid
	a := 1.0 - drift / width
	if a < 0.0 {
		return 0.0
	}
	if a > 1.0 {
		return 1.0
	}
	return a
}

// learn groups proposals by (parent, operation, target) × env and returns edges.
pub fn (l CausalDeltaLearner) learn(proposals []TemplateProposal) []CausalEdge {
	mut groups := map[string]map[string][]f64{}
	mut order := []string{}
	for p in proposals {
		if p.parent_family == '' {
			continue
		}
		env := default_env_key(p)
		for target, val in l.targets(p) {
			key := '${p.parent_family}|${p.operation}|${target}'
			if key !in groups {
				groups[key] = map[string][]f64{}
				order << key
			}
			groups[key][env] << val
		}
	}
	mut edges := []CausalEdge{}
	for key in order {
		parts := key.split('|')
		parent := parts[0]
		op := parts[1]
		target := parts[2]
		mut ranges := [][]f64{}
		mut mids := []f64{}
		for _, vals in groups[key] {
			mut lo := 1e300
			mut hi := -1e300
			for v in vals {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			ranges << [lo, hi]
			mids << 0.5 * (lo + hi)
		}
		mut pooled_lo := 1e300
		mut pooled_hi := -1e300
		for r in ranges {
			if r[0] < pooled_lo {
				pooled_lo = r[0]
			}
			if r[1] > pooled_hi {
				pooled_hi = r[1]
			}
		}
		ag := l.agreement(mids, ranges)
		edges << CausalEdge{
			parent_family: parent
			operation:     op
			target:        target
			env_midpoints: mids
			env_ranges:    ranges
			pooled_range:  [pooled_lo, pooled_hi]
			agreement:     ag
			n_envs:        mids.len
		}
	}
	// sort by (-agreement, operation, target)
	for i in 0 .. edges.len {
		for j := i + 1; j < edges.len; j++ {
			if edges[j].agreement > edges[i].agreement
				|| (edges[j].agreement == edges[i].agreement
				&& (edges[j].operation < edges[i].operation
				|| (edges[j].operation == edges[i].operation && edges[j].target < edges[i].target))) {
				edges[i], edges[j] = edges[j], edges[i]
			}
		}
	}
	return edges
}

pub fn meta_as_f64(v MetaValue) f64 {
	return v as f64
}

pub fn meta_value_str(v MetaValue) string {
	match v {
		string { return v }
		int { return v.str() }
		f64 { return v.str() }
		map[string]f64 { return '' }
		else { return '' }
	}
}
