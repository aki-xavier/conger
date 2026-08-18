module conger

// composite_template_proposer.v — residual-driven bounded template proposals
// (V port of src/composite_template_proposer.py).
import cga
import math
import mlx

struct CompositeTemplateProposer {
	complexity_weight f64   = 1.0
	ratios            []f64 = [0.45, 0.60]
	lateral_ratios    []f64 = [-0.20, 0.0, 0.20]
	part_kinds        []int = [0, 1, 2]
	part_hues         []int = [0, 1, 2, 3, 4, 5]
	grammar           TemplateGrammar
	max_cases         int = 4
	max_proposals     int = 5
	codebook          CompositeCodebook
}

// new_composite_template_proposer builds the default proposer.
fn new_composite_template_proposer() CompositeTemplateProposer {
	return CompositeTemplateProposer{
		grammar:  new_template_grammar(['attach'], 2, [0, 1, 2])
		codebook: new_composite_codebook(InverseConfig{ scene_family: 'composite' })
	}
}

// ctp_observed_delta extracts a measured delta from a frame pair ({} if unreliable).
fn ctp_observed_delta(fl mlx.Array, fr mlx.Array, operation string, part_kind int) map[string]MetaValue {
	if operation == 'attach' || operation == 'layer' {
		ev := cg_disk_evidence(fl, fr)
		if e := ev {
			mut out := map[string]MetaValue{}
			out['scale_ratio'] = e[0]
			out['lateral_ratio'] = e[1]
			if operation == 'layer' {
				st := sl_estimate(fl, fr)
				if st.len >= 8 {
					out['depth_gap'] = math.abs(st[2] - st[6])
				}
			}
			return out
		}
		return map[string]MetaValue{}
	}
	if operation == 'mirror' || operation == 'repeat' {
		if gg := lgc_corrected_gap(fl, fr, part_kind) {
			mut out := map[string]MetaValue{}
			out['period_ratio'] = gg / lc_spacing_factor(operation)
			return out
		}
		return map[string]MetaValue{}
	}
	return map[string]MetaValue{}
}

// ctp_base_from_params maps 8- or 14-dim params → (k,u,v,s,z,hue,lcol,ldir).
fn ctp_base_from_params(params []f64) ?[]f64 {
	if params.len == 8 {
		return params.clone()
	}
	if params.len == 14 {
		return [params[0], params[1], params[2], params[3], params[4], params[5], params[12], params[13]]
	}
	return none
}

// ctp_attach builds a 14-dim attached composite.
fn ctp_attach(base []f64, part_kind int, part_hue int, ratio f64, lateral_ratio f64) []f64 {
	k0 := base[0]
	u0 := base[1]
	v0 := base[2]
	s0 := base[3]
	z0 := base[4]
	h0 := base[5]
	lcol := base[6]
	ldir := base[7]
	s1 := s0 * ratio
	z1 := z0
	x0, y0 := unproject(u0, v0, z0)
	x1 := x0 + lateral_ratio * (s0 + s1)
	y1 := y0 + s0 + s1 - 0.05 * math_min_f64(s0, s1)
	zc1 := cam_z - z1
	u1 := f64(img_w - 1) / 2.0 + x1 * fx / zc1
	v1 := f64(img_h - 1) / 2.0 - y1 * fy / zc1
	return [k0, u0, v0, s0, z0, h0, f64(part_kind), u1, v1, s1, z1, f64(part_hue), lcol, ldir]
}

// ctp_layer builds a 14-dim layered (part behind base).
fn ctp_layer(base []f64, part_kind int, part_hue int, ratio f64, lateral_ratio f64) []f64 {
	k0 := base[0]
	u0 := base[1]
	v0 := base[2]
	s0 := base[3]
	z0 := base[4]
	h0 := base[5]
	lcol := base[6]
	ldir := base[7]
	s1 := s0 * ratio
	z1 := fmax2(2.2, z0 - 0.8)
	x0, y0 := unproject(u0, v0, z0)
	x1 := x0 + lateral_ratio * (s0 + s1)
	zc1 := cam_z - z1
	u1 := f64(img_w - 1) / 2.0 + x1 * fx / zc1
	v1 := f64(img_h - 1) / 2.0 - y0 * fy / zc1
	return [k0, u0, v0, s0, z0, h0, f64(part_kind), u1, v1, s1, z1, f64(part_hue), lcol, ldir]
}

// ctp_lateral builds a 14-dim mirror/repeat composite (same kind/hue).
fn ctp_lateral(base []f64, operation string, ratio f64, lateral_ratio f64) []f64 {
	k0 := base[0]
	u0 := base[1]
	v0 := base[2]
	s0 := base[3]
	z0 := base[4]
	h0 := base[5]
	lcol := base[6]
	ldir := base[7]
	s1 := s0 * ratio
	z1 := z0
	x0, y0 := unproject(u0, v0, z0)
	scale := lc_spacing_factor(operation)
	x1 := x0 + lateral_ratio * scale * (s0 + s1)
	zc1 := cam_z - z1
	u1 := f64(img_w - 1) / 2.0 + x1 * fx / zc1
	v1 := f64(img_h - 1) / 2.0 - y0 * fy / zc1
	return [k0, u0, v0, s0, z0, h0, k0, u1, v1, s1, z1, h0, lcol, ldir]
}

// ctp_params_for_rule builds candidate params for a grammar rule.
fn (p CompositeTemplateProposer) ctp_params_for_rule(base []f64, rule TemplateRule, part_hue int, ratio f64, lateral f64) []f64 {
	rule_base := [f64(rule.base_kind), base[1], base[2], base[3], base[4], base[5], base[6], base[7]]
	if rule.operation == 'attach' {
		return ctp_attach(rule_base, rule.part_kind, part_hue, ratio, lateral)
	}
	if rule.operation == 'layer' {
		return ctp_layer(rule_base, rule.part_kind, part_hue, ratio, lateral)
	}
	return ctp_lateral(rule_base, rule.operation, ratio, lateral)
}

// ctp_render_residual re-renders a candidate and returns the foreground MSE.
fn (p CompositeTemplateProposer) ctp_render_residual(params []f64, fl mlx.Array, fr mlx.Array, mut renderer cga.Renderer, cam_l cga.PerspectiveCamera, cam_r cga.PerspectiveCamera) f64 {
	scene := p.codebook.to_scene(params)
	cl := renderer.render(scene, cam_l)
	cr := renderer.render(scene, cam_r)
	wl := foreground_weights(fl)
	wr := foreground_weights(fr)
	return 0.5 * (sr_masked_mse(fl, cl, wl) + sr_masked_mse(fr, cr, wr))
}

// ctp_propose_case enumerates grammar candidates for one birth case.
fn (p CompositeTemplateProposer) ctp_propose_case(case StructureCase, case_index int) []TemplateProposal {
	b := ctp_base_from_params(case.params) or { return []TemplateProposal{} }
	fl := case.fl
	fr := case.fr
	mut baseline := 1e18
	for _, v in case.residuals {
		if v < baseline {
			baseline = v
		}
	}
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	mut out := []TemplateProposal{}
	for rule in p.grammar.composites() {
		mut hues := p.part_hues.clone()
		if rule.operation == 'mirror' || rule.operation == 'repeat' {
			hues = [int(b[5])]
		} else if rule.part_kind !in p.part_kinds {
			continue
		}
		for part_hue in hues {
			for ratio in p.ratios {
				for lateral in p.lateral_ratios {
					if (rule.operation == 'mirror' || rule.operation == 'repeat')
						&& math.abs(lateral) < 1e-12 {
						continue
					}
					params := p.ctp_params_for_rule(b, rule, part_hue, ratio, lateral)
					if !(lcb_inside(params[1], params[2], params[3], params[4])
						&& lcb_inside(params[7], params[8], params[9], params[10])) {
						continue
					}
					residual := p.ctp_render_residual(params, fl, fr, mut renderer, cam_l, cam_r)
					score := residual + p.complexity_weight * rule.complexity
					family := if rule.operation == 'layer' { 'layered' } else { 'composite' }
					default_parent := if rule.operation == 'attach' { 'layered' } else { 'single' }
					parent_family := if case.structure_id != 'unknown' {
						case.structure_id
					} else {
						default_parent
					}
					mut delta := map[string]MetaValue{}
					delta['relation'] = rule.operation
					delta['base_kind'] = rule.base_kind
					delta['part_kind'] = rule.part_kind
					delta['part_hue'] = part_hue
					delta['ratio'] = ratio
					delta['lateral_ratio'] = lateral
					if rule.operation == 'layer' {
						delta['depth_gap'] = 0.8
					}
					mut metadata := map[string]MetaValue{}
					metadata['signature'] = rule.signature()
					metadata['relation'] = rule.operation
					metadata['base_kind'] = rule.base_kind
					metadata['part_kind'] = rule.part_kind
					metadata['part_hue'] = part_hue
					metadata['ratio'] = ratio
					metadata['lateral_ratio'] = lateral
					metadata['case_index'] = case_index
					metadata['residual_gain'] = baseline - residual
					out << TemplateProposal{
						family:        family
						operation:     rule.operation
						params:        params
						residual:      residual
						complexity:    rule.complexity
						score:         score
						parent_family: parent_family
						delta:         delta
						metadata:      metadata
					}
				}
			}
		}
	}
	return out
}

// ctp_propose aggregates the top proposals across the first max_cases.
fn (p CompositeTemplateProposer) ctp_propose(cases []StructureCase) []TemplateProposal {
	mut proposals := []TemplateProposal{}
	limit := if cases.len < p.max_cases { cases.len } else { p.max_cases }
	for i in 0 .. limit {
		proposals << p.ctp_propose_case(cases[i], i)
	}
	proposals.sort_with_compare(fn (a &TemplateProposal, b &TemplateProposal) int {
		if a.score < b.score {
			return -1
		}
		if a.score > b.score {
			return 1
		}
		return 0
	})
	if proposals.len > p.max_proposals {
		return proposals[..p.max_proposals]
	}
	return proposals
}
