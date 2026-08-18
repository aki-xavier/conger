module conger

// structure_geometry.v — observation-level geometric compatibility evidence
// for the structure gate (V port of src/structure_geometry.py).

import math

import mlx

// sg_iou_cost returns 1 − IoU(template, fg).
fn sg_iou_cost(template mlx.Array, fg mlx.Array) f64 {
	inter := template.logical_and(fg).astype(.float32).sum().item_f32()
	un_ := template.logical_or(fg).astype(.float32).sum().item_f32()
	return 1.0 - f64(inter) / fmax2(f64(un_), 1.0)
}

// sg_single_cost returns the single-template compactness cost.
fn sg_single_cost(fg mlx.Array, split_score ?f64) f64 {
	fgd := jlo_down_mask(fg)
	sel := nonzero_indices(fgd)
	if sel.dim(0) < 8 {
		return 1.0
	}
	h := fgd.dim(0)
	w := fgd.dim(1)
	ids := mlx.arange(0.0, f64(fgd.size()), 1.0, .int32).take(sel)
	xs := ids.remainder(mlx.int_scalar(w)).astype(.float32)
	ys := ids.floor_divide(mlx.int_scalar(w)).astype(.float32)
	height := f64(ys.max().item_f32() - ys.min().item_f32() + 1.0)
	width := f64(xs.max().item_f32() - xs.min().item_f32() + 1.0)
	aspect := fmax2(height / fmax2(width, 1.0), width / fmax2(height, 1.0))
	tmpl := jlo_bbox_template(fgd, 0.0, none)
	mut cost := sg_iou_cost(tmpl.mask(h, w), fgd)
	cost += 0.5 * clamp01(aspect - 1.35)
	if ss := split_score {
		cost += 0.5 * clamp01(0.35 - ss)
	}
	return clamp01(cost)
}

// sg_layered_cost returns the two-layer disparity/separation cost.
fn sg_layered_cost(fl mlx.Array, fr mlx.Array, fg mlx.Array) f64 {
	disp, _, valid0 := sl_disparity_map(fl, fr)
	valid := valid0.logical_and(fg)
	fw := foreground_weights(fl)
	clustered := sl_cluster_layers(disp, fw, valid)
	c := clustered or { return 1.0 }
	c_front := c.front_c
	c_back := c.back_c
	separation := c_front[2] - c_back[2]
	dx := c_front[0] - c_back[0]
	dy := c_front[1] - c_back[1]
	spatial := math.sqrt(dx * dx + dy * dy)
	separation_cost := clamp01((2.5 - separation) / 2.5)
	spatial_cost := clamp01((20.0 - spatial) / 20.0)
	return fmax2(separation_cost, spatial_cost)
}

// sg_composite_cost returns the attached-composite contact cost.
fn sg_composite_cost(fl mlx.Array, fr mlx.Array, split ?SplitResult) f64 {
	s := split or { return 1.0 }
	split_cost := s.score
	st := cg_estimate(fl, fr)
	depth_gap := math.abs(st[2] - st[6])
	depth_cost := clamp01((depth_gap - 0.15) / 0.5)
	return clamp01(split_cost + 0.7 * depth_cost)
}

// sg_lateral_cost returns the lateral composite cost.
fn sg_lateral_cost(fl mlx.Array, fr mlx.Array, fg mlx.Array) f64 {
	sp := lgc_split_score(fg)
	s := sp or { return 1.0 }
	split_cost := s.score
	st := lgc_estimate(fl, fr)
	depth_gap := math.abs(st[2] - st[6])
	depth_cost := clamp01((depth_gap - 0.15) / 0.5)
	return clamp01(split_cost + 0.7 * depth_cost)
}

// sg_geometry_stats returns [u,v,z,area]×2 for a base structure family.
fn sg_geometry_stats(family string, fl mlx.Array, fr mlx.Array) ?[]f64 {
	if family == 'layered' {
		return sl_estimate(fl, fr)
	}
	if family == 'composite' {
		return cg_estimate(fl, fr)
	}
	if family == 'lateral' {
		return lgc_estimate(fl, fr)
	}
	return none
}

// sg_range_term is the narrow-support log-evidence term for a delta range.
fn sg_range_term(delta map[string]MetaValue, key string, observed f64, default_lo f64, default_hi f64) f64 {
	lst := meta_list(delta, key)
	if lst.len == 0 {
		return 0.0
	}
	lo := lst[0]
	hi := lst[1]
	outside := fmax2(lo - observed, fmax2(observed - hi, 0.0))
	if outside > 0.0 {
		return 4.0 * outside
	}
	width := fmax2(hi - lo, 1e-6)
	default_width := default_hi - default_lo
	return 0.25 * math.log(width / default_width)
}

// sg_lateral_gap_core is the pure mirror/repeat discriminant logic (g is the
// near-cap-corrected world-normalised gap). Split out so it can be unit-tested
// without rendering.
fn sg_lateral_gap_core(relation string, delta map[string]MetaValue, g f64) f64 {
	if relation != 'mirror' && relation != 'repeat' {
		return 0.0
	}
	spacing := lc_spacing_factor(relation)
	other_spacing := lc_spacing_factor(if relation == 'mirror' { 'repeat' } else { 'mirror' })
	learned := meta_list(delta, 'period_ratio')
	mut lo := lc_part_period_lo
	mut hi := lc_part_period_hi
	if learned.len == 2 {
		lo = learned[0]
		hi = learned[1]
	}
	own_p := g / spacing
	other_p := g / other_spacing
	own_out := fmax2(lo - own_p, fmax2(own_p - hi, 0.0))
	mut cost := 4.0 * own_out
	if own_out > 0.0 && lc_part_period_lo <= other_p && other_p <= lc_part_period_hi {
		cost += 1.0
	}
	return fmin2(cost, 2.0)
}

// sg_lateral_gap_cost returns the mirror/repeat lateral-gap evidence.
fn sg_lateral_gap_cost(relation string, delta map[string]MetaValue, fl ?mlx.Array, fr ?mlx.Array) f64 {
	if relation != 'mirror' && relation != 'repeat' {
		return 0.0
	}
	if fl == none || fr == none {
		return 0.0
	}
	mut kind := 1
	pks := meta_list(delta, 'part_kinds')
	if pks.len > 0 {
		kind = int(pks[0])
	}
	f := fl or { return 0.0 }
	gframe := fr or { return 0.0 }
	if gap := lgc_corrected_gap(f, gframe, kind) {
		return sg_lateral_gap_core(relation, delta, gap)
	}
	return 0.0
}

// sg_delta_cost returns the child-template delta match/specificity cost.
fn sg_delta_cost(family string, delta map[string]MetaValue, stats ?[]f64, fl ?mlx.Array, fr ?mlx.Array) f64 {
	if delta.len == 0 || stats == none {
		return 0.0
	}
	st := stats or { return 0.0 }
	u0 := st[0]
	z0 := st[2]
	a0 := st[3]
	u1 := st[4]
	z1 := st[6]
	a1 := st[7]
	q0 := math.sqrt(fmax2(a0, 1e-8) / math.pi) * (cam_z - z0) / fx
	q1 := math.sqrt(fmax2(a1, 1e-8) / math.pi) * (cam_z - z1) / fx
	mut observed_ratio := q1 / fmax2(q0, 1e-8)
	x_gap := math.abs((u1 - u0) * (cam_z - (z0 + z1) / 2.0) / fx)
	mut lateral := x_gap / fmax2(q0 + q1, 1e-8)
	if family == 'composite' {
		if f := fl {
			if g := fr {
				if ev := cg_disk_evidence(f, g) {
					observed_ratio = ev[0]
					lateral = ev[1]
				}
			}
		}
	}
	mut cost := 0.0
	cost += sg_range_term(delta, 'scale_ratio', observed_ratio, 0.35, 0.75)
	relation := meta_str(delta, 'relation')
	if relation == 'mirror' || relation == 'repeat' {
		cost += sg_lateral_gap_cost(relation, delta, fl, fr)
	} else {
		if relation == 'attach' {
			cost += sg_range_term(delta, 'lateral_ratio', lateral, -0.25, 0.25)
		} else {
			cost += sg_range_term(delta, 'lateral_ratio', lateral, -0.75, 0.75)
		}
	}
	return cost
}

// sg_costs returns the per-family geometry costs for one observation.
fn sg_costs(fl mlx.Array, fr mlx.Array) map[string]f64 {
	fw := foreground_weights(fl)
	fg := fw.greater(mlx.f32_scalar(0.01))
	sp := cg_split_score(fg)
	mut split_score := ?f64(none)
	if s := sp {
		split_score = s.score
	}
	mut single := sg_single_cost(fg, split_score)
	mut layered := sg_layered_cost(fl, fr, fg)
	mut composite := sg_composite_cost(fl, fr, sp)
	mut lateral := sg_lateral_cost(fl, fr, fg)
	// a lateral composite whose evidence clearly beats the horizontal contact
	// line must not be misread as an attach composite
	if lateral < composite - 0.15 && lateral < 0.4 {
		composite = fmax2(composite, 1.6)
	}
	layered_raw := layered
	if single < 0.35 && layered_raw > 0.5 {
		single -= 1.3
	}
	lateral_blocks_layer := lateral < 0.35 && layered_raw > 0.05
	if !lateral_blocks_layer {
		if layered < composite - 0.03 && layered < single + 0.05 {
			layered -= 1.0
		}
		if layered + 0.05 < fmin2(single, composite) {
			layered -= 2.0
		}
	}
	if composite < 0.10 {
		composite -= 0.2
	}
	if lateral < 0.10 {
		lateral -= 0.2
	}
	return {
		'single': single
		'layered': layered
		'composite': composite
		'lateral': lateral
	}
}
