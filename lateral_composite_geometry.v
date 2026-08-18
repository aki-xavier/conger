module conger

// lateral_composite_geometry.v — mirror/repeat lateral composite part anchors
// (V port of src/lateral_composite_geometry.py). Shares the composite area /
// part-depth machinery but searches for a VERTICAL split line.

import math

import mlx

const lgc_near_cap_delta = [0.0, 1.1, 1.0]

// lgc_split_score searches for the vertical split → (score, left base, right part).
fn lgc_split_score(fg mlx.Array) ?SplitResult {
	fgd := cg_down_mask(fg)
	h := fgd.dim(0)
	w := fgd.dim(1)
	sel := nonzero_indices(fgd)
	if sel.dim(0) < 4 * cg_min_part_pixels {
		return none
	}
	ids := mlx.arange(0.0, f64(fgd.size()), 1.0, .int32).take(sel)
	xs_idx := ids.remainder(mlx.int_scalar(w))
	x0 := xs_idx.min().item_i32()
	x1 := xs_idx.max().item_i32()
	if x1 - x0 < 5 {
		return none
	}
	xx := mlx.arange(0.0, f64(w), 1.0, .float32).expand_dims(0).broadcast_to([h, w])
	mut best_score := 1e18
	mut best_base := LayerTemplate{}
	mut best_part := LayerTemplate{}
	mut found := false
	for split in x0 + 2 .. x1 - 1 {
		left := fgd.logical_and(xx.less_equal(mlx.f32_scalar(f32(split))))
		right := fgd.logical_and(xx.greater(mlx.f32_scalar(f32(split))))
		if cg_count(left) < cg_min_part_pixels || cg_count(right) < cg_min_part_pixels {
			continue
		}
		base := jlo_bbox_template(left, 0.0, none)
		part := jlo_bbox_template(right, 0.0, none)
		if base.cx >= part.cx {
			continue
		}
		dx := part.cx - base.cx
		rr := base.r + part.r
		if !(0.45 * rr <= dx && dx <= 3.0 * rr) {
			continue
		}
		if math.abs(base.cy - part.cy) > 0.8 * rr {
			continue
		}
		a0 := cg_area(base)
		a1 := cg_area(part)
		ratio := math.sqrt(a1 / fmax2(a0, 1e-8))
		mut ratio_penalty := 0.0
		if 0.25 - ratio > ratio_penalty {
			ratio_penalty = 0.25 - ratio
		}
		if ratio - 1.1 > ratio_penalty {
			ratio_penalty = ratio - 1.1
		}
		un_ := base.mask(h, w).logical_or(part.mask(h, w))
		inter := un_.logical_and(fgd).astype(.float32).sum().item_f32()
		union_n := un_.logical_or(fgd).astype(.float32).sum().item_f32()
		mask_cost := 1.0 - f64(inter) / fmax2(f64(union_n), 1.0)
		score := mask_cost + 2.0 * ratio_penalty
		if !found || score < best_score {
			found = true
			best_score = score
			best_base = base
			best_part = part
		}
	}
	if !found {
		return none
	}
	return SplitResult{score: best_score, base: best_base, part: best_part}
}

// lgc_corrected_gap returns the kind-aware near-cap-corrected world-normalised
// gap g = |x1-x0|/(s0+s1), or none when the split is unreliable.
fn lgc_corrected_gap(fl mlx.Array, fr mlx.Array, kind int) ?f64 {
	wl := foreground_weights(fl)
	fg := wl.greater(mlx.f32_scalar(0.01))
	sp := lgc_split_score(fg)
	if sp == none {
		return none
	}
	s := sp
	q := cg_down
	b := cg_disk_fit(fg, s.base, q)
	p := cg_disk_fit(fg, s.part, q)
	if b == none || p == none {
		return none
	}
	bb := b or { [0.0, 0.0, 0.0] }
	pp := p or { [0.0, 0.0, 0.0] }
	u0 := bb[0]
	r0 := bb[2]
	u1 := pp[0]
	r1 := pp[2]
	z_global, _, _ := StereoDepth{}.estimate(fl, fr)
	zc := cam_z - z_global
	mut delta := 1.1
	if 0 <= kind && kind < lgc_near_cap_delta.len {
		delta = lgc_near_cap_delta[kind]
	}
	s0 := r0 * zc / (fx + delta * r0)
	s1 := r1 * zc / (fx + delta * r1)
	c := f64(img_w - 1) / 2.0
	x0 := (u0 - c) * (zc - delta * s0) / fx
	x1 := (u1 - c) * (zc - delta * s1) / fx
	return math.abs(x1 - x0) / fmax2(s0 + s1, 1e-8)
}

// lgc_estimate returns [u,v,z,area]×2 for a lateral base/part composite
// (same machinery as composite, but with the vertical split).
fn lgc_estimate(fl mlx.Array, fr mlx.Array) []f64 {
	wl := foreground_weights(fl)
	wr := foreground_weights(fr)
	fg := wl.greater(mlx.f32_scalar(0.01))
	z_global, d_global, area_global := StereoDepth{}.estimate(fl, fr)
	sp := lgc_split_score(fg)
	q := f64(cg_down)
	if sp == none {
		u, v, _ := cg_centroid(wl)
		return [u, v, z_global, 0.7 * area_global, u, v, z_global, 0.3 * area_global]
	}
	s := sp
	base := s.base
	part := s.part
	z0 := cg_part_depth(wl, wr, base, d_global)
	z1 := cg_part_depth(wl, wr, part, d_global)
	return [base.cx * q, base.cy * q, z0, cg_area(base), part.cx * q, part.cy * q,
		z1, cg_area(part)]
}
