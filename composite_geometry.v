module conger

// composite_geometry.v — part-aware binocular geometry anchors for attached
// composites (V port of src/composite_geometry.py).
import math
import mlx

const cg_down = 3
const cg_min_part_pixels = 5

// cg_down_mask max-pools a bool mask by q×q blocks.
fn cg_down_mask(mask mlx.Array) mlx.Array {
	q := cg_down
	h := mask.dim(0)
	w := mask.dim(1)
	hh := h / q
	ww := w / q
	m := mask.take_axis(mlx.arange(0.0, f64(hh * q), 1.0, .int32), 0).take_axis(mlx.arange(0.0,
		f64(ww * q), 1.0, .int32), 1)
	return m.reshape([hh, q, ww, q]).max_axes([1, 3], false).greater(mlx.f32_scalar(0.0))
}

// cg_centroid returns (u, v, total) of a weight map.
fn cg_centroid(weights mlx.Array) (f64, f64, f64) {
	h := weights.dim(0)
	w := weights.dim(1)
	ys := mlx.arange(0.0, f64(h), 1.0, .float32).expand_dims(1).broadcast_to([h, w])
	xs := mlx.arange(0.0, f64(w), 1.0, .float32).expand_dims(0).broadcast_to([h, w])
	total := weights.sum().item_f32()
	if total <= 1e-8 {
		return f64(w - 1) / 2.0, f64(h - 1) / 2.0, 0.0
	}
	u := weights.multiply(xs).sum().item_f32() / total
	v := weights.multiply(ys).sum().item_f32() / total
	return f64(u), f64(v), f64(total)
}

// cg_area returns the physical area of a template.
fn cg_area(t LayerTemplate) f64 {
	return (if t.shape == 2 {
		4.0
	} else {
		math.pi
	}) * (t.r * f64(cg_down)) * (t.r * f64(cg_down))
}

// cg_count returns the number of true pixels in a bool mask.
fn cg_count(m mlx.Array) int {
	return m.astype(.int32).sum().item_i32()
}

// cg_disk_fit fits a full-resolution disk within a template footprint.
fn cg_disk_fit(fg mlx.Array, tmpl LayerTemplate, q int) ?[]f64 {
	h := fg.dim(0)
	w := fg.dim(1)
	cx := tmpl.cx * f64(q)
	cy := tmpl.cy * f64(q)
	rad := tmpl.r * f64(q) * 1.15
	ys := mlx.arange(0.0, f64(h), 1.0, .float32).expand_dims(1).broadcast_to([h, w])
	xs := mlx.arange(0.0, f64(w), 1.0, .float32).expand_dims(0).broadcast_to([h, w])
	fp :=
		xs.subtract(mlx.f32_scalar(f32(cx))).square().add(ys.subtract(mlx.f32_scalar(f32(cy))).square()).less_equal(mlx.f32_scalar(f32(rad * rad)))
	m := fg.logical_and(fp)
	tot := m.astype(.float32).sum().item_f32()
	if tot < 1e-6 {
		return none
	}
	cx2 := m.astype(.float32).multiply(xs).sum().item_f32() / tot
	cy2 := m.astype(.float32).multiply(ys).sum().item_f32() / tot
	return [f64(cx2), f64(cy2), math.sqrt(f64(tot) / math.pi)]
}

// SplitResult is a composite split (score + base/part templates).
struct SplitResult {
	score f64
	base  LayerTemplate
	part  LayerTemplate
}

// cg_split_score searches for the horizontal contact line → (score, base, part).
fn cg_split_score(fg mlx.Array) ?SplitResult {
	fgd := cg_down_mask(fg)
	h := fgd.dim(0)
	w := fgd.dim(1)
	sel := nonzero_indices(fgd)
	if sel.dim(0) < 4 * cg_min_part_pixels {
		return none
	}
	ids := mlx.arange(0.0, f64(fgd.size()), 1.0, .int32).take(sel)
	ys_idx := ids.floor_divide(mlx.int_scalar(w))
	y0 := ys_idx.min().item_i32()
	y1 := ys_idx.max().item_i32()
	if y1 - y0 < 5 {
		return none
	}
	yy := mlx.arange(0.0, f64(h), 1.0, .float32).expand_dims(1).broadcast_to([h, w])
	mut best_score := 1e18
	mut best_base := LayerTemplate{}
	mut best_part := LayerTemplate{}
	mut found := false
	for split in y0 + 2 .. y1 - 1 {
		top := fgd.logical_and(yy.less_equal(mlx.f32_scalar(f32(split))))
		bottom := fgd.logical_and(yy.greater(mlx.f32_scalar(f32(split))))
		if cg_count(top) < cg_min_part_pixels || cg_count(bottom) < cg_min_part_pixels {
			continue
		}
		part := jlo_bbox_template(top, 0.0, none)
		base := jlo_bbox_template(bottom, 0.0, none)
		if part.cy >= base.cy {
			continue
		}
		dy := base.cy - part.cy
		rr := base.r + part.r
		if !(0.45 * rr <= dy && dy <= 1.35 * rr) {
			continue
		}
		if math.abs(base.cx - part.cx) > 1.2 * rr {
			continue
		}
		a0 := cg_area(base)
		a1 := cg_area(part)
		ratio := math.sqrt(a1 / fmax2(a0, 1e-8))
		mut ratio_penalty := 0.0
		if 0.25 - ratio > ratio_penalty {
			ratio_penalty = 0.25 - ratio
		}
		if ratio - 0.95 > ratio_penalty {
			ratio_penalty = ratio - 0.95
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
	return SplitResult{
		score: best_score
		base:  best_base
		part:  best_part
	}
}

// cg_window_centroid returns the local-window centroid (x, y).
fn cg_window_centroid(weights mlx.Array, cx f64, cy f64, r f64) ?[]f64 {
	h := weights.dim(0)
	w := weights.dim(1)
	pad := fmax2(2.0, 0.25 * r)
	mut x0 := int(math.floor(cx - r - pad))
	if x0 < 0 {
		x0 = 0
	}
	mut x1 := int(math.ceil(cx + r + pad + 1.0))
	if x1 > w {
		x1 = w
	}
	mut y0 := int(math.floor(cy - r - pad))
	if y0 < 0 {
		y0 = 0
	}
	mut y1 := int(math.ceil(cy + r + pad + 1.0))
	if y1 > h {
		y1 = h
	}
	win := weights.take_axis(mlx.arange(f64(y0), f64(y1), 1.0, .int32), 0).take_axis(mlx.arange(f64(x0),
		f64(x1), 1.0, .int32), 1)
	total := win.sum().item_f32()
	if total <= 1e-8 {
		return none
	}
	ys := mlx.arange(f64(y0), f64(y1), 1.0, .float32).expand_dims(1).broadcast_to([
		y1 - y0,
		x1 - x0,
	])
	xs := mlx.arange(f64(x0), f64(x1), 1.0, .float32).expand_dims(0).broadcast_to([
		y1 - y0,
		x1 - x0,
	])
	ux := win.multiply(xs).sum().item_f32() / total
	uy := win.multiply(ys).sum().item_f32() / total
	return [f64(ux), f64(uy)]
}

// cg_disk_evidence returns full-resolution disk-fit (scale_ratio, lateral_ratio).
fn cg_disk_evidence(fl mlx.Array, _ mlx.Array) ?[]f64 {
	wl := foreground_weights(fl)
	fg := wl.greater(mlx.f32_scalar(0.01))
	sp := cg_split_score(fg)
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
	return [r1 / fmax2(r0, 1e-8), (u1 - u0) / fmax2(r0 + r1, 1e-8)]
}

// cg_part_depth estimates one part's disparity depth.
fn cg_part_depth(wl mlx.Array, wr mlx.Array, t LayerTemplate, d_global f64) f64 {
	q := cg_down
	cx := t.cx * f64(q)
	cy := t.cy * f64(q)
	r := t.r * f64(q)
	left := cg_window_centroid(wl, cx, cy, r)
	right := cg_window_centroid(wr, cx - d_global, cy, r)
	if left == none || right == none {
		return cam_z - fx * stereo_base / fmax2(d_global, 1e-6)
	}
	l := left or { [0.0, 0.0] }
	ri := right or { [0.0, 0.0] }
	mut d := l[0] - ri[0]
	if d < 4.0 {
		d = 4.0
	}
	if d > 14.0 {
		d = 14.0
	}
	return cam_z - fx * stereo_base / d
}

// cg_estimate returns [u,v,z,area]×2 (base, attached part).
fn cg_estimate(fl mlx.Array, fr mlx.Array) []f64 {
	wl := foreground_weights(fl)
	wr := foreground_weights(fr)
	fg := wl.greater(mlx.f32_scalar(0.01))
	z_global, d_global, area_global := StereoDepth{}.estimate(fl, fr)
	sp := cg_split_score(fg)
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
	return [base.cx * q, base.cy * q, z0, cg_area(base), part.cx * q, part.cy * q, z1,
		cg_area(part)]
}
