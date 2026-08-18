module conger

// joint_layer_optimizer.v — occluded two-layer template/disparity/assignment
// joint optimisation (V port of src/joint_layer_optimizer.py).

import math

import mlx

const jlo_down = 12
const jlo_scale_steps = [0.85, 1.0, 1.15]
const jlo_d_steps = [-0.5, 0.0, 0.5]

struct LayerTemplate {
	shape int // 0=circle, 2=square
	cx    f64
	cy    f64
	r     f64
	d     f64
}

fn (t LayerTemplate) mask(h int, w int) mlx.Array {
	return cc_template(t.shape, t.cx, t.cy, t.r, h, w)
}

// jlo_down_mask max-pools a bool mask by q×q blocks.
fn jlo_down_mask(mask mlx.Array) mlx.Array {
	q := jlo_down
	h := mask.dim(0)
	w := mask.dim(1)
	hh := h / q
	ww := w / q
	m := mask.take_axis(mlx.arange(0.0, f64(hh * q), 1.0, .int32), 0).take_axis(mlx.arange(0.0,
		f64(ww * q), 1.0, .int32), 1)
	return m.reshape([hh, q, ww, q]).max_axes([1, 3], false).greater(mlx.f32_scalar(0.0))
}

// jlo_down_mean averages a float array by q×q blocks.
fn jlo_down_mean(x mlx.Array) mlx.Array {
	q := jlo_down
	h := x.dim(0)
	w := x.dim(1)
	hh := h / q
	ww := w / q
	xc := x.take_axis(mlx.arange(0.0, f64(hh * q), 1.0, .int32), 0).take_axis(mlx.arange(0.0,
		f64(ww * q), 1.0, .int32), 1)
	return xc.reshape([hh, q, ww, q]).mean_axes([1, 3], false)
}

// jlo_bbox_template builds the initial bbox-derived template.
fn jlo_bbox_template(mask mlx.Array, d f64, occluder ?mlx.Array) LayerTemplate {
	h := mask.dim(0)
	w := mask.dim(1)
	sel := nonzero_indices(mask)
	idx := mlx.arange(0.0, f64(mask.size()), 1.0, .float32).take(sel)
	if idx.dim(0) == 0 {
		return LayerTemplate{
			shape: 0
			cx: f64(w - 1) / 2.0
			cy: f64(h - 1) / 2.0
			r: 4.0
			d: d
		}
	}
	xs := idx.remainder(mlx.f32_scalar(f32(w)))
	ys := idx.floor_divide(mlx.f32_scalar(f32(w)))
	x0 := f64(xs.min().item_f32())
	x1 := f64(xs.max().item_f32())
	y0 := f64(ys.min().item_f32())
	y1 := f64(ys.max().item_f32())
	r := fmax2(x1 - x0 + 1.0, y1 - y0 + 1.0) / 2.0
	cx := (x0 + x1) / 2.0
	cy := (y0 + y1) / 2.0
	mut best_shape := 0
	mut best_score := 1e18
	zlike := mlx.zeros([h, w], .bool_)
	for shape in [0, 2] {
		mut t := cc_template(shape, cx, cy, r, h, w)
		if oc := occluder {
			t = t.logical_and(oc.logical_not())
		}
		sc := cc_score(t, zlike, mask)
		if sc < best_score {
			best_score = sc
			best_shape = shape
		}
	}
	return LayerTemplate{
		shape: best_shape
		cx: cx
		cy: cy
		r: fmax2(r, 1.5)
		d: d
	}
}

// jlo_score returns the joint template score.
fn jlo_score(front LayerTemplate, back LayerTemplate, fg mlx.Array, disp mlx.Array, valid mlx.Array) f64 {
	if front.d <= back.d + 0.3 {
		return 1e18
	}
	h := fg.dim(0)
	w := fg.dim(1)
	tf := front.mask(h, w)
	tb := back.mask(h, w)
	vf := tf
	vb := tb.logical_and(tf.logical_not())
	pred := vf.logical_or(vb)
	inter := pred.logical_and(fg).astype(.float32).sum().item_f32()
	un := pred.logical_or(fg).astype(.float32).sum().item_f32()
	mask_cost := 1.0 - f64(inter) / fmax2(f64(un), 1.0)
	ratio := f64(vb.astype(.float32).sum().item_f32()) / fmax2(f64(tb.astype(.float32).sum().item_f32()),
		1.0)
	occlusion_penalty := fmax2(0.0, 0.25 - ratio) * 4.0
	mut d_cost := 0.0
	mf := valid.logical_and(vf)
	nf := f64(mf.astype(.float32).sum().item_f32())
	if nf > 4.0 {
		errf := mlx.where(mf, disp.subtract(mlx.f32_scalar(f32(front.d))).square(),
			mlx.f32_scalar(0.0))
		d_cost += f64(errf.sum().item_f32()) / nf / 4.0
	}
	mb := valid.logical_and(vb)
	nb := f64(mb.astype(.float32).sum().item_f32())
	if nb > 4.0 {
		errb := mlx.where(mb, disp.subtract(mlx.f32_scalar(f32(back.d))).square(),
			mlx.f32_scalar(0.0))
		d_cost += f64(errb.sum().item_f32()) / nb / 4.0
	}
	return mask_cost + occlusion_penalty + d_cost
}

// jlo_optimize_layer runs one layer's coordinate search.
fn jlo_optimize_layer(which int, front LayerTemplate, back LayerTemplate, fg mlx.Array, disp mlx.Array, valid mlx.Array) (LayerTemplate, LayerTemplate) {
	mut best_score := jlo_score(front, back, fg, disp, valid)
	mut cur := if which == 0 { front } else { back }
	other := if which == 0 { back } else { front }
	mut candidates := []LayerTemplate{}
	for sm in jlo_scale_steps {
		for dd in jlo_d_steps {
			candidates << LayerTemplate{
				shape: cur.shape
				cx: cur.cx
				cy: cur.cy
				r: fmax2(cur.r * sm, 1.5)
				d: cur.d + dd
			}
		}
	}
	for cand in candidates {
		mut sc := 0.0
		if which == 0 {
			sc = jlo_score(cand, other, fg, disp, valid)
		} else {
			sc = jlo_score(other, cand, fg, disp, valid)
		}
		if sc < best_score {
			best_score = sc
			cur = cand
		}
	}
	if which == 0 {
		return cur, other
	}
	return other, cur
}

// jlo_optimize returns (u0,v0,z0,area0,u1,v1,z1,area1).
fn jlo_optimize(fg mlx.Array, disp mlx.Array, valid mlx.Array, front0 mlx.Array, back0 mlx.Array, d_front f64, d_back f64) []f64 {
	fg_d := jlo_down_mask(fg)
	disp_d := jlo_down_mean(disp)
	valid_d := jlo_down_mask(valid)
	front_mask := jlo_down_mask(front0)
	back_mask := jlo_down_mask(back0)
	mut front := jlo_bbox_template(front_mask, d_front, none)
	mut back := jlo_bbox_template(back_mask, d_back, front_mask)
	for _ in 0 .. 1 {
		front, back = jlo_optimize_layer(0, front, back, fg_d, disp_d, valid_d)
		front, back = jlo_optimize_layer(1, front, back, fg_d, disp_d, valid_d)
	}
	q := f64(jlo_down)
	mut out := []f64{}
	for tmpl in [front, back] {
		area := (if tmpl.shape == 2 { 4.0 } else { math.pi }) * (tmpl.r * q) * (tmpl.r * q)
		z := cam_z - fx * stereo_base / fmax2(tmpl.d, 1e-6)
		out << tmpl.cx * q
		out << tmpl.cy * q
		out << z
		out << area
	}
	return out
}
