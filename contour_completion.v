module conger

// contour_completion.v — visible back-layer contour → full shape template
// (V port of src/contour_completion.py).

import math

import mlx

const cc_down = 3
const cc_center_steps = [-6.0, -3.0, 0.0, 3.0, 6.0]
const cc_scale_steps = [0.85, 1.0, 1.2, 1.45, 1.75]

fn meshgrid_ij(h int, w int) (mlx.Array, mlx.Array) {
	yy := mlx.arange(0.0, f64(h), 1.0, .float32).expand_dims(1).broadcast_to([h, w])
	xx := mlx.arange(0.0, f64(w), 1.0, .float32).expand_dims(0).broadcast_to([h, w])
	return yy, xx
}

// cc_down downsamples a bool mask by max-pooling q×q blocks.
fn cc_down(mask mlx.Array) mlx.Array {
	q := cc_down
	h := mask.dim(0)
	w := mask.dim(1)
	hh := h / q
	ww := w / q
	m := mask.take_axis(mlx.arange(0.0, f64(hh * q), 1.0, .int32), 0).take_axis(mlx.arange(0.0,
		f64(ww * q), 1.0, .int32), 1)
	return m.reshape([hh, q, ww, q]).max_axes([1, 3], false).greater(mlx.f32_scalar(0.0))
}

// cc_centroid returns the mask centroid (x, y).
fn cc_centroid(mask mlx.Array) (f64, f64) {
	idx := mlx.arange(0.0, f64(mask.size()), 1.0, .float32)
	sel := nonzero_indices(mask)
	n := sel.dim(0)
	if n == 0 {
		return f64(mask.dim(1) - 1) / 2.0, f64(mask.dim(0) - 1) / 2.0
	}
	ids := idx.take(sel)
	w := mask.dim(1)
	xs := ids.remainder(mlx.f32_scalar(f32(w))).mean().item_f32()
	ys := ids.floor_divide(mlx.f32_scalar(f32(w))).mean().item_f32()
	return f64(xs), f64(ys)
}

// cc_template builds a circle (shape!=2) or square (shape==2) template mask.
fn cc_template(shape int, cx f64, cy f64, r f64, h int, w int) mlx.Array {
	yy, xx := meshgrid_ij(h, w)
	if shape == 2 {
		return xx.subtract(mlx.f32_scalar(f32(cx))).abs().less_equal(mlx.f32_scalar(f32(r))).logical_and(yy.subtract(mlx.f32_scalar(f32(cy))).abs().less_equal(mlx.f32_scalar(f32(r))))
	}
	dx := xx.subtract(mlx.f32_scalar(f32(cx)))
	dy := yy.subtract(mlx.f32_scalar(f32(cy)))
	return dx.multiply(dx).add(dy.multiply(dy)).less_equal(mlx.f32_scalar(f32(r * r)))
}

// cc_score returns 1 − IoU of (template\front) vs back.
fn cc_score(template mlx.Array, front mlx.Array, back mlx.Array) f64 {
	visible := template.logical_and(front.logical_not())
	inter := visible.logical_and(back).astype(.float32).sum().item_f32()
	un := visible.logical_or(back).astype(.float32).sum().item_f32()
	return 1.0 - f64(inter) / fmax2(f64(un), 1.0)
}

// cc_fit_shape runs coordinate descent for one shape → (cx, cy, r, score).
fn cc_fit_shape(shape int, front mlx.Array, back mlx.Array) (f64, f64, f64, f64) {
	h := back.dim(0)
	w := back.dim(1)
	sel := nonzero_indices(back)
	ids := mlx.arange(0.0, f64(back.size()), 1.0, .float32).take(sel)
	xs := ids.remainder(mlx.f32_scalar(f32(w)))
	ys := ids.floor_divide(mlx.f32_scalar(f32(w)))
	x0 := f64(xs.min().item_f32())
	x1 := f64(xs.max().item_f32())
	y0 := f64(ys.min().item_f32())
	y1 := f64(ys.max().item_f32())
	mut bcx := (x0 + x1) / 2.0
	mut bcy := (y0 + y1) / 2.0
	mut br := fmax2(x1 - x0 + 1.0, y1 - y0 + 1.0) / 2.0
	mut best_score := 1e18
	for _ in 0 .. 3 {
		for dc in cc_center_steps {
			for dcy in cc_center_steps {
				for mul in cc_scale_steps {
					r1 := fmax2(br * mul, 1.5)
					t := cc_template(shape, bcx + dc, bcy + dcy, r1, h, w)
					sc := cc_score(t, front, back)
					if sc < best_score {
						best_score = sc
						bcx = bcx + dc
						bcy = bcy + dcy
						br = r1
					}
				}
			}
		}
	}
	return bcx, bcy, br, best_score
}

// cc_complete returns (u, v, area, kind, score) from front/back visible masks.
fn cc_complete(front_mask mlx.Array, back_mask mlx.Array) (f64, f64, f64, int, f64) {
	front := cc_down(front_mask)
	back := cc_down(back_mask)
	h := back.dim(0)
	w := back.dim(1)
	if back.astype(.int32).sum().item_i32() < 8 {
		cx, cy := cc_centroid(back)
		return cx * cc_down, cy * cc_down, 0.0, 0, 1.0
	}
	c0, cy0, r0, s0 := cc_fit_shape(0, front, back)
	c2, cy2, r2, s2 := cc_fit_shape(2, front, back)
	mut shape := 0
	mut cx := c0
	mut cy := cy0
	mut r := r0
	mut score := s0
	if s2 < s0 {
		shape = 2
		cx = c2
		cy = cy2
		r = r2
		score = s2
	}
	area := (if shape == 2 { 4.0 } else { math.pi }) * (r * f64(cc_down)) * (r * f64(cc_down))
	return cx * cc_down, cy * cc_down, area, shape, score
}
