module conger

// stereo_layers.v — occlusion-aware per-layer binocular geometry
// (V port of src/stereo_layers.py).
import mlx

const sl_d_range_lo = 5
const sl_d_range_hi = 12
const sl_patch_r = 4
const sl_min_conf = 1.08

// SLCluster is the 2-means (x,y,disparity) split result.
struct SLCluster {
	front   mlx.Array
	front_c [3]f64 // (cx, cy, d) of the near/front layer
	back_c  [3]f64 // (cx, cy, d) of the back layer
}

// sl_centroid returns (u, v, effective area) of a weight map.
fn sl_centroid(weights mlx.Array) (f64, f64, f64) {
	h := weights.dim(0)
	w := weights.dim(1)
	ys := mlx.arange(0.0, f64(h), 1.0, .float32).expand_dims(1)
	xs := mlx.arange(0.0, f64(w), 1.0, .float32).expand_dims(0)
	total := weights.sum().item_f32()
	if total <= 1e-8 {
		c := f64(w - 1) / 2.0
		return c, c, 0.0
	}
	u := weights.multiply(xs).sum().item_f32() / total
	v := weights.multiply(ys).sum().item_f32() / total
	return f64(u), f64(v), f64(total)
}

// sl_features concatenates rgb with chroma + luminance gradients → (h,w,7).
fn sl_features(rgb mlx.Array, frame mlx.Array) mlx.Array {
	lum := frame_lum(frame)
	re, im := frame_chroma(frame)
	gx := roll_axis(lum, -1, 1).subtract(lum)
	gy := roll_axis(lum, -1, 0).subtract(lum)
	return mlx.concatenate([
		rgb,
		re.expand_dims(-1),
		im.expand_dims(-1),
		gx.multiply(mlx.f32_scalar(5.0)).expand_dims(-1),
		gy.multiply(mlx.f32_scalar(5.0)).expand_dims(-1),
	], 2)
}

// sl_disparity_map returns (disp, conf, valid) per-pixel block matching.
fn sl_disparity_map(fl mlx.Array, fr mlx.Array) (mlx.Array, mlx.Array, mlx.Array) {
	rgb_l :=
		fl.take_axis(mlx.arange(0.0, 3.0, 1.0, .int32), -1).astype(.float32).divide(mlx.f32_scalar(255.0))
	rgb_r :=
		fr.take_axis(mlx.arange(0.0, 3.0, 1.0, .int32), -1).astype(.float32).divide(mlx.f32_scalar(255.0))
	fl_f := sl_features(rgb_l, fl)
	fr_f := sl_features(rgb_r, fr)
	h := fl_f.dim(0)
	w := fl_f.dim(1)
	d0 := sl_d_range_lo
	d1 := sl_d_range_hi
	r := sl_patch_r
	x0 := d1 + r
	x1 := w - r
	ih := h - 2 * r
	iw := x1 - x0
	mut costs := []mlx.Array{cap: d1 - d0 + 1}
	for d in d0 .. d1 + 1 {
		mut c := mlx.zeros([ih, iw], .float32)
		for dy in 0 .. 2 * r + 1 {
			for dx in 0 .. 2 * r + 1 {
				lp := fl_f.take_axis(mlx.arange(f64(dy), f64(ih + dy), 1.0, .int32), 0).take_axis(mlx.arange(f64(
					x0 - r + dx), f64(x1 - r + dx), 1.0, .int32), 1)
				rp := fr_f.take_axis(mlx.arange(f64(dy), f64(ih + dy), 1.0, .int32), 0).take_axis(mlx.arange(f64(
					x0 - r - d + dx), f64(x1 - r - d + dx), 1.0, .int32), 1)
				c = c.add(lp.subtract(rp).square().sum_axis(2, false))
			}
		}
		costs << c
	}
	cost := mlx.stack(costs, 2) // (ih, iw, D)
	order := cost.sort_axis(2)
	best := order.take_axis(sel1(0), 2).squeeze_axis(2)
	second := order.take_axis(sel1(1), 2).squeeze_axis(2)
	conf := second.divide(best.maximum(mlx.f32_scalar(1e-6)))
	disp := cost.argmin_axis(2, false).astype(.float32).add(mlx.f32_scalar(f32(d0)))
	full_d := overwrite_region(mlx.zeros([h, w], .float32), disp, r, d1 + r)
	full_conf := overwrite_region(mlx.zeros([h, w], .float32), conf, r, d1 + r)
	valid_conf := conf.greater(mlx.f32_scalar(f32(sl_min_conf)))
	valid := overwrite_region(mlx.zeros([h, w], .bool_), valid_conf, r, d1 + r)
	return full_d, full_conf, valid
}

// sl_cluster_layers splits (x,y,disparity) via weighted 2-means.
fn sl_cluster_layers(disp mlx.Array, fw mlx.Array, valid mlx.Array) ?SLCluster {
	h := disp.dim(0)
	w := disp.dim(1)
	idx := nonzero_indices(valid.reshape([h * w]))
	if idx.dim(0) < 32 {
		return none
	}
	xs := idx.remainder(mlx.int_scalar(w)).astype(.float32)
	ys := idx.floor_divide(mlx.int_scalar(w)).astype(.float32)
	flat_d := disp.reshape([h * w])
	flat_w := fw.reshape([h * w])
	ds := flat_d.take(idx)
	ws := flat_w.take(idx)
	ds_sorted := ds.sort()
	n := ds_sorted.dim(0)
	d_lo := f64(ds_sorted.take(sel1(n / 4)).item_f32())
	d_hi := f64(ds_sorted.take(sel1((3 * n) / 4)).item_f32())
	cx := f64(xs.multiply(ws).sum().item_f32() / ws.sum().item_f32())
	cy := f64(ys.multiply(ws).sum().item_f32() / ws.sum().item_f32())
	mut c_lo := [cx, cy, d_lo]!
	mut c_hi := [cx, cy, d_hi]!
	for _ in 0 .. 16 {
		dl :=
			xs.subtract(mlx.f32_scalar(f32(c_lo[0]))).divide(mlx.f32_scalar(40.0)).square().add(ys.subtract(mlx.f32_scalar(f32(c_lo[1]))).divide(mlx.f32_scalar(40.0)).square()).add(ds.subtract(mlx.f32_scalar(f32(c_lo[2]))).divide(mlx.f32_scalar(4.0)).square())
		dh :=
			xs.subtract(mlx.f32_scalar(f32(c_hi[0]))).divide(mlx.f32_scalar(40.0)).square().add(ys.subtract(mlx.f32_scalar(f32(c_hi[1]))).divide(mlx.f32_scalar(40.0)).square()).add(ds.subtract(mlx.f32_scalar(f32(c_hi[2]))).divide(mlx.f32_scalar(4.0)).square())
		near_hi := dh.less(dl)
		// assign to c_lo (far layer) with mask ~near_hi
		{
			wm := mlx.where(near_hi.logical_not(), ws, mlx.f32_scalar(0.0))
			tot := wm.sum().item_f32()
			if tot > 1e-8 {
				c_lo[0] = f64(xs.multiply(wm).sum().item_f32() / tot)
				c_lo[1] = f64(ys.multiply(wm).sum().item_f32() / tot)
				c_lo[2] = f64(ds.multiply(wm).sum().item_f32() / tot)
			}
		}
		// assign to c_hi (near layer) with mask near_hi
		{
			wm := mlx.where(near_hi, ws, mlx.f32_scalar(0.0))
			tot := wm.sum().item_f32()
			if tot > 1e-8 {
				c_hi[0] = f64(xs.multiply(wm).sum().item_f32() / tot)
				c_hi[1] = f64(ys.multiply(wm).sum().item_f32() / tot)
				c_hi[2] = f64(ds.multiply(wm).sum().item_f32() / tot)
			}
		}
	}
	if c_hi[2] - c_lo[2] < 0.5 {
		return none
	}
	yy, xx := meshgrid_ij(h, w)
	dl :=
		xx.subtract(mlx.f32_scalar(f32(c_lo[0]))).divide(mlx.f32_scalar(40.0)).square().add(yy.subtract(mlx.f32_scalar(f32(c_lo[1]))).divide(mlx.f32_scalar(40.0)).square()).add(disp.subtract(mlx.f32_scalar(f32(c_lo[2]))).divide(mlx.f32_scalar(4.0)).square())
	dh :=
		xx.subtract(mlx.f32_scalar(f32(c_hi[0]))).divide(mlx.f32_scalar(40.0)).square().add(yy.subtract(mlx.f32_scalar(f32(c_hi[1]))).divide(mlx.f32_scalar(40.0)).square()).add(disp.subtract(mlx.f32_scalar(f32(c_hi[2]))).divide(mlx.f32_scalar(4.0)).square())
	front := valid.logical_and(dh.less(dl))
	return SLCluster{
		front:   front
		front_c: c_hi
		back_c:  c_lo
	}
}

// sl_fallback returns the degenerate two-layer placeholder when separation fails.
fn sl_fallback(fl mlx.Array, fr mlx.Array) []f64 {
	fw := foreground_weights(fl)
	u, v, area := sl_centroid(fw)
	z, _, _ := StereoDepth{}.estimate(fl, fr)
	return [u, v, z, area, u, v, z, area]
}

// sl_estimate returns (u0,v0,z0,area0,u1,v1,z1,area1); index 0 = front/near layer.
fn sl_estimate(fl mlx.Array, fr mlx.Array) []f64 {
	fw := foreground_weights(fl)
	fg := fw.greater(mlx.f32_scalar(0.01))
	disp, _, valid0 := sl_disparity_map(fl, fr)
	valid := valid0.logical_and(fg)
	clustered := sl_cluster_layers(disp, fw, valid)
	if c := clustered {
		front := c.front
		d_front := c.front_c[2]
		d_back := c.back_c[2]
		back := valid.logical_and(front.logical_not())
		_, _, area0 := sl_centroid(fw.multiply(front.astype(.float32)))
		mut u1, mut v1, mut area1 := sl_centroid(fw.multiply(back.astype(.float32)))
		cu, cv, c_area, _, c_score := cc_complete(front, back)
		if c_area > 0.0 {
			mut w := (0.30 - c_score) / 0.25
			if w < 0.0 {
				w = 0.0
			}
			if w > 1.0 {
				w = 1.0
			}
			ca := c_area * 2.0 / 3.0
			u1 = (1.0 - w) * u1 + w * cu
			v1 = (1.0 - w) * v1 + w * cv
			area1 = (1.0 - w) * area1 + w * ca
		}
		// jlo_optimize always returns a joint template (the Python `| None`
		// branch is dead code), so the joint centre/depth always wins.
		joint := jlo_optimize(fg, disp, valid, front, back, d_front, d_back)
		return [joint[0], joint[1], joint[2], area0, joint[4], joint[5], joint[6], area1]
	}
	return sl_fallback(fl, fr)
}

// sl_scaled maps (N,8) raw layer stats to a scaled feature concatenation (N,8).
fn sl_scaled(stats mlx.Array) mlx.Array {
	mut cols := []mlx.Array{cap: 8}
	for off in [0, 4] {
		cols << col(stats, off).divide(mlx.f32_scalar(f32(img_w)))
		cols << col(stats, off + 1).divide(mlx.f32_scalar(f32(img_h)))
		cols << col(stats, off + 2)
		cols << col(stats, off + 3).divide(mlx.f32_scalar(1000.0))
	}
	return mlx.stack(cols, 1)
}
