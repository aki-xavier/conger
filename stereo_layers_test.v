module conger

// stereo_layers_test.v — V port of tests/test_stereo_layers.py
// (occlusion-aware two-layer disparity → front/back geometry).
import math
import mlx

// sl_layered_frames builds a two-texture-layer stereo pair: front layer d=12,
// back layer d=6, with the front layer occluding the back in the left frame.
fn sl_layered_frames() (mlx.Array, mlx.Array) {
	h := img_h
	w := img_w
	keys := split_keys(9, 2)
	back := mlx.random_uniform(mlx.f32_scalar(0.0), mlx.f32_scalar(1.0), [40, 40, 3], .float32,
		keys[0]).multiply(mlx.f32_scalar(255.0)).astype(.uint8)
	front := mlx.random_uniform(mlx.f32_scalar(0.0), mlx.f32_scalar(1.0), [40, 40, 3], .float32,
		keys[1]).multiply(mlx.f32_scalar(255.0)).astype(.uint8)
	mut fl_rgb := mlx.full([h, w, 3], mlx.int_scalar(20), .uint8)
	mut fr_rgb := mlx.full([h, w, 3], mlx.int_scalar(20), .uint8)
	// left: back painted first, then front occluding the overlap
	fl_rgb = overwrite_region(fl_rgb, back, 45, 55)
	fl_rgb = overwrite_region(fl_rgb, front, 60, 70)
	// right: same layers shifted left by their disparity (back d=6, front d=12)
	fr_rgb = overwrite_region(fr_rgb, back, 45, 49)
	fr_rgb = overwrite_region(fr_rgb, front, 60, 58)
	a := mlx.full([h, w, 1], mlx.int_scalar(255), .uint8)
	return mlx.concatenate([fl_rgb, a], 2), mlx.concatenate([fr_rgb, a], 2)
}

fn test_layered_disparity() {
	fl, fr := sl_layered_frames()
	out := sl_estimate(fl, fr)
	u0 := out[0]
	v0 := out[1]
	z0 := out[2]
	a0 := out[3]
	u1 := out[4]
	v1 := out[5]
	z1 := out[6]
	a1 := out[7]
	// d=12 → z=4.0; d=6 → z=2.5
	assert math.abs(z0 - 4.0) < 0.25, 'front z ${z0}'
	assert math.abs(z1 - 2.5) < 0.35, 'back z ${z1}'
	assert math.abs(u0 - 89.5) < 9.0 && math.abs(v0 - 79.5) < 9.0
	// completion is confidence-gated: the recovered back centre must stay
	// within the true contour span either way
	assert u1 >= 65.0 && u1 <= 78.0 && v1 >= 55.0 && v1 <= 70.0
	assert a0 > 1000.0 && a1 > 500.0
}
