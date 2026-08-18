module conger

// riesz_selftest_test.v — synthetic GT checks for the Riesz frontend
// (V port of the mlx-based synthetic part of src/riesz_selftest.py).
import math
import mlx

fn rz_make_grating(h int, w int, wavelength f64, angle_rad f64) mlx.Array {
	yy, xx := meshgrid_ij(h, w)
	xr :=
		xx.multiply(mlx.f32_scalar(f32(math.cos(angle_rad)))).add(yy.multiply(mlx.f32_scalar(f32(math.sin(angle_rad)))))
	s := xr.multiply(mlx.f32_scalar(f32(2.0 * math.pi / wavelength))).sin()
	return s.add(mlx.f32_scalar(1.0)).multiply(mlx.f32_scalar(0.5))
}

fn rz_make_step_edge(h int, w int) mlx.Array {
	left := mlx.zeros([h, w / 2], .float32)
	right := mlx.ones([h, w - w / 2], .float32)
	return mlx.concatenate([left, right], 1)
}

fn rz_synthesize_noise(size int) mlx.Array {
	noise := mlx.random_normal([size, size], .float32, 0.0, 1.0, mlx.random_key(0))
	clipped := noise.clip(mlx.f32_scalar(-2.0), mlx.f32_scalar(2.0))
	mn := clipped.min().item_f32()
	mx := clipped.max().item_f32()
	return clipped.subtract(mlx.f32_scalar(mn)).divide(mlx.f32_scalar(mx - mn + 1e-12))
}

fn test_riesz_update_consistency() {
	step := rz_make_step_edge(128, 128)
	mut rw := new_riesz_wavelet(step, 3.0, 0, 1.0)
	amp0 := rw.scales[0].amp
	rw.rz_update(step)
	diff := rw.scales[0].amp.subtract(amp0).abs().max().item_f32()
	assert diff < 1e-4
	img_diff := rw.img.subtract(step).abs().max().item_f32()
	assert img_diff < 1e-6
}

fn test_riesz_grating_orientation() {
	angle := math.radians(30.0)
	grating := rz_make_grating(128, 128, 16.0, angle)
	mut rw := new_riesz_wavelet(grating, 3.0, 0, 1.0)
	mut best := 0
	mut best_e := -1e18
	for i, s in rw.scales {
		e := f64(s.energy.mean().item_f32())
		if e > best_e {
			best_e = e
			best = i
		}
	}
	sc := rw.scales[best]
	re := f64(sc.ori.multiply(mlx.f32_scalar(2.0)).cos().mean().item_f32())
	im := f64(sc.ori.multiply(mlx.f32_scalar(2.0)).sin().mean().item_f32())
	theta := math.atan2(im, re) / 2.0
	mut t := theta
	if t < 0.0 {
		t += math.pi
	}
	if t >= math.pi {
		t -= math.pi
	}
	assert math.abs(t - angle) < 0.35, 'theta ${t} vs ${angle}'
}

fn test_riesz_feature_discrimination() {
	grating := rz_make_grating(128, 128, 16.0, math.radians(30.0))
	noise := rz_synthesize_noise(128)
	step := rz_make_step_edge(128, 128)
	mut rw_g := new_riesz_wavelet(grating, 3.0, 0, 1.0)
	f_g := rw_g.rz_features(false, 0)
	mut rw_n := new_riesz_wavelet(noise, 3.0, 0, 1.0)
	f_n := rw_n.rz_features(false, 0)
	mut rw_s := new_riesz_wavelet(step, 3.0, 0, 1.0)
	f_s := rw_s.rz_features(false, 0)
	g_res := f64(f_g.residual.mean().item_f32())
	n_res := f64(f_n.residual.mean().item_f32())
	s_res := f64(f_s.residual.mean().item_f32())
	// a pure grating has a strongly peaked spectrum → highest power-law fit
	// residual; broadband noise is closest to a power law → lowest residual.
	assert g_res > n_res
	assert g_res > s_res
}
