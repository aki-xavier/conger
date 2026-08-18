module conger

// scm_proxy_test.v — appearance-mechanism proxy black-box tests.
import math
import mlx

fn synthetic_rgb(n_hue int, n_lcol int, n_ldir int, noise f32, seed u64) (mlx.Array, mlx.Array) {
	rng := mlx.random_key(seed)
	a :=
		mlx.random_uniform(mlx.f32_scalar(0.0), mlx.f32_scalar(1.0), [n_hue, 3], .float32, rng).add(mlx.f32_scalar(0.2))
	k1, k2 := mlx.random_split(rng)
	g := mlx.random_uniform(mlx.f32_scalar(0.0), mlx.f32_scalar(1.0), [n_lcol, n_ldir, 3],
		.float32, k1).add(mlx.f32_scalar(0.2))
	mut rgb := a.expand_dims(1).expand_dims(1).multiply(g.expand_dims(0))
	if noise > 0.0 {
		rgb = rgb.add(mlx.random_normal(rgb.shape(), .float32, 0.0, noise, k2))
	}
	return rgb, a
}

fn test_fit_recovers_albedo_up_to_per_channel_scale() {
	rgb, a_true := synthetic_rgb(6, 3, 3, 0.0, 0)
	mut m := AppearanceMechanism{}
	m.fit(rgb)
	albedo := m.albedo or { panic('') }
	ratio := albedo.divide(a_true)
	spread := ratio.max_axis(0, false).subtract(ratio.min_axis(0, false)).max().item_f32()
	assert spread < 1e-4
}

fn test_perfect_modularity_has_invariance_near_one() {
	rgb, _ := synthetic_rgb(6, 3, 3, 0.0, 0)
	mut m := AppearanceMechanism{}
	m.fit(rgb)
	assert math.abs(m.albedo_invariance(rgb) - 1.0) < 1e-4
}

fn test_noise_reduces_invariance_but_not_catastrophically() {
	rgb, _ := synthetic_rgb(6, 3, 3, 0.02, 0)
	mut m := AppearanceMechanism{}
	m.fit(rgb)
	score := m.albedo_invariance(rgb)
	assert 0.9 < score && score < 1.0
}

fn test_do_lighting_counterfactual() {
	rgb, _ := synthetic_rgb(6, 3, 3, 0.0, 0)
	mut m := AppearanceMechanism{}
	m.fit(rgb)
	got := m.do_lighting(2, 0, 0, 1, 2)
	expect := m.predict(2, 1, 2)
	assert got.subtract(expect).abs().max().item_f32() < 1e-5
	target :=
		rgb.take_axis(sel1(2), 0).squeeze_axis(0).take_axis(sel1(1), 0).squeeze_axis(0).take_axis(sel1(2), 0).squeeze_axis(0)
	assert got.subtract(target).abs().max().item_f32() < 1e-5
}

fn test_foreground_mean_rgb() {
	frame := mlx.array_f32([f32(10.0), 20.0, 30.0, 30.0, 60.0, 90.0], [1, 2, 3])
	weights := mlx.array_f32([f32(1.0), 3.0], [1, 2])
	mean := foreground_mean_rgb(frame, weights)
	vals := mean.data_f32()
	assert math.abs(f64(vals[0]) - 25.0) < 1e-4
	assert math.abs(f64(vals[1]) - 50.0) < 1e-4
	assert math.abs(f64(vals[2]) - 75.0) < 1e-4
}
