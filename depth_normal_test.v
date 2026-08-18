module conger

// depth_normal_test.v — DepthNormalModel (depth↔normal) black-box test.
import math

fn test_depth_normal_denoises_depth() {
	n := 64
	x := linspace(0.0, 1.0, n)
	mut z_true := []f64{len: n}
	for i in 0 .. n {
		z_true[i] = math.sin(3.0 * x[i]) + 0.5 * x[i]
	}
	mut s_true := diff(z_true)
	for i in 0 .. s_true.len {
		s_true[i] *= f64(n - 1)
	}
	mut rng := new_rng(0)
	mut z_obs := []f64{len: n}
	mut s_obs := []f64{len: n - 1}
	for i in 0 .. n {
		z_obs[i] = z_true[i] + rng.normal(0.0, 0.3)
	}
	for i in 0 .. n - 1 {
		s_obs[i] = s_true[i] + rng.normal(0.0, 0.02)
	}

	model := new_depth_normal_model(z_obs, s_obs, 0.5, 0.02)
	mut loop := EMLoop[DepthNormalModel, []f64, []f64]{
		model:     model
		max_iters: 30
		tol:       1e-8
	}
	result := loop.run([]f64{}, z_obs)

	mut got_rmse := 0.0
	mut obs_rmse := 0.0
	for i in 0 .. n {
		got_rmse += (result.params[i] - z_true[i]) * (result.params[i] - z_true[i])
		obs_rmse += (z_obs[i] - z_true[i]) * (z_obs[i] - z_true[i])
	}
	assert math.sqrt(got_rmse / f64(n)) < math.sqrt(obs_rmse / f64(n))
}
