module conger

// mrf_learning_test.v — closed-form M-step checks, E-step shape/exact-value
// checks, and parameter recovery on a Gibbs-sampled GMRF field.
import math

fn approx(a f64, b f64, tol f64) bool {
	return math.abs(a - b) < tol
}

// gibbs_field samples a rows×cols GMRF(μ, β, σ²) by Gibbs sweeps from a
// random init (deterministic via `seed`).
fn gibbs_field(rows int, cols int, mu f64, beta f64, sig2 f64, sweeps int, seed u64) []f64 {
	mut rng := new_rng(seed)
	n := rows * cols
	mut x := []f64{cap: n}
	for _ in 0 .. n {
		x << rng.normal(mu, 1.0)
	}
	sd := math.sqrt(sig2)
	for _ in 0 .. sweeps {
		for i in 0 .. n {
			mut c := mu
			for j in grid4_nb_idx(i, rows, cols) {
				c += beta * (x[j] - mu)
			}
			x[i] = c + rng.normal(0.0, sd)
		}
	}
	return x
}

fn test_gmrf_m_step_closed_form() {
	// two sites, m = [3, 1] → μ = 2, v = [-1, +1]
	// OLS β = -2/2 = -1, clamped to bmax = 0.95·(1+1/1)/4 = 0.475
	l := new_gmrf_learner(1, 2, 0.0)
	newp := l.maximize(GMRFMeans{
		means: [3.0, 1.0]
	}, [3.0, 1.0], [0.0, 0.0, 0.0], 0.0)
	assert approx(newp[0], 2.0, 1e-12)
	assert approx(newp[1], -0.475, 1e-12)
	// residuals with β=-0.475: d = [±0.525] → R = 0.525²; with the mean-field
	// variance correction (τ²=1): s = (R + √(R²+4R))/2
	r1 := 0.525 * 0.525
	want_s := (r1 + math.sqrt(r1 * r1 + 4.0 * r1)) / 2.0
	assert approx(math.exp(newp[2]), want_s, 1e-12)
	// damping blends toward the old params
	pd := l.maximize(GMRFMeans{
		means: [3.0, 1.0]
	}, [3.0, 1.0], [10.0, 0.0, 0.0], 0.5)
	assert approx(pd[0], 6.0, 1e-12)
	assert approx(math.exp(pd[2]), math.sqrt(want_s), 1e-12)
}

fn test_gmrf_e_step_no_coupling_exact() {
	// β=0: each site's posterior mean is the independent fusion
	// m = (y/τ² + μ/σ²)/(1/τ² + 1/σ²), here σ²=τ²=1, μ=0 → m = y/2
	l := new_gmrf_learner(2, 2, 0.0)
	y := [1.0, -2.0, 0.5, 3.0]
	resp := l.responsibilities([0.0, 0.0, 0.0], y, 1.0)
	assert resp.means.len == 4
	for i, v in resp.means {
		assert approx(v, y[i] / 2.0, 1e-6)
	}
}

fn test_gmrf_learning_recovers_params() {
	// 16×16 field from GMRF(μ=0.5, β=0.2, σ²=0.5), observed almost cleanly
	// (τ²=0.01); learn from a deliberately bad init.
	rows, cols := 16, 16
	field := gibbs_field(rows, cols, 0.5, 0.2, 0.5, 400, 11)
	l := new_gmrf_learner(rows, cols, math.log(0.01))
	mut loop := EMLoop[GMRFLearner, []f64, GMRFMeans]{
		max_iters: 40
		tol:       1e-4
		damping:   0.2
		model:     l
	}
	init := [0.0, 0.02, math.log(2.0)]
	ll0 := l.log_likelihood(init, field)
	res := loop.run(field, init)
	assert res.log_likelihood > ll0
	assert math.abs(res.params[0] - 0.5) < 0.2
	assert res.params[1] > 0.08 && res.params[1] < 0.32
	s2 := math.exp(res.params[2])
	assert s2 > 0.15 && s2 < 1.5
	println('clean-field: mu=${res.params[0]:.3f} beta=${res.params[1]:.3f} sig2=${s2:.3f} iters=${res.iterations}')
}

fn test_gmrf_learning_with_observation_noise() {
	// same field, noisy observation y = x + N(0, 0.25) with τ² held fixed;
	// the learner must still move β from a near-zero init toward the truth.
	rows, cols := 16, 16
	field := gibbs_field(rows, cols, 0.5, 0.2, 0.5, 400, 11)
	mut rng := new_rng(23)
	tau2 := 0.25
	mut y := []f64{cap: field.len}
	for v in field {
		y << v + rng.normal(0.0, math.sqrt(tau2))
	}
	l := new_gmrf_learner(rows, cols, math.log(tau2))
	mut loop := EMLoop[GMRFLearner, []f64, GMRFMeans]{
		max_iters: 40
		tol:       1e-4
		damping:   0.2
		model:     l
	}
	init := [0.0, 0.02, math.log(2.0)]
	ll0 := l.log_likelihood(init, y)
	res := loop.run(y, init)
	assert res.log_likelihood > ll0
	assert res.params[1] > init[1]
	assert res.params[1] > 0.05 && res.params[1] < 0.35
	println('noisy-field: mu=${res.params[0]:.3f} beta=${res.params[1]:.3f} sig2=${math.exp(res.params[2]):.3f} iters=${res.iterations}')
}
