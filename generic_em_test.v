module conger

// generic_em_test.v — minimal end-to-end exercise of the generic EMLoop.
import math

// --- minimal 2-component 1-D Gaussian mixture -------------------------------
// Fixed variance σ²=1 and fixed mixing weight π=0.5; only the two means are
// estimated. This is a textbook EM problem that must converge from an
// asymmetric init, exercising responsibilities → maximize → log_likelihood.

struct ToyGMM {}

struct GMMResp {
	r [][]f64 // (N, 2) responsibilities
}

fn gaussian_pdf(x f64, mu f64) f64 {
	d := x - mu
	return math.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
}

fn (g ToyGMM) responsibilities(params []f64, observation []f64, temperature f64) GMMResp {
	mu1 := params[0]
	mu2 := params[1]
	inv_t := 1.0 / temperature
	mut r := [][]f64{len: observation.len, init: []f64{len: 2}}
	for i, x in observation {
		w1 := math.pow(gaussian_pdf(x, mu1), inv_t)
		w2 := math.pow(gaussian_pdf(x, mu2), inv_t)
		s := w1 + w2
		r[i][0] = w1 / s
		r[i][1] = w2 / s
	}
	return GMMResp{
		r: r
	}
}

fn (g ToyGMM) maximize(resp GMMResp, observation []f64, params []f64, damping f64) []f64 {
	mut w1 := 0.0
	mut w2 := 0.0
	mut wx1 := 0.0
	mut wx2 := 0.0
	for i, x in observation {
		w1 += resp.r[i][0]
		w2 += resp.r[i][1]
		wx1 += resp.r[i][0] * x
		wx2 += resp.r[i][1] * x
	}
	mu1_new := wx1 / w1
	mu2_new := wx2 / w2
	return [
		(1.0 - damping) * mu1_new + damping * params[0],
		(1.0 - damping) * mu2_new + damping * params[1],
	]
}

fn (g ToyGMM) log_likelihood(params []f64, observation []f64) f64 {
	mu1 := params[0]
	mu2 := params[1]
	mut ll := 0.0
	for x in observation {
		ll += math.log(0.5 * gaussian_pdf(x, mu1) + 0.5 * gaussian_pdf(x, mu2))
	}
	return ll
}

fn test_emloop_converges_on_toy_gmm() {
	mut data := []f64{}
	mut rng := new_rng(42)
	for _ in 0 .. 400 {
		data << rng.normal(-2.0, 1.0)
	}
	for _ in 0 .. 400 {
		data << rng.normal(2.0, 1.0)
	}
	mut loop := EMLoop[ToyGMM, []f64, GMMResp]{
		max_iters: 100
		tol:       1e-6
		model:     ToyGMM{}
	}
	res := loop.run(data, [-1.0, 1.0])
	mut means := [res.params[0], res.params[1]]
	means.sort()
	assert math.abs(means[0] - (-2.0)) < 0.2
	assert math.abs(means[1] - 2.0) < 0.2
	assert res.iterations > 1
	assert res.iterations <= 100
	// standard EM (no damping) must be monotonically non-decreasing
	for i in 1 .. res.trajectory.len {
		assert res.trajectory[i] >= res.trajectory[i - 1] - 1e-9
	}
}
