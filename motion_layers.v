module conger

// motion_layers.v — motion segmentation↔optical flow EM (1D optical flow → K
// motion layers).
//
// Observation u(x) = per-pixel velocity, composed of K layers each with a
// constant velocity v_k. Latent = per-pixel layer assignment; θ = (v_1,…,v_K).
// E step computes soft assignments + spatial smoothing; M step re-estimates
// each layer velocity by weighted averaging.
import math

struct MotionLayersModel {
	k      int
	n      int
	sigma  f64
	smooth int
	x      []f64
}

fn new_motion_layers_model(k int, n int, sigma f64, smooth int) MotionLayersModel {
	mut x := arange_f64(n)
	for i in 0 .. n {
		x[i] = x[i] / f64(max_int(n - 1, 1))
	}
	return MotionLayersModel{
		k:      k
		n:      n
		sigma:  sigma
		smooth: smooth
		x:      x
	}
}

// sample draws a piecewise-constant optical flow field (uniform blocks) + noise.
fn (m MotionLayersModel) sample(params []f64, seed u64) []f64 {
	mut rng := new_rng(seed)
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		seg := i * m.k / m.n
		out[i] = params[seg] + rng.normal(0.0, m.sigma)
	}
	return out
}

// responsibilities returns soft layer assignments (n, k), spatially smoothed.
fn (m MotionLayersModel) responsibilities(params []f64, observation []f64, temperature f64) [][]f64 {
	inv := 1.0 / math.max(temperature, 1e-8)
	mut logq := [][]f64{len: m.n, init: []f64{len: m.k}}
	for i in 0 .. m.n {
		mut rowmax := -1e300
		for j in 0 .. m.k {
			d := observation[i] - params[j]
			logq[i][j] = -d * d / (2.0 * m.sigma * m.sigma) * inv
			if logq[i][j] > rowmax {
				rowmax = logq[i][j]
			}
		}
		mut s := 0.0
		for j in 0 .. m.k {
			logq[i][j] -= rowmax
			q := math.exp(logq[i][j])
			logq[i][j] = q
			s += q
		}
		for j in 0 .. m.k {
			logq[i][j] /= s
		}
	}
	// spatial smoothing (roll along axis 0)
	mut q := logq.clone()
	for _ in 0 .. m.smooth {
		mut nq := [][]f64{len: m.n, init: []f64{len: m.k}}
		for i in 0 .. m.n {
			ip := (i - 1 + m.n) % m.n
			inext := (i + 1) % m.n
			for j in 0 .. m.k {
				nq[i][j] = (q[ip][j] + q[i][j] + q[inext][j]) / 3.0
			}
		}
		q = nq.clone()
	}
	return q
}

// maximize re-estimates each layer velocity by assignment-weighted averaging.
fn (m MotionLayersModel) maximize(q [][]f64, observation []f64, params []f64, damping f64) []f64 {
	mut new := []f64{len: m.k}
	for j in 0 .. m.k {
		mut num := 0.0
		mut den := 0.0
		for i in 0 .. m.n {
			num += q[i][j] * observation[i]
			den += q[i][j]
		}
		new[j] = num / math.max(den, 1e-12)
	}
	if damping > 0.0 {
		for j in 0 .. m.k {
			new[j] = (1.0 - damping) * new[j] + damping * params[j]
		}
	}
	return new
}

// log_likelihood returns the mixture velocity log likelihood.
fn (m MotionLayersModel) log_likelihood(params []f64, observation []f64) f64 {
	mut ll := 0.0
	for i in 0 .. m.n {
		mut rowmax := -1e300
		mut logp := []f64{len: m.k}
		for j in 0 .. m.k {
			d := observation[i] - params[j]
			logp[j] = -d * d / (2.0 * m.sigma * m.sigma)
			if logp[j] > rowmax {
				rowmax = logp[j]
			}
		}
		mut s := 0.0
		for j in 0 .. m.k {
			s += math.exp(logp[j] - rowmax)
		}
		ll += math.log(s / f64(m.k)) + rowmax
	}
	return ll
}
