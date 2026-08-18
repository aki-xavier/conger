module conger

// soft_icp.v — EM-ICP soft correspondence + rigid transform (GenericEM instance).
//
// Source points S and target points T_obs; latent = each target point's soft
// correspondence (which source point generated it); θ = rigid transform
// (rotation θ + translation t). E step: soft correspondence; M step: expected
// source → Kabsch rigid fit.

import math

struct SoftICPModel {
	source [][]f64 // (N,2)
	sigma  f64
	n      int
}

fn new_soft_icp_model(source [][]f64, sigma f64) SoftICPModel {
	assert source.len > 0 && source[0].len == 2
	return SoftICPModel{
		source: source
		sigma: sigma
		n: source.len
	}
}

fn rotation_matrix(theta f64) [][]f64 {
	c := math.cos(theta)
	s := math.sin(theta)
	mut r := [][]f64{len: 2, init: []f64{len: 2}}
	r[0][0] = c
	r[0][1] = -s
	r[1][0] = s
	r[1][1] = c
	return r
}

// transform applies (θ, tx, ty) to the source points.
fn (m SoftICPModel) transform(params []f64) [][]f64 {
	theta := params[0]
	tx := params[1]
	ty := params[2]
	r := rotation_matrix(theta)
	mut out := [][]f64{len: m.n, init: []f64{len: 2}}
	for i in 0 .. m.n {
		out[i][0] = r[0][0] * m.source[i][0] + r[0][1] * m.source[i][1] + tx
		out[i][1] = r[1][0] * m.source[i][0] + r[1][1] * m.source[i][1] + ty
	}
	return out
}

// sample draws a target point cloud = transformed source + noise.
fn (m SoftICPModel) sample(params []f64, seed u64) [][]f64 {
	t := m.transform(params)
	mut rng := new_rng(seed)
	mut out := [][]f64{len: m.n, init: []f64{len: 2}}
	for i in 0 .. m.n {
		out[i][0] = t[i][0] + rng.normal(0.0, m.sigma)
		out[i][1] = t[i][1] + rng.normal(0.0, m.sigma)
	}
	return out
}

// responsibilities returns soft correspondences q(j,i) (N,N).
fn (m SoftICPModel) responsibilities(params []f64, observation [][]f64, temperature f64) [][]f64 {
	t := m.transform(params)
	inv := 1.0 / math.max(temperature, 1e-8)
	mut q := [][]f64{len: m.n, init: []f64{len: m.n}}
	for j in 0 .. m.n {
		mut rowmax := -1e300
		for i in 0 .. m.n {
			d0 := observation[j][0] - t[i][0]
			d1 := observation[j][1] - t[i][1]
			d := d0 * d0 + d1 * d1
			q[j][i] = -d / (2.0 * m.sigma * m.sigma) * inv
			if q[j][i] > rowmax {
				rowmax = q[j][i]
			}
		}
		mut s := 0.0
		for i in 0 .. m.n {
			q[j][i] = math.exp(q[j][i] - rowmax)
			s += q[j][i]
		}
		for i in 0 .. m.n {
			q[j][i] /= s
		}
	}
	return q
}

// maximize estimates the rigid transform via Kabsch on the expected source.
fn (m SoftICPModel) maximize(resp [][]f64, observation [][]f64, params []f64, damping f64) []f64 {
	// expected source ŝ_j = Σ_i q(j,i) · s_i  (N,2)
	mut expected := [][]f64{len: m.n, init: []f64{len: 2}}
	for j in 0 .. m.n {
		for i in 0 .. m.n {
			expected[j][0] += resp[j][i] * m.source[i][0]
			expected[j][1] += resp[j][i] * m.source[i][1]
		}
	}
	r, t := kabsch_2d(expected, observation)
	theta := math.atan2(r[1][0], r[0][0])
	mut new := [theta, t[0], t[1]]
	if damping > 0.0 {
		for i in 0 .. 3 {
			new[i] = (1.0 - damping) * new[i] + damping * params[i]
		}
	}
	return new
}

// log_likelihood returns the mixture correspondence log likelihood.
fn (m SoftICPModel) log_likelihood(params []f64, observation [][]f64) f64 {
	t := m.transform(params)
	mut ll := 0.0
	for j in 0 .. m.n {
		mut rowmax := -1e300
		mut logp := []f64{len: m.n}
		for i in 0 .. m.n {
			d0 := observation[j][0] - t[i][0]
			d1 := observation[j][1] - t[i][1]
			d := d0 * d0 + d1 * d1
			logp[i] = -d / (2.0 * m.sigma * m.sigma)
			if logp[i] > rowmax {
				rowmax = logp[i]
			}
		}
		mut s := 0.0
		for i in 0 .. m.n {
			s += math.exp(logp[i] - rowmax)
		}
		ll += rowmax + math.log(s)
	}
	return ll
}
