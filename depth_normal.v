module conger

// depth_normal.v — depth↔normal coordinate ascent (GenericEM instance).
//
// 1D surface: depth z(x), normal tangent projection s(x) = dz/dx. Observations
// give noisy depth ẑ and noisy slope ŝ, coupled by s = dz/dx:
//
//   E step: fixed z, reconciled normal s = (ŝ + dz/dx) / 2
//   M step: fixed s, fit depth z = argmin ‖z−ẑ‖² + λ·‖Dz−s‖² (Tikhonov)

struct DepthNormalModel {
	z_obs []f64
	s_obs []f64
	lam   f64
	sigma f64
	n     int
	d     [][]f64 // (n-1, n) difference operator
}

fn new_depth_normal_model(z_obs []f64, s_obs []f64, lam f64, sigma f64) DepthNormalModel {
	assert z_obs.len == s_obs.len + 1
	n := z_obs.len
	mut d := [][]f64{len: n - 1, init: []f64{len: n}}
	for i in 0 .. n - 1 {
		d[i][i] = -1.0
		d[i][i + 1] = 1.0
	}
	return DepthNormalModel{
		z_obs: z_obs
		s_obs: s_obs
		lam: lam
		sigma: sigma
		n: n
		d: d
	}
}

// responsibilities returns the reconciled slope s = (ŝ + dz/dx)/2.
fn (m DepthNormalModel) responsibilities(z []f64, _ []f64, _ f64) []f64 {
	grad := diff(z)
	mut out := []f64{len: grad.len}
	for i in 0 .. grad.len {
		out[i] = (m.s_obs[i] + grad[i]) / 2.0
	}
	return out
}

// maximize solves the Tikhonov linear system (I + λDᵀD) z = ẑ + λDᵀs.
fn (m DepthNormalModel) maximize(resp []f64, _ []f64, z []f64, damping f64) []f64 {
	// A = I + λ DᵀD
	mut aa := [][]f64{len: m.n, init: []f64{len: m.n}}
	for i in 0 .. m.n {
		aa[i][i] = 1.0
	}
	for k in 0 .. m.n - 1 {
		for i in 0 .. m.n {
			for j in 0 .. m.n {
				aa[i][j] += m.lam * m.d[k][i] * m.d[k][j]
			}
		}
	}
	// b = ẑ + λ Dᵀs
	mut b := m.z_obs.clone()
	for k in 0 .. m.n - 1 {
		for j in 0 .. m.n {
			b[j] += m.lam * m.d[k][j] * resp[k]
		}
	}
	mut z_new := solve_n(aa, b)
	if damping > 0.0 {
		for i in 0 .. m.n {
			z_new[i] = (1.0 - damping) * z_new[i] + damping * z[i]
		}
	}
	return z_new
}

// log_likelihood returns the negative joint residual.
fn (m DepthNormalModel) log_likelihood(z []f64, _ []f64) f64 {
	grad := diff(z)
	mut s := 0.0
	for i in 0 .. m.n {
		d := z[i] - m.z_obs[i]
		s += d * d
	}
	for i in 0 .. grad.len {
		d := grad[i] - m.s_obs[i]
		s += m.lam * d * d
	}
	return -s / (2.0 * m.sigma * m.sigma)
}
