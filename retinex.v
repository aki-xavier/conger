module conger

// retinex.v — albedo↔lighting multiplicative decomposition (Retinex coordinate
// ascent, GenericEM instance).
//
// I(x) = A(x)·L(x); in log domain log I = log A + log L. Albedo log A is
// piecewise constant (K segments with known boundaries), lighting log L is
// smooth (linear). This is coordinate ascent (no latent variable); mapped onto
// GenericEM by treating lighting (l0,l1) as θ and albedo log A as the E-step
// "responsibilities".
import math

struct RetinexModel {
	segments []int
	sigma    f64
	n        int
	k        int
	x        []f64
	xm       [][]f64 // (n,2) [1, x]
}

fn new_retinex_model(segments []int, sigma f64) RetinexModel {
	n := segments.len
	mut k := 0
	for s in segments {
		if s + 1 > k {
			k = s + 1
		}
	}
	mut x := arange_f64(n)
	for i in 0 .. n {
		x[i] = x[i] / f64(max_int(n - 1, 1))
	}
	mut xm := [][]f64{len: n, init: []f64{len: 2}}
	for i in 0 .. n {
		xm[i][0] = 1.0
		xm[i][1] = x[i]
	}
	return RetinexModel{
		segments: segments
		sigma:    sigma
		n:        n
		k:        k
		x:        x
		xm:       xm
	}
}

fn max_int(a int, b int) int {
	return if a > b { a } else { b }
}

fn (m RetinexModel) log_l(params []f64) []f64 {
	l0 := params[0]
	l1 := params[1]
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		out[i] = l0 + l1 * m.x[i]
	}
	return out
}

// render builds I = exp(log A + log L + noise).
fn (m RetinexModel) render(log_a_seg []f64, params []f64, seed u64) []f64 {
	logl := m.log_l(params)
	mut rng := new_rng(seed)
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		loga := log_a_seg[m.segments[i]]
		out[i] = math.exp(loga + logl[i] + rng.normal(0.0, m.sigma))
	}
	return out
}

// sample is the interface-consistency forward model (all-zero albedo).
fn (m RetinexModel) sample(params []f64, seed u64) []f64 {
	return m.render([]f64{len: m.k}, params, seed)
}

// responsibilities estimates piecewise-constant log A (centred to break the
// scale freedom), returned per-pixel.
fn (m RetinexModel) responsibilities(params []f64, observation []f64, _ f64) []f64 {
	mut log_i := []f64{len: m.n}
	logl := m.log_l(params)
	for i in 0 .. m.n {
		log_i[i] = math.log(observation[i] + 1e-12)
	}
	mut resid := []f64{len: m.n}
	for i in 0 .. m.n {
		resid[i] = log_i[i] - logl[i]
	}
	mut seg_mean := []f64{len: m.k}
	mut seg_size := []f64{len: m.k}
	for i in 0 .. m.n {
		seg_mean[m.segments[i]] += resid[i]
		seg_size[m.segments[i]] += 1.0
	}
	for j in 0 .. m.k {
		if seg_size[j] > 0 {
			seg_mean[j] /= seg_size[j]
		}
	}
	mut total := 0.0
	for j in 0 .. m.k {
		total += seg_mean[j] * seg_size[j]
	}
	total /= f64(m.n)
	for j in 0 .. m.k {
		seg_mean[j] -= total
	}
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		out[i] = seg_mean[m.segments[i]]
	}
	return out
}

// maximize re-estimates lighting via linear fit of (log I − log A).
fn (m RetinexModel) maximize(resp []f64, observation []f64, params []f64, damping f64) []f64 {
	mut log_i := []f64{len: m.n}
	for i in 0 .. m.n {
		log_i[i] = math.log(observation[i] + 1e-12)
	}
	mut y := []f64{len: m.n}
	for i in 0 .. m.n {
		y[i] = log_i[i] - resp[i]
	}
	mut coef := lstsq_2(m.xm, y)
	if damping > 0.0 {
		coef[0] = (1.0 - damping) * coef[0] + damping * params[0]
		coef[1] = (1.0 - damping) * coef[1] + damping * params[1]
	}
	return coef
}

// log_likelihood returns the negative log-domain reconstruction residual.
fn (m RetinexModel) log_likelihood(params []f64, observation []f64) f64 {
	mut log_i := []f64{len: m.n}
	for i in 0 .. m.n {
		log_i[i] = math.log(observation[i] + 1e-12)
	}
	loga := m.responsibilities(params, observation, 1.0)
	logl := m.log_l(params)
	mut s := 0.0
	for i in 0 .. m.n {
		d := log_i[i] - loga[i] - logl[i]
		s += d * d
	}
	return -s / (2.0 * m.sigma * m.sigma)
}
