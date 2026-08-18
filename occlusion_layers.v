module conger

// occlusion_layers.v — two-layer occlusion + depth-order EM instance (1D probe).
//
// L1 occupies [0,b], L2 occupies [a,1], overlap [a,b] (a<b). Latent D ∈ {0,1}
// is the depth order (0 = L1 in front); params θ = (a0,a1,b0,b1) are the two
// linear layer coefficients.
import math

struct OcclusionLayerModel {
	n            int
	a            f64
	b            f64
	sigma        f64
	x            []f64
	xm           [][]f64 // (n,2) [1, x]
	mask_1_only  []bool
	mask_2_only  []bool
	mask_overlap []bool
}

fn new_occlusion_layer_model(n int, a f64, b f64, sigma f64) OcclusionLayerModel {
	assert 0.0 < a && a < b && b < 1.0
	x := linspace(0.0, 1.0, n)
	mut xm := [][]f64{len: n, init: []f64{len: 2}}
	mut m1 := []bool{len: n}
	mut m2 := []bool{len: n}
	mut mo := []bool{len: n}
	for i in 0 .. n {
		xm[i][0] = 1.0
		xm[i][1] = x[i]
		m1[i] = x[i] < a
		m2[i] = x[i] > b
		mo[i] = x[i] >= a && x[i] <= b
	}
	return OcclusionLayerModel{
		n:            n
		a:            a
		b:            b
		sigma:        sigma
		x:            x
		xm:           xm
		mask_1_only:  m1
		mask_2_only:  m2
		mask_overlap: mo
	}
}

fn (m OcclusionLayerModel) layer(coeff []f64) []f64 {
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		out[i] = coeff[0] + coeff[1] * m.x[i]
	}
	return out
}

// render builds an observation from (a0,a1,b0,b1) with `front` the front layer.
fn (m OcclusionLayerModel) render(params []f64, front int, seed u64) []f64 {
	l1 := m.layer(params[..2])
	l2 := m.layer(params[2..])
	mut rng := new_rng(seed)
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		mut v := 0.0
		if m.mask_1_only[i] {
			v = l1[i]
		} else if m.mask_2_only[i] {
			v = l2[i]
		} else if front == 0 {
			v = l1[i]
		} else {
			v = l2[i]
		}
		out[i] = v + rng.normal(0.0, m.sigma)
	}
	return out
}

// sample draws an observation with depth order sampled uniformly.
fn (m OcclusionLayerModel) sample(params []f64, seed u64) []f64 {
	mut rng := new_rng(seed)
	front := if rng.f64() < 0.5 { 0 } else { 1 }
	// NOTE: render re-seeds its own RNG with the same seed, so the noise stream
	// used by the returned observation is deterministic and independent of front.
	return m.render(params, front, seed)
}

// responsibilities returns the depth-order posterior q = P(D=0|I) as [P(D=0), P(D=1)].
fn (m OcclusionLayerModel) responsibilities(params []f64, observation []f64, temperature f64) []f64 {
	l1 := m.layer(params[..2])
	l2 := m.layer(params[2..])
	inv := 1.0 / math.max(temperature, 1e-8)
	mut r1 := 0.0
	mut r2 := 0.0
	for i in 0 .. m.n {
		if m.mask_overlap[i] {
			o := observation[i]
			r1 += (o - l1[i]) * (o - l1[i])
			r2 += (o - l2[i]) * (o - l2[i])
		}
	}
	r1 *= inv
	r2 *= inv
	logq0 := -r1 / (2.0 * m.sigma * m.sigma)
	logq1 := -r2 / (2.0 * m.sigma * m.sigma)
	mm := math.max(logq0, logq1)
	e0 := math.exp(logq0 - mm)
	e1 := math.exp(logq1 - mm)
	s := e0 + e1
	return [e0 / s, e1 / s]
}

// maximize re-estimates the two linear layers via weighted fits.
fn (m OcclusionLayerModel) maximize(resp []f64, observation []f64, params []f64, damping f64) []f64 {
	q := resp[0]
	mut w1 := []f64{len: m.n}
	mut w2 := []f64{len: m.n}
	for i in 0 .. m.n {
		if m.mask_1_only[i] {
			w1[i] = 1.0
			w2[i] = 0.0
		} else if m.mask_2_only[i] {
			w1[i] = 0.0
			w2[i] = 1.0
		} else {
			w1[i] = q
			w2[i] = 1.0 - q
		}
	}
	mut c1 := m.fit(w1, observation)
	mut c2 := m.fit(w2, observation)
	if damping > 0.0 {
		c1[0] = (1.0 - damping) * c1[0] + damping * params[0]
		c1[1] = (1.0 - damping) * c1[1] + damping * params[1]
		c2[0] = (1.0 - damping) * c2[0] + damping * params[2]
		c2[1] = (1.0 - damping) * c2[1] + damping * params[3]
	}
	return [c1[0], c1[1], c2[0], c2[1]]
}

// fit solves the weighted 2-parameter linear least squares (with tiny ridge).
fn (m OcclusionLayerModel) fit(w []f64, observation []f64) []f64 {
	mut xtx := [][]f64{len: 2, init: []f64{len: 2}}
	mut xty := []f64{len: 2}
	for i in 0 .. m.n {
		x0 := m.xm[i][0]
		x1 := m.xm[i][1]
		wi := w[i]
		xtx[0][0] += wi * x0 * x0
		xtx[0][1] += wi * x0 * x1
		xtx[1][0] += wi * x0 * x1
		xtx[1][1] += wi * x1 * x1
		xty[0] += wi * x0 * observation[i]
		xty[1] += wi * x1 * observation[i]
	}
	xtx[0][0] += 1e-8
	xtx[1][1] += 1e-8
	return solve_2x2(xtx, xty)
}

// log_likelihood returns the depth-order mixture log likelihood.
fn (m OcclusionLayerModel) log_likelihood(params []f64, observation []f64) f64 {
	l1 := m.layer(params[..2])
	l2 := m.layer(params[2..])
	log_norm := -0.5 * math.log(2.0 * math.pi * m.sigma * m.sigma)
	mut ll := 0.0
	for i in 0 .. m.n {
		if m.mask_1_only[i] {
			d := (observation[i] - l1[i]) / m.sigma
			ll += -0.5 * d * d + log_norm
		} else if m.mask_2_only[i] {
			d := (observation[i] - l2[i]) / m.sigma
			ll += -0.5 * d * d + log_norm
		} else {
			o := observation[i]
			p := 0.5 * math.exp(-0.5 * ((o - l1[i]) / m.sigma) * ((o - l1[i]) / m.sigma)) +
				0.5 * math.exp(-0.5 * ((o - l2[i]) / m.sigma) * ((o - l2[i]) / m.sigma))
			ll += math.log(p + 1e-12)
		}
	}
	return ll
}
