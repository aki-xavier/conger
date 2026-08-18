module conger

// transparent_layers.v — two-layer transparent superposition EM instance (1D probe).
//
// Soft-assignment mixture: Z(x) ∈ {1,2} latent, P(Z=1|x) = α(x), with
// I(x)|Z(x)=k ~ N(c_k, σ²). E step computes per-pixel soft assignment; M step
// re-estimates the two uniform layer intensities by weighted averaging.

import math

struct TransparentLayerModel {
	alpha []f64
	sigma f64
	n     int
}

fn new_transparent_layer_model(alpha []f64, sigma f64) TransparentLayerModel {
	for a in alpha {
		if a < 0.0 || a > 1.0 {
			panic('alpha must be in [0,1]')
		}
	}
	return TransparentLayerModel{
		alpha: alpha
		sigma: sigma
		n: alpha.len
	}
}

// sample draws an observation from layer intensities (c1, c2).
fn (m TransparentLayerModel) sample(params []f64, seed u64) []f64 {
	c1 := params[0]
	c2 := params[1]
	mut rng := new_rng(seed)
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		z := rng.f64() < m.alpha[i]
		val := if z { c1 } else { c2 }
		out[i] = val + rng.normal(0.0, m.sigma)
	}
	return out
}

// responsibilities returns per-pixel soft posterior q(Z=1|x).
fn (m TransparentLayerModel) responsibilities(params []f64, observation []f64, temperature f64) []f64 {
	c1 := params[0]
	c2 := params[1]
	inv := 1.0 / math.max(temperature, 1e-8)
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		d := observation[i]
		logp1 := -0.5 * ((d - c1) / m.sigma) * ((d - c1) / m.sigma) * inv
		logp2 := -0.5 * ((d - c2) / m.sigma) * ((d - c2) / m.sigma) * inv
		w1 := math.log(m.alpha[i] + 1e-12) + logp1
		w2 := math.log(1.0 - m.alpha[i] + 1e-12) + logp2
		mm := math.max(w1, w2)
		e1 := math.exp(w1 - mm)
		e2 := math.exp(w2 - mm)
		out[i] = e1 / (e1 + e2)
	}
	return out
}

// maximize re-estimates the two intensities by weighted averaging.
fn (m TransparentLayerModel) maximize(resp []f64, observation []f64, params []f64, damping f64) []f64 {
	c1_old := params[0]
	c2_old := params[1]
	mut s := 0.0
	mut n1 := 0.0
	mut n2 := 0.0
	for i in 0 .. m.n {
		s += resp[i]
		n1 += resp[i] * observation[i]
		n2 += (1.0 - resp[i]) * observation[i]
	}
	mut c1 := n1 / math.max(s, 1e-12)
	mut c2 := n2 / math.max(f64(m.n) - s, 1e-12)
	if damping > 0.0 {
		c1 = (1.0 - damping) * c1 + damping * c1_old
		c2 = (1.0 - damping) * c2 + damping * c2_old
	}
	return [c1, c2]
}

// log_likelihood returns the mixture-Gaussian observation log likelihood.
fn (m TransparentLayerModel) log_likelihood(params []f64, observation []f64) f64 {
	c1 := params[0]
	c2 := params[1]
	mut s := 0.0
	for i in 0 .. m.n {
		d := observation[i]
		p1 := m.alpha[i] * math.exp(-0.5 * ((d - c1) / m.sigma) * ((d - c1) / m.sigma))
		p2 := (1.0 - m.alpha[i]) * math.exp(-0.5 * ((d - c2) / m.sigma) * ((d - c2) / m.sigma))
		s += math.log(p1 + p2 + 1e-12)
	}
	return s
}
