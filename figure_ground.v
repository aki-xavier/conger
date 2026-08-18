module conger

// figure_ground.v — segmentation↔pose EM (1D figure-ground).
//
// Foreground is an interval [c−r, c+r] (pose = centre c + half-width r) with
// intensity f; background intensity b. Latent = per-pixel foreground
// assignment; θ = (c, r, f, b). E step: soft assignment (soft pose prior ×
// intensity likelihood); M step: intensities via weighted averaging, pose via
// coordinate search minimising the negative mixture log likelihood.

import math

struct FigureGroundModel {
	n        int
	sigma    f64
	delta_c  f64
	delta_r  f64
	boundary f64
	x        []f64
}

fn new_figure_ground_model(n int, sigma f64, delta_c f64, delta_r f64, boundary f64) FigureGroundModel {
	return FigureGroundModel{
		n: n
		sigma: sigma
		delta_c: delta_c
		delta_r: delta_r
		boundary: boundary
		x: linspace(0.0, 1.0, n)
	}
}

fn (m FigureGroundModel) fg_mask(c f64, r f64) []bool {
	mut out := []bool{len: m.n}
	for i in 0 .. m.n {
		out[i] = math.abs(m.x[i] - c) <= r
	}
	return out
}

fn (m FigureGroundModel) fg_prior(c f64, r f64) []f64 {
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		out[i] = 1.0 / (1.0 + math.exp((math.abs(m.x[i] - c) - r) / m.boundary))
	}
	return out
}

fn (m FigureGroundModel) mixture_ll(c f64, r f64, f f64, b f64, observation []f64) f64 {
	prior := m.fg_prior(c, r)
	mut s := 0.0
	for i in 0 .. m.n {
		p := prior[i] * math.exp(-0.5 * ((observation[i] - f) / m.sigma) * ((observation[i] - f) / m.sigma)) +
			(1.0 - prior[i]) * math.exp(-0.5 * ((observation[i] - b) / m.sigma) * ((observation[i] - b) / m.sigma))
		s += math.log(p + 1e-12)
	}
	return s
}

// sample draws a noisy observation from (c, r, f, b).
fn (m FigureGroundModel) sample(params []f64, seed u64) []f64 {
	c := params[0]
	r := params[1]
	f := params[2]
	b := params[3]
	mask := m.fg_mask(c, r)
	mut rng := new_rng(seed)
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		val := if mask[i] { f } else { b }
		out[i] = val + rng.normal(0.0, m.sigma)
	}
	return out
}

// responsibilities returns soft foreground assignment q(x).
fn (m FigureGroundModel) responsibilities(params []f64, observation []f64, temperature f64) []f64 {
	c := params[0]
	r := params[1]
	f := params[2]
	b := params[3]
	prior := m.fg_prior(c, r)
	inv := 1.0 / math.max(temperature, 1e-8)
	mut out := []f64{len: m.n}
	for i in 0 .. m.n {
		df := (observation[i] - f) / m.sigma
		db := (observation[i] - b) / m.sigma
		wf := math.log(prior[i] + 1e-12) - 0.5 * df * df * inv
		wb := math.log(1.0 - prior[i] + 1e-12) - 0.5 * db * db * inv
		mm := math.max(wf, wb)
		ef := math.exp(wf - mm)
		eb := math.exp(wb - mm)
		out[i] = ef / (ef + eb)
	}
	return out
}

// maximize re-estimates intensities and searches pose coordinates.
fn (m FigureGroundModel) maximize(q []f64, observation []f64, params []f64, damping f64) []f64 {
	c := params[0]
	r := params[1]
	mut sq := 0.0
	mut nf := 0.0
	mut nb := 0.0
	for i in 0 .. m.n {
		sq += q[i]
		nf += q[i] * observation[i]
		nb += (1.0 - q[i]) * observation[i]
	}
	f := nf / math.max(sq, 1e-12)
	b := nb / math.max(f64(m.n) - sq, 1e-12)

	cost := fn [m, f, b, observation] (cc f64, rr f64) f64 {
		if rr <= 0.0 {
			return 1e9
		}
		return -m.mixture_ll(cc, rr, f, b, observation)
	}

	mut best_c := c
	mut best_r := r
	mut best_e := cost(c, r)
	for _, dc in [-m.delta_c, 0.0, m.delta_c] {
		for _, dr in [-m.delta_r, 0.0, m.delta_r] {
			cc := c + dc
			rr := r + dr
			if rr <= 0.0 {
				continue
			}
			e := cost(cc, rr)
			if e < best_e {
				best_c = cc
				best_r = rr
				best_e = e
			}
		}
	}
	mut new := [best_c, best_r, f, b]
	if damping > 0.0 {
		for i in 0 .. 4 {
			new[i] = (1.0 - damping) * new[i] + damping * params[i]
		}
	}
	return new
}

// log_likelihood returns the soft-prior mixture log likelihood.
fn (m FigureGroundModel) log_likelihood(params []f64, observation []f64) f64 {
	return m.mixture_ll(params[0], params[1], params[2], params[3], observation)
}
