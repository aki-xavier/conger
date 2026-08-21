module conger

// deq_implicit_test.v — validates the IFT-based DEQ operator (deq.v) against
// two independent references on the same GMRF-chain problem as the unroll
// baseline (deq_learning_test.v): (1) the exact linear-solve fixed point, and
// (2) a finite-difference gradient of the loss. Gradient descent with the
// implicit gradient (NO unrolling) must recover the coupling β.
//
// This is the non-unrolling counterpart the unroll baseline documents as the
// next step: forward = capped contractive iteration (real-time friendly),
// backward = implicit function theorem through the adjoint fixed point.
import math

fn dq_gibbs_chain(n int, beta f64, sig2 f64, sweeps int, seed u64) []f64 {
	mut rng := new_rng(seed)
	mut x := []f64{len: n}
	for i in 0 .. n {
		x[i] = rng.normal(0.0, 1.0)
	}
	for _ in 0 .. sweeps {
		for i in 0 .. n {
			mut c := 0.0
			if i > 0 {
				c += beta * x[i - 1]
			}
			if i + 1 < n {
				c += beta * x[i + 1]
			}
			x[i] = c + rng.normal(0.0, math.sqrt(sig2))
		}
	}
	return x
}

fn dq_adjacency(n int) [][]f64 {
	mut a := [][]f64{len: n, init: []f64{len: n}}
	for i in 0 .. n {
		if i > 0 {
			a[i][i - 1] = 1.0
		}
		if i + 1 < n {
			a[i][i + 1] = 1.0
		}
	}
	return a
}

// dq_gmrf_map builds the affine map A(β)= (β/3)·adj, b = y_scaled (constant).
fn dq_gmrf_map(adj [][]f64, y_scaled []f64) AffineMap {
	return fn [adj, y_scaled] (theta []f64) ([][]f64, []f64) {
		beta := theta[0]
		mut a := [][]f64{len: adj.len, init: []f64{len: adj.len}}
		for i in 0 .. adj.len {
			for j in 0 .. adj.len {
				a[i][j] = beta / 3.0 * adj[i][j]
			}
		}
		return a, y_scaled
	}
}

// dq_exact_fixed_point solves (I − A)x = b directly.
fn dq_exact_fixed_point(adj [][]f64, y_scaled []f64, beta f64) []f64 {
	n := y_scaled.len
	mut sys := [][]f64{len: n, init: []f64{len: n}}
	for i in 0 .. n {
		for j in 0 .. n {
			sys[i][j] = -beta / 3.0 * adj[i][j]
		}
		sys[i][i] += 1.0
	}
	return solve_n(sys, y_scaled)
}

fn dq_mse(x []f64, true_ []f64) f64 {
	mut s := 0.0
	for i in 0 .. x.len {
		s += (x[i] - true_[i]) * (x[i] - true_[i])
	}
	return s / f64(x.len)
}

fn dq_maxdiff(a []f64, b []f64) f64 {
	mut m := 0.0
	for i in 0 .. a.len {
		d := math.abs(a[i] - b[i])
		if d > m {
			m = d
		}
	}
	return m
}

fn test_deq_forward_matches_exact_fixed_point() {
	n := 64
	beta := 0.2
	tau2 := 0.25
	field := dq_gibbs_chain(n, 0.2, 0.5, 400, 31)
	mut rng := new_rng(32)
	mut ys := []f64{cap: n}
	for v in field {
		ys << v + rng.normal(0.0, math.sqrt(tau2))
	}
	mut y_scaled := []f64{cap: n}
	for v in ys {
		y_scaled << v / (6.0 * tau2)
	}
	adj := dq_adjacency(n)
	mp := dq_gmrf_map(adj, y_scaled)
	x0 := []f64{len: n, init: 0.0}
	xstar, conv := deq_forward(mp, [beta], x0, 1e-8, 2000)
	assert conv, 'contractive map must converge'
	exact := dq_exact_fixed_point(adj, y_scaled, beta)
	assert dq_maxdiff(xstar, exact) < 1e-6
}

fn test_deq_implicit_grad_matches_finite_difference() {
	n := 64
	tau2 := 0.25
	field := dq_gibbs_chain(n, 0.2, 0.5, 400, 31)
	mut rng := new_rng(32)
	mut ys := []f64{cap: n}
	for v in field {
		ys << v + rng.normal(0.0, math.sqrt(tau2))
	}
	mut y_scaled := []f64{cap: n}
	for v in ys {
		y_scaled << v / (6.0 * tau2)
	}
	adj := dq_adjacency(n)
	mp := dq_gmrf_map(adj, y_scaled)
	beta := 0.2
	xstar, conv := deq_forward(mp, [beta], []f64{len: n, init: 0.0}, 1e-10, 3000)
	assert conv
	// L = mean((x−field)²), dl/dx_i = 2(x_i−field_i)/n
	mut d_ldx := []f64{len: n}
	for i in 0 .. n {
		d_ldx[i] = 2.0 * (xstar[i] - field[i]) / f64(n)
	}
	g_ift := deq_grad(n, mp, [beta], xstar, d_ldx)[0]
	// finite-difference reference: perturb β, recompute the *exact* fixed
	// point and the loss.
	h := 1e-6
	lp := dq_mse(dq_exact_fixed_point(adj, y_scaled, beta + h), field)
	lm := dq_mse(dq_exact_fixed_point(adj, y_scaled, beta - h), field)
	g_fd := (lp - lm) / (2.0 * h)
	assert math.abs(g_ift - g_fd) < 1e-3 * math.max(1.0, math.abs(g_fd))
}

fn test_deq_ift_gradient_descent_recovers_beta() {
	n := 64
	tau2 := 0.25
	field := dq_gibbs_chain(n, 0.2, 0.5, 400, 31)
	mut rng := new_rng(32)
	mut ys := []f64{cap: n}
	for v in field {
		ys << v + rng.normal(0.0, math.sqrt(tau2))
	}
	mut y_scaled := []f64{cap: n}
	for v in ys {
		y_scaled << v / (6.0 * tau2)
	}
	adj := dq_adjacency(n)
	mp := dq_gmrf_map(adj, y_scaled)
	mut beta := 0.05
	mut first_loss := 0.0
	mut last_loss := 0.0
	for step in 0 .. 600 {
		xstar, conv := deq_forward(mp, [beta], []f64{len: n, init: 0.0}, 1e-10, 3000)
		assert conv
		loss := dq_mse(xstar, field)
		if step == 0 {
			first_loss = loss
		}
		last_loss = loss
		mut d_ldx := []f64{len: n}
		for i in 0 .. n {
			d_ldx[i] = 2.0 * (xstar[i] - field[i]) / f64(n)
		}
		g := deq_grad(n, mp, [beta], xstar, d_ldx)[0]
		beta -= 0.05 * g
	}
	// independent reference: grid-search the sample-MSE-optimal β on the
	// exact fixed point.
	mut best_beta, mut best_mse := 0.0, 1e300
	for gi in 0 .. 101 {
		b := 0.005 * f64(gi)
		m := dq_exact_fixed_point(adj, y_scaled, b)
		mse := dq_mse(m, field)
		if mse < best_mse {
			best_mse = mse
			best_beta = b
		}
	}
	println('deq-ift: beta 0.05 -> ${beta:.4f} (grid-optimal ${best_beta:.3f}), loss ${first_loss:.4f} -> ${last_loss:.4f}')
	assert last_loss < first_loss
	assert math.abs(beta - best_beta) < 0.03
}
