module conger

// deq.v — Deep-Equilibrium (DEQ) inference + implicit-differentiation
// training for the linear-contractive fixed point that likelihood-kernel
// networks realize.
//
// A kernel network with linear kernels (e.g. a Gaussian-MRF lattice or the
// coupled-factor Gauss-Seidel loop) reaches its steady state at an *affine*
// fixed point
//
//     x* = A(θ)·x* + b(θ)
//
// where A is the step-to-step contraction (whose spectral radius
// `feedback_spectral_radius` estimates) and θ the kernel parameters. Training
// θ on a loss at x* is a DEQ problem: forward = solve the fixed point, and the
// gradient can be obtained WITHOUT unrolling the relaxation, via the implicit
// function theorem (IFT):
//
//     (I − A)ᵀ λ = ∂L/∂x,     ∂L/∂θ_j = −λᵀ( ∂A/∂θ_j · x* + ∂b/∂θ_j )
//
// This module supplies that operator in pure f64 (the kernel/state domain).
// It is the non-unrolling counterpart of the `deq_learning_test.v` baseline,
// and makes the real-time forward (bounded contractive iteration, capped for
// a frame budget) usable directly as the "inference = DEQ" stage while the
// structure gate/birth/template loop keeps operating at low frequency to hand
// the next operator its graph.
//
// Kernels whose relaxation is not affine should supply their own map through
// the same `AffineMap` signature; only the Jacobian of the map w.r.t. x is
// assumed constant (A), which is exactly the linear/contractive regime the
// repo's stability check guards.
import math

// AffineMap maps parameters θ to the affine fixed-point operator
// (A, b) of the relaxation, i.e. x* = A·x* + b. A is row-major [n][n], b is [n].
pub type AffineMap = fn (theta []f64) ([][]f64, []f64)

// deq_forward solves the affine fixed point by contractive iteration from x0,
// capped at `max_iter` (for a real-time frame budget), stopping once
// |x_{k+1} − x_k|_∞ < tol. Returns the fixed point and whether it converged.
pub fn deq_forward(mp AffineMap, theta []f64, x0 []f64, tol f64, max_iter int) ([]f64, bool) {
	// NB: explicit panic, not `assert` — V strips asserts in `-prod` builds.
	if x0.len == 0 {
		panic('deq_forward: x0 must be non-empty')
	}
	if max_iter < 1 {
		panic('deq_forward: max_iter must be >= 1 (got ${max_iter})')
	}
	if tol <= 0.0 {
		panic('deq_forward: tol must be > 0 (got ${tol})')
	}
	a, b := mp(theta)
	if a.len != x0.len || b.len != x0.len {
		panic('deq_forward: map dim mismatch (A ${a.len}, b ${b.len}, x0 ${x0.len})')
	}
	for i, row in a {
		if row.len != x0.len {
			panic('deq_forward: A row ${i} width ${row.len} != ${x0.len}')
		}
	}
	mut x := x0.clone()
	mut converged := false
	for _ in 0 .. max_iter {
		mut xn := []f64{len: x.len}
		for i in 0 .. x.len {
			mut s := b[i]
			for j in 0 .. x.len {
				s += a[i][j] * x[j]
			}
			xn[i] = s
		}
		mut dmax := 0.0
		for i in 0 .. x.len {
			d := math.abs(xn[i] - x[i])
			if d > dmax {
				dmax = d
			}
		}
		x = xn.clone()
		if dmax < tol {
			converged = true
			break
		}
	}
	return x, converged
}

// deq_grad returns ∂L/∂θ for a loss whose fixed-point gradient is `d_ldx`
// (i.e. dl/dx* is already computed), via the implicit function theorem — no
// unrolling. It solves the adjoint (I − A)ᵀ λ = d_ldx and differentiates the
// map w.r.t. θ numerically (central difference), so it needs only the map and
// the converged fixed point.
pub fn deq_grad(n int, mp AffineMap, theta []f64, x_star []f64, d_ldx []f64) []f64 {
	// NB: explicit panic, not `assert` — V strips asserts in `-prod` builds.
	if x_star.len != n || d_ldx.len != n {
		panic('deq_grad: dim mismatch (n ${n}, x_star ${x_star.len}, d_ldx ${d_ldx.len})')
	}
	if theta.len == 0 {
		return []f64{}
	}
	a, _ := mp(theta)
	// adjoint system: (I − A)ᵀ λ = d_ldx, i.e. row j: λ[j] − Σ_i A[i][j]·λ[i] = d_ldx[j]
	mut adj := [][]f64{len: n, init: []f64{len: n}}
	for j in 0 .. n {
		for i in 0 .. n {
			adj[j][i] = -a[i][j]
		}
		adj[j][j] += 1.0
	}
	lam := solve_n(adj, d_ldx)
	mut grads := []f64{len: theta.len}
	h := 1e-6
	for tj in 0 .. theta.len {
		mut hp := theta.clone()
		mut hm := theta.clone()
		step := h * math.max(1.0, math.abs(theta[tj]))
		hp[tj] += step
		hm[tj] -= step
		ap, bp := mp(hp)
		am, bm := mp(hm)
		inv2 := 0.5 / step
		mut dl := 0.0
		for i in 0 .. n {
			mut da := 0.0
			for j in 0 .. n {
				da += (ap[i][j] - am[i][j]) * inv2 * x_star[j]
			}
			db := (bp[i] - bm[i]) * inv2
			dl += lam[i] * (da + db)
		}
		grads[tj] = dl
	}
	return grads
}
