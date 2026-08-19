module conger

// deq_learning_test.v — gradient-based kernel parameter learning through the
// fixed point (DEQ-style, via unrolling): the E step's relaxation is itself
// differentiable, so with mlx's value_and_grad we can fit a kernel parameter
// by gradient descent on a loss evaluated *at the fixed point* — no
// closed-form M step. Setup: 1-D GMRF chain (16 sites, known coupling
// β=0.2, σ²=0.5, observation noise τ²=0.25), learn β from init 0.05 by
// unrolling 40 Jacobi sweeps and descending mean squared error between the
// fixed point and the true field. (The implicit-function-theorem backward —
// solving the adjoint fixed point with the same recurrent machinery instead
// of unrolling — is the documented next step; unrolling is the correctness
// baseline.)
import math
import mlx

// gibbs_chain samples a 1-D GMRF chain (μ=0) by Gibbs sweeps.
fn gibbs_chain(n int, beta f64, sig2 f64, sweeps int, seed u64) []f64 {
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

fn chain_adjacency(n int) []f64 {
	mut a := []f64{len: n * n}
	for i in 0 .. n {
		if i > 0 {
			a[i * n + i - 1] = 1.0
		}
		if i + 1 < n {
			a[i * n + i + 1] = 1.0
		}
	}
	return a
}

// deq_loss unrolls the Jacobi relaxation and scores the fixed point:
// m' = y_scaled + (β/3)·A·m with y_scaled = y·(τ² factor) folded in,
// L = mean((m_T − x_true)²). params = [β, y_scaled, A, x_true].
fn deq_loss(params []mlx.Array) []mlx.Array {
	beta := params[0]
	y_scaled := params[1]
	amat := params[2]
	x_true := params[3]
	third := mlx.f32_scalar(1.0 / 3.0)
	defer {
		third.free()
	}
	mut m := mlx.zeros([64, 1], .float32)
	for _ in 0 .. 40 {
		coef := beta.multiply(third)
		m2 := y_scaled.add(amat.matmul(m).multiply(coef))
		m.free()
		coef.free()
		m = m2
	}
	return [m.subtract(x_true).square().mean()]
}

fn test_deq_gradient_learning_recovers_beta() {
	n := 64
	beta_true := 0.2
	tau2 := 0.25
	field := gibbs_chain(n, beta_true, 0.5, 400, 31)
	mut rng := new_rng(32)
	mut y := []f64{cap: n}
	for v in field {
		y << v + rng.normal(0.0, math.sqrt(tau2))
	}
	// m' = (y/τ² + β·A·m/σ²) · σ²τ²/(σ²+τ²); with σ²=0.5, τ²=0.25 the
	// observation weight is 1/6/τ² and the neighbour weight β/3
	mut y_scaled := []f64{cap: n}
	for v in y {
		y_scaled << v / 6.0 / tau2
	}
	amat := mlx.arr32(chain_adjacency(n), [n, n])
	ys := mlx.arr32(y_scaled, [n, 1])
	xt := mlx.arr32(field, [n, 1])
	defer {
		amat.free()
		ys.free()
		xt.free()
	}
	mut beta := mlx.f32_scalar(0.05)
	defer {
		beta.free()
	}
	mut vag := mlx.value_and_grad(deq_loss, [0])
	defer {
		vag.free()
	}
	lr := mlx.f32_scalar(0.2)
	defer {
		lr.free()
	}
	mut first_loss := f32(0.0)
	mut last_loss := f32(0.0)
	for step in 0 .. 1000 {
		values, grads := vag.apply([beta, ys, amat, xt])
		l := values[0].item_f32()
		if step == 0 {
			first_loss = l
		}
		last_loss = l
		b2 := beta.subtract(grads[0].multiply(lr))
		beta.free()
		beta = b2
		for v in values {
			v.free()
		}
		for g in grads {
			g.free()
		}
	}
	beta_learned := f64(beta.item_f32())
	// independent reference: exact fixed point m* = (I − (β/3)A)⁻¹(2y/3)
	// (via solve_n), grid-searched for the sample-MSE-optimal β
	mut amat64 := [][]f64{len: n}
	ca := chain_adjacency(n)
	for i in 0 .. n {
		amat64[i] = ca[i * n..(i + 1) * n].clone()
	}
	mut best_beta, mut best_mse := 0.0, 1e300
	for gi in 0 .. 101 {
		b := 0.005 * f64(gi)
		mut sys := [][]f64{len: n}
		for i in 0 .. n {
			mut row := []f64{len: n}
			for j in 0 .. n {
				row[j] = -b / 3.0 * amat64[i][j]
			}
			row[i] += 1.0
			sys[i] = row
		}
		mut rhs := []f64{len: n}
		for i in 0 .. n {
			rhs[i] = 2.0 * y[i] / 3.0
		}
		m := solve_n(sys, rhs)
		mut mse := 0.0
		for i in 0 .. n {
			mse += (m[i] - field[i]) * (m[i] - field[i])
		}
		if mse < best_mse {
			best_mse = mse
			best_beta = b
		}
	}
	println('deq: beta 0.05 -> ${beta_learned:.4f} (grid-optimal ${best_beta:.3f}), loss ${first_loss:.4f} -> ${last_loss:.4f}')
	assert last_loss < first_loss
	assert math.abs(beta_learned - best_beta) < 0.03
}
