module conger

// coupled_factors_test.v — end-to-end validation of cross-kernel EM on the
// kernel-graph skeleton: Gauss-Seidel and Jacobi schedules converge to the
// closed-form MAP, the joint objective is monotone under Gauss-Seidel, flat
// priors expose the scale ambiguity (anchored at the zero-init feedback), and
// the shared-factor fan-in graph matches the 4×4 normal equations.
import math

fn cf_approx(a f64, b f64, tol f64) bool {
	return math.abs(a - b) < tol
}

// joint_objective is the true joint log posterior (up to a constant):
// log p(y | g, l) + log p(g) + log p(l), reconstructed from estimates only
// (no double counting of the shared data term).
fn joint_objective(y f64, g f64, l f64, noise_var f64, mug f64, varg f64, mul f64, varl f64) f64 {
	return diag_gaussian_ll([y], [g + l], [math.log(noise_var)]) +
		diag_gaussian_ll([g], [mug], [math.log(varg)]) +
		diag_gaussian_ll([l], [mul], [math.log(varl)])
}

fn repeat_obs(step_obs map[string][]f64, n int) []map[string][]f64 {
	mut obs := []map[string][]f64{cap: n}
	for _ in 0 .. n {
		obs << step_obs.clone()
	}
	return obs
}

fn test_gauss_seidel_converges_to_map_with_monotone_objective() {
	y := 3.0
	nv := 0.5
	mug, varg := 1.0, 4.0
	mul, varl := 0.5, 1.0
	g := new_coupled_pair_graph(nv, mug, varg, mul, varl, false)
	n_steps := 60
	trace := run_recurrent(g, repeat_obs({
		'f0': [y]
		'f1': [y]
	}, n_steps)) or { panic(err) }
	// Gauss-Seidel = coordinate ascent: joint objective never decreases.
	mut prev := -1e300
	for t in 0 .. n_steps {
		ft := joint_objective(y, trace.output(t, 'f0')[1], trace.output(t, 'f1')[1], nv, mug, varg,
			mul, varl)
		if t > 0 {
			assert ft >= prev - 1e-9, 'objective decreased at step ${t}: ${prev} -> ${ft}'
		}
		prev = ft
	}
	g_ref, l_ref := pair_map(y, nv, mug, varg, mul, varl)
	assert cf_approx(trace.output(n_steps - 1, 'f0')[1], g_ref, 1e-6)
	assert cf_approx(trace.output(n_steps - 1, 'f1')[1], l_ref, 1e-6)
	// sanity: the MAP is a genuine split, not a corner
	assert g_ref > 0.5 && l_ref > 0.2 && cf_approx(g_ref + l_ref, y, 0.5)
}

fn test_jacobi_converges_to_same_fixed_point() {
	y := 3.0
	nv := 0.5
	mug, varg := 1.0, 4.0
	mul, varl := 0.5, 1.0
	g := new_coupled_pair_graph(nv, mug, varg, mul, varl, true)
	n_steps := 150 // Jacobi contracts slower than Gauss-Seidel
	trace := run_recurrent(g, repeat_obs({
		'f0': [y]
		'f1': [y]
	}, n_steps)) or { panic(err) }
	g_ref, l_ref := pair_map(y, nv, mug, varg, mul, varl)
	assert cf_approx(trace.output(n_steps - 1, 'f0')[1], g_ref, 1e-6)
	assert cf_approx(trace.output(n_steps - 1, 'f1')[1], l_ref, 1e-6)
}

fn test_flat_prior_scale_ambiguity_anchors_at_zero_init() {
	// Flat priors: the joint likelihood is invariant under (g+c, l-c), so the
	// iteration converges to the split determined by the zero-initialised
	// feedback (l_{-1} = 0): g absorbs everything, l stays at 0.
	y := 3.0
	inf := math.inf(1)
	g := new_coupled_pair_graph(0.5, 0.0, inf, 0.0, inf, false)
	trace := run_recurrent(g, repeat_obs({
		'f0': [y]
		'f1': [y]
	}, 10)) or { panic(err) }
	assert cf_approx(trace.output(9, 'f0')[1], y, 1e-9)
	assert cf_approx(trace.output(9, 'f1')[1], 0.0, 1e-9)
}

fn test_shared_factor_fan_in_matches_normal_equations() {
	// y_i = g_i + h + ε: three geometry-like factors sharing one
	// illumination-like factor. Reference: exact joint MAP via the 4×4
	// normal equations, independent of the iteration.
	nv := 0.25
	mus := [0.0, 0.0, 0.0]
	vars_ := [4.0, 4.0, 4.0]
	smu, svar := 0.0, 1.0
	ys := [1.9, 2.75, 3.85]
	g := new_shared_two_factor_graph(nv, mus, vars_, smu, svar)
	n_steps := 120
	trace := run_recurrent(g, repeat_obs({
		'g0':     [ys[0]]
		'g1':     [ys[1]]
		'g2':     [ys[2]]
		'shared': ys
	}, n_steps)) or { panic(err) }
	// normal equations for x = [g0, g1, g2, h]
	a := 1.0 / nv
	mut h := [][]f64{len: 4, init: []f64{len: 4}}
	mut b := []f64{len: 4}
	for i in 0 .. 3 {
		h[i][i] += a
		h[i][3] += a
		h[3][i] += a
		h[3][3] += a
		b[i] += a * ys[i]
		b[3] += a * ys[i]
		h[i][i] += 1.0 / vars_[i]
		b[i] += mus[i] / vars_[i]
	}
	h[3][3] += 1.0 / svar
	b[3] += smu / svar
	ref := solve_n(h, b)
	for i in 0 .. 3 {
		assert cf_approx(trace.output(n_steps - 1, 'g${i}')[1], ref[i], 1e-6)
	}
	assert cf_approx(trace.output(n_steps - 1, 'shared')[1], ref[3], 1e-6)
	// the shared estimate pools all three channels: it sits between the
	// prior mean (0) and the raw channel mean (Σy/3 ≈ 2.83) — the geometry
	// priors absorb part of each channel, shrinking h away from the mean
	mean_y := (ys[0] + ys[1] + ys[2]) / 3.0
	h_est := trace.output(n_steps - 1, 'shared')[1]
	assert h_est > 0.3
	assert h_est < mean_y
}

fn test_factor_kernel_message_layout() {
	// slot contract: [objective, estimate]; sibling estimates read from the
	// odd positions of concat(feed, back).
	k := new_additive_factor_kernel(0.0, 4.0, 0.5, 1)
	out := k.step(KernelContext{
		obs:  [3.0]
		feed: [-999.0, 1.0] // sibling's [objective, estimate]: only slot 1 matters
	})
	// est = ((3-1)/0.5 + 0/4) / (1/0.5 + 1/4) = 4 / 2.25 = 16/9
	assert cf_approx(out[1], 16.0 / 9.0, 1e-12)
	assert k.out_dim() == 2
}
