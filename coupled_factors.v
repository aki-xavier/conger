module conger

// coupled_factors.v — coupled-latent validation domain for cross-kernel EM
// (ECM / mean-field) on the kernel-graph skeleton.
//
// Model (log-domain intrinsic decomposition, the geometry × illumination
// archetype): each observation channel is an additive superposition of two
// latent factors plus Gaussian noise,
//
//   y_c = f_self + f_other,c + ε_c,   ε_c ~ N(0, noise_var)
//   f_self ~ N(prior_mu, prior_var)            (factor prior)
//
// One `AdditiveFactorKernel` plays one factor. Its E-step is the exact
// conditional posterior mean given the sibling factor's *current* estimate:
//
//   est = ( Σ_c (y_c − ê_other,c)/noise_var + prior_mu/prior_var )
//         / ( width/noise_var + 1/prior_var )
//
// and the pair of kernels alternating this update is coordinate ascent on the
// joint posterior — i.e. cross-likelihood EM. Wiring chooses the schedule
// (see kernel_graph.v): a feedback edge reads the sibling's previous iterate
// (Jacobi when both sides use feedback), a feed-forward edge reads the
// sibling's same-step update (Gauss-Seidel when exactly one side does).
//
// Message layout convention: every factor kernel outputs exactly
// [objective, estimate], and reads its siblings' estimates from the odd
// positions of concat(ctx.feed, ctx.back) — sibling c's estimate at index
// 2c+1, with ctx.obs[c] the matching observation channel. A per-observation
// factor (e.g. geometry of sample i) uses width 1; a shared factor (e.g.
// illumination shared across samples) uses width = n_samples and fans in all
// per-observation siblings.
//
// The reported objective (slot 0) is the kernel's own conditional objective
// log p(y | est, siblings) + log p(est); summing slots 0 across kernels
// double-counts the shared data term, so the joint objective should be
// reconstructed from the estimates (slot 1) — see coupled_factors_test.v.
//
// Scale ambiguity: with flat priors (prior_var = +inf) the joint likelihood
// is invariant under (f_self + c, f_other − c) and the iteration anchors at
// the zero-initialised feedback (f_other,₋₁ = 0). Finite priors pull the
// split toward the prior means and make the MAP unique (pair_map).
import math

// AdditiveFactorKernel is one factor of a two-factor additive model.
pub struct AdditiveFactorKernel {
pub:
	prior_mu  f64
	prior_var f64 // +inf = flat prior
	noise_var f64
	width     int // observation channels this factor participates in
}

// new_additive_factor_kernel validates the configuration eagerly.
pub fn new_additive_factor_kernel(prior_mu f64, prior_var f64, noise_var f64, width int) AdditiveFactorKernel {
	// NB: explicit panic, not `assert` — V strips asserts in `-prod` builds.
	if width < 1 {
		panic('new_additive_factor_kernel: width must be >= 1 (got ${width})')
	}
	if noise_var <= 0.0 || math.is_nan(noise_var) {
		panic('new_additive_factor_kernel: noise_var must be > 0 (got ${noise_var})')
	}
	if prior_var <= 0.0 || math.is_nan(prior_var) {
		panic('new_additive_factor_kernel: prior_var must be > 0 (got ${prior_var})')
	}
	return AdditiveFactorKernel{
		prior_mu:  prior_mu
		prior_var: prior_var
		noise_var: noise_var
		width:     width
	}
}

pub fn (k AdditiveFactorKernel) out_dim() int {
	return 2 // [objective, estimate]
}

pub fn (k AdditiveFactorKernel) step(ctx KernelContext) []f64 {
	if ctx.obs.len != k.width {
		panic('AdditiveFactorKernel.step: obs width ${ctx.obs.len} != declared width ${k.width}')
	}
	mut inp := []f64{cap: 2 * k.width}
	inp << ctx.feed
	inp << ctx.back
	if inp.len != 2 * k.width {
		panic('AdditiveFactorKernel.step: sibling input width ${inp.len} != 2·width (${2 * k.width}); each sibling must output [objective, estimate]')
	}
	data_prec := 1.0 / k.noise_var
	prior_prec := 1.0 / k.prior_var // 0 for a flat prior
	mut est := k.prior_mu * prior_prec
	for c in 0 .. k.width {
		est += (ctx.obs[c] - inp[2 * c + 1]) * data_prec
	}
	est /= f64(k.width) * data_prec + prior_prec
	// conditional objective: data term over channels + prior term
	mut mu := []f64{len: k.width}
	mut lv := []f64{len: k.width}
	for c in 0 .. k.width {
		mu[c] = inp[2 * c + 1] + est
		lv[c] = math.log(k.noise_var)
	}
	obj := diag_gaussian_ll(ctx.obs, mu, lv) +
		diag_gaussian_ll([est], [k.prior_mu], [math.log(k.prior_var)])
	return [obj, est]
}

// new_coupled_pair_graph builds the two-factor graph y = f0 + f1 + ε.
// Gauss-Seidel (jacobi = false): f0 reads f1's previous iterate (feedback),
// f1 reads f0's same-step update (feed-forward) — one sweep per step.
// Jacobi (jacobi = true): both read previous iterates via feedback.
pub fn new_coupled_pair_graph(noise_var f64, mu0 f64, var0 f64, mu1 f64, var1 f64, jacobi bool) KernelGraph {
	k0 := new_additive_factor_kernel(mu0, var0, noise_var, 1)
	k1 := new_additive_factor_kernel(mu1, var1, noise_var, 1)
	mut parents1 := []string{}
	mut feedback1 := []string{}
	if jacobi {
		feedback1 = ['f0']
	} else {
		parents1 = ['f0']
	}
	return KernelGraph{
		nodes: {
			'f0': KernelNode{
				kernel:   k0
				feedback: ['f1']
			}
			'f1': KernelNode{
				kernel:   k1
				parents:  parents1
				feedback: feedback1
			}
		}
	}
}

// new_shared_two_factor_graph builds the fan-in graph y_i = g_i + h + ε with
// per-observation factors g_i and one shared factor h (the geometry ×
// shared-illumination archetype). Each g_i reads h's previous iterate via
// feedback; h reads all g_i same-step via feed-forward (Gauss-Seidel sweep).
// Observations: node 'g{i}' takes [y_i], node 'shared' takes all n channels
// in g0..g{n-1} order.
pub fn new_shared_two_factor_graph(noise_var f64, mus []f64, vars []f64, shared_mu f64, shared_var f64) KernelGraph {
	// NB: explicit panic, not `assert` — V strips asserts in `-prod` builds.
	if mus.len == 0 {
		panic('new_shared_two_factor_graph: need at least one observation factor')
	}
	if mus.len != vars.len {
		panic('new_shared_two_factor_graph: mus/vars length mismatch (${mus.len} vs ${vars.len})')
	}
	mut nodes := map[string]KernelNode{}
	mut gnames := []string{cap: mus.len}
	for i in 0 .. mus.len {
		name := 'g${i}'
		gnames << name
		nodes[name] = KernelNode{
			kernel:   new_additive_factor_kernel(mus[i], vars[i], noise_var, 1)
			feedback: ['shared']
		}
	}
	nodes['shared'] = KernelNode{
		kernel:  new_additive_factor_kernel(shared_mu, shared_var, noise_var, mus.len)
		parents: gnames
	}
	return KernelGraph{
		nodes: nodes
	}
}

// pair_map returns the closed-form joint MAP of the two-factor model
// y = f0 + f1 + ε (2×2 normal equations) — the fixed point both schedules
// must converge to.
pub fn pair_map(y f64, noise_var f64, mu0 f64, var0 f64, mu1 f64, var1 f64) (f64, f64) {
	a := 1.0 / noise_var
	b := 1.0 / var0
	c := 1.0 / var1
	sol := solve_2x2([[a + b, a], [a, a + c]], [a * y + b * mu0, a * y + c * mu1])
	return sol[0], sol[1]
}
