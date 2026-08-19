module conger

// mrf_learning.v — parameter learning for the Gaussian-MRF kernel
// (`GMRFKernel`, mrf_kernels.v) via the generic EM loop (`generic_em.v`).
//
// The two previously disjoint subsystems meet here: the E step *is* a
// kernel-network run — the lattice of GMRF site kernels is relaxed with
// `run_recurrent_opts` and the converged per-site posterior means serve as
// the (mean-field) posterior over the latent field. The M step is a Besag
// pseudo-likelihood fit on those means, which for the auto-Gaussian model is
// closed-form least squares: with v_i = Σ_{j∈∂i}(m_j − μ),
//
//	μ  = mean(m)
//	β  = Σ (m_i − μ)·v_i / Σ v_i²          (OLS through the origin)
//	σ² solves s = R + s·τ²/(s+τ²), R = mean( (m_i − μ − β·v_i)² ) — the
//	mean-field posterior-variance correction, closed form (see maximize)
//
// Scope cuts (documented, deliberate):
//
//   - The coupling β is homogeneous (one scalar shared by every edge), so the
//     M step stays a one-regressor OLS; per-edge couplings would need per-edge
//     regressors.
//   - The observation noise τ² is held fixed (a learner field, not a fitted
//     param): its closed-form update needs posterior *variances*, which the
//     mean-only relaxation does not provide.
//   - The M step maximises the pseudo-likelihood of the smoothed field, not
//     the exact observed-data likelihood, so the monitored trajectory is not
//     guaranteed monotone; the convergence check uses the deterministic
//     observation pseudo-likelihood PL(y; θ) = Σ_i log N(y_i; c_i(y_∂), σ²+τ²).
//
// β is clamped to keep the fitted model a *proper* GMRF (prior precision
// diagonally dominant: Σ_j |β_ij| < 1, hence |β| < 0.95/4 per edge).
import math

// GMRFLearner is the EMLoop model for GMRF kernel parameters. The observation
// is the flattened (row-major) lattice of noisy observations y.
pub struct GMRFLearner {
pub:
	rows        int
	cols        int
	obs_log_var f64 // log τ²: observation noise, fixed (not fitted)
	relax_damp  f64 = 0.4 // damping of the inner relaxation (E step)
	inner_steps int = 300 // max relaxation sweeps per E step
	inner_tol   f64 = 1e-9
}

// new_gmrf_learner builds a learner for a rows×cols lattice with known
// observation noise τ² (log scale).
pub fn new_gmrf_learner(rows int, cols int, obs_log_var f64) GMRFLearner {
	if rows < 1 || cols < 1 {
		panic('new_gmrf_learner: rows/cols must be >= 1 (got ${rows}×${cols})')
	}
	return GMRFLearner{
		rows:        rows
		cols:        cols
		obs_log_var: obs_log_var
	}
}

// GMRFMeans is the EMLoop responsibility type: the converged per-site
// posterior means of the latent field (row-major).
pub struct GMRFMeans {
pub:
	means []f64
}

// grid4_nb_idx returns the flat (row-major) indices of the 4-neighbours of
// site i, in up/down/left/right order (matching grid4_nodes).
fn grid4_nb_idx(i int, rows int, cols int) []int {
	r, c := i / cols, i % cols
	mut nb := []int{}
	if r > 0 {
		nb << i - cols
	}
	if r + 1 < rows {
		nb << i + cols
	}
	if c > 0 {
		nb << i - 1
	}
	if c + 1 < cols {
		nb << i + 1
	}
	return nb
}

// gmrf_beta_clamp bounds |β| so the fitted model stays a *proper* GMRF:
// diagonal dominance of the prior precision requires Σ_j |β_ij| < 1, hence
// |β| < 1/4 per edge on a max-degree-4 lattice (0.95 safety factor). The
// looser posterior-properness bound (1 + σ²/τ²)/4 is also applied, but the
// prior-PD bound is what stops the EM spiral (bigger σ² → looser clamp →
// bigger β → bigger σ² → divergence).
fn gmrf_beta_clamp(beta f64, var2 f64, tau2 f64) f64 {
	bmax := math.min(0.95 * (1.0 + var2 / tau2), 0.95) / 4.0
	return math.min(bmax, math.max(-bmax, beta))
}

// responsibilities runs the E step: relax the GMRF kernel network at the
// current params and return the converged posterior means. `temperature` is
// accepted for the EMLoop contract but unused — sharpening does not apply to
// a Gaussian mean posterior.
pub fn (l GMRFLearner) responsibilities(params []f64, observation []f64, _temperature f64) GMRFMeans {
	if observation.len != l.rows * l.cols {
		panic('GMRFLearner.responsibilities: observation width ${observation.len} != lattice ${l.rows * l.cols}')
	}
	mu, beta, log_var := params[0], params[1], params[2]
	nodes := grid4_nodes(l.rows, l.cols, fn [mu, beta, log_var, l] (r int, c int, n_neighbors int) LikelihoodKernel {
		return new_gmrf_kernel(mu, log_var, l.obs_log_var, [beta].repeat(n_neighbors))
	})
	mut obs0 := map[string][]f64{}
	for r in 0 .. l.rows {
		for c in 0 .. l.cols {
			obs0[grid4_name(r, c)] = [observation[r * l.cols + c]]
		}
	}
	mut obs := []map[string][]f64{cap: l.inner_steps}
	for _ in 0 .. l.inner_steps {
		obs << obs0
	}
	trace := run_recurrent_opts(KernelGraph{
		nodes: nodes
	}, obs, RecurrentOptions{
		damping: l.relax_damp
		tol:     l.inner_tol
	}) or { panic(err) }
	last := trace.steps.len - 1
	mut means := []f64{cap: l.rows * l.cols}
	for r in 0 .. l.rows {
		for c in 0 .. l.cols {
			means << trace.output(last, grid4_name(r, c))[0]
		}
	}
	return GMRFMeans{
		means: means
	}
}

// maximize runs the M step: closed-form pseudo-likelihood least squares on
// the posterior means, blended toward the old params by `damping`.
pub fn (l GMRFLearner) maximize(resp GMRFMeans, observation []f64, params []f64, damping f64) []f64 {
	m := resp.means
	n := m.len
	mut mu := 0.0
	for v in m {
		mu += v
	}
	mu /= f64(n)
	mut num := 0.0
	mut den := 0.0
	for i in 0 .. n {
		mut vi := 0.0
		for j in grid4_nb_idx(i, l.rows, l.cols) {
			vi += m[j] - mu
		}
		num += (m[i] - mu) * vi
		den += vi * vi
	}
	var2 := math.exp(params[2])
	tau2 := math.exp(l.obs_log_var)
	mut beta := params[1]
	if den > 1e-12 {
		beta = gmrf_beta_clamp(num / den, var2, tau2)
	}
	mut rss := 0.0
	for i in 0 .. n {
		mut vi := 0.0
		for j in grid4_nb_idx(i, l.rows, l.cols) {
			vi += m[j] - mu
		}
		d := m[i] - mu - beta * vi
		rss += d * d
	}
	// Mean-field variance correction: the smoothed residual (m_i − c_i)²
	// systematically under-estimates E[(x_i − c_i)² | y] by the per-site
	// posterior variance σ_p² = σ²τ²/(σ²+τ²) (without it σ² collapses to 0).
	// Adding it makes the update implicit — s = R + s·τ²/(s+τ²) — which
	// solves in closed form: s = (R + √(R² + 4Rτ²)) / 2.
	r := rss / f64(n)
	mut sig2 := (r + math.sqrt(r * r + 4.0 * r * tau2)) / 2.0
	sig2 = math.max(sig2, 1e-12)
	return [
		(1.0 - damping) * mu + damping * params[0],
		(1.0 - damping) * beta + damping * params[1],
		(1.0 - damping) * math.log(sig2) + damping * params[2],
	]
}

// log_likelihood is the deterministic observation pseudo-likelihood
// PL(y; θ) = Σ_i log N(y_i; c_i(y_∂), σ²+τ²) used for convergence monitoring.
pub fn (l GMRFLearner) log_likelihood(params []f64, observation []f64) f64 {
	mu, beta := params[0], params[1]
	v := math.exp(params[2]) + math.exp(l.obs_log_var)
	mut ll := 0.0
	for i, y in observation {
		mut c := mu
		for j in grid4_nb_idx(i, l.rows, l.cols) {
			c += beta * (observation[j] - mu)
		}
		d := y - c
		ll += -0.5 * (d * d / v + math.log(v) + math.log(2.0 * math.pi))
	}
	return ll
}
