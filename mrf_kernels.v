module conger

// mrf_kernels.v — Markov-random-field likelihood kernels for the kernel-graph
// skeleton (`kernel_graph.v`).
//
// An MRF factorises over a graph of sites; each site's full conditional
// p(x_i | x_∂i) depends only on its Markov blanket (its neighbours). That is
// exactly the kernel-graph contract: a site is a node, its neighbourhood is
// declared with feedback edges (cycles allowed, one step of time lag = one
// parallel relaxation sweep), and `ctx.back` delivers the neighbours'
// previous-sweep outputs. Spatial/contextual coupling therefore lives in the
// *topology*, while the kernel only evaluates the local conditional. Loopy
// convergence is handled by the skeleton: damped fixed-point iteration
// (`run_recurrent_opts`), static cycle inspection (`feedback_cycle_nodes`),
// and residual-driven asynchronous scheduling (`run_residual`, in the spirit
// of residual belief propagation).
//
// Two site kernels are provided:
//
//   - `GMRFKernel` — continuous scalar site of a Gaussian MRF
//     (auto-Gaussian / CAR model) with a noisy local observation. Emits
//     [posterior_mean, predictive_log_likelihood] (out_dim 2); at the damped
//     fixed point the posterior means equal the exact joint posterior mean
//     E[x | y] of the Gaussian field (see `mrf_kernels_test.v`).
//   - `PottsKernel` — discrete C-class site (Potts/Ising). Emits an
//     unnormalised log-posterior vector (out_dim C) fusing local observation
//     log-likelihoods, a class prior, and the neighbours' (softmaxed)
//     previous posteriors — a mean-field/ICM-style parallel update.
//
// Both kernels read neighbour outputs from the concatenated [feed; back]
// input at a fixed stride equal to their own out_dim, so every declared
// parent of an MRF node must be another MRF site kernel of the same kind.
// `grid4_nodes` wires a regular 2-D lattice with 4-neighbourhood feedback
// edges; arbitrary adjacency can be wired by hand with the same convention.
import math

// GMRFKernel is one site of a Gaussian MRF with local noisy observation
// y = x + N(0, τ²). The spatial full conditional is
//
//	p(x_i | x_∂i) = N(μ + Σ_j β_j·(x̂_j − μ), σ²)
//
// and the kernel fuses it with the observation likelihood:
//
//	x̂_i = (y/τ² + c/σ²) / (1/τ² + 1/σ²),   c = μ + Σ_j β_j·(x̂_j − μ)
//
// Output: [x̂_i, log N(y; c, σ² + τ²)] — the fused estimate and the
// predictive (pseudo-)likelihood of the routed observation. `betas` lists one
// coupling per declared neighbour, in parents ++ feedback declaration order;
// neighbour j's estimate is read at position 2j of the concatenated input
// (GMRF out_dim is 2). For a valid joint field the couplings must be
// symmetric (β_ij = β_ji) and the precision matrix positive definite —
// contraction of the relaxation is the practical check, with damping as the
// fallback.
pub struct GMRFKernel {
pub:
	mu          f64
	log_var     f64   // log σ²: spatial conditional variance
	obs_log_var f64   // log τ²: observation noise variance
	betas       []f64 // coupling per declared neighbour (parents ++ feedback order)
}

// new_gmrf_kernel builds a Gaussian-MRF site kernel; `betas` may be empty
// (isolated site — the estimate is then a pure observation/prior fusion).
pub fn new_gmrf_kernel(mu f64, log_var f64, obs_log_var f64, betas []f64) GMRFKernel {
	return GMRFKernel{
		mu:          mu
		log_var:     log_var
		obs_log_var: obs_log_var
		betas:       betas
	}
}

pub fn (k GMRFKernel) out_dim() int {
	return 2
}

pub fn (k GMRFKernel) step(ctx KernelContext) []f64 {
	if ctx.obs.len != 1 {
		panic('GMRFKernel.step: obs width ${ctx.obs.len} != 1 (route the local noisy observation)')
	}
	mut inp := []f64{cap: 2 * k.betas.len}
	inp << ctx.feed
	inp << ctx.back
	if inp.len != 2 * k.betas.len {
		panic('GMRFKernel.step: parent width ${inp.len} != 2·${k.betas.len} (every parent must be a GMRF site, out_dim 2)')
	}
	var2 := math.exp(k.log_var)
	tau2 := math.exp(k.obs_log_var)
	mut c := k.mu
	for j in 0 .. k.betas.len {
		c += k.betas[j] * (inp[2 * j] - k.mu)
	}
	y := ctx.obs[0]
	m := (y / tau2 + c / var2) / (1.0 / tau2 + 1.0 / var2)
	d := y - c
	v := var2 + tau2
	ll := -0.5 * (d * d / v + math.log(v) + math.log(2.0 * math.pi))
	return [m, ll]
}

// PottsKernel is one site of a C-class Potts/Ising MRF. The routed
// observation is the local per-class log-likelihood vector (width C); the
// kernel emits the unnormalised log-posterior
//
//	out[c] = obs[c] + log_prior[c] + Σ_j J_j · softmax(back_j)[c]
//
// where back_j is neighbour j's previous-sweep log-posterior (softmax
// normalised before voting, so a neighbour's confidence scales its vote).
// `couplings` lists one weight J_j per declared neighbour, in parents ++
// feedback declaration order; neighbour j's output is read at positions
// jC .. jC+C of the concatenated input (Potts out_dim is C). Positive J
// encourages agreement (ferromagnetic); damping in `run_recurrent_opts` is
// recommended for strong couplings.
pub struct PottsKernel {
pub:
	log_prior []f64
	couplings []f64 // coupling per declared neighbour (parents ++ feedback order)
}

// new_potts_kernel builds a C-class Potts site kernel.
pub fn new_potts_kernel(log_prior []f64, couplings []f64) PottsKernel {
	if log_prior.len < 2 {
		panic('new_potts_kernel: need at least 2 classes (got ${log_prior.len})')
	}
	return PottsKernel{
		log_prior: log_prior
		couplings: couplings
	}
}

pub fn (k PottsKernel) out_dim() int {
	return k.log_prior.len
}

pub fn (k PottsKernel) step(ctx KernelContext) []f64 {
	n_classes := k.log_prior.len
	if ctx.obs.len != n_classes {
		panic('PottsKernel.step: obs width ${ctx.obs.len} != classes ${n_classes} (route per-class log-likelihoods)')
	}
	mut inp := []f64{cap: n_classes * k.couplings.len}
	inp << ctx.feed
	inp << ctx.back
	if inp.len != n_classes * k.couplings.len {
		panic('PottsKernel.step: parent width ${inp.len} != ${n_classes}·${k.couplings.len} (every parent must be a Potts site, out_dim ${n_classes})')
	}
	mut out := []f64{len: n_classes}
	for c in 0 .. n_classes {
		out[c] = ctx.obs[c] + k.log_prior[c]
	}
	for j in 0 .. k.couplings.len {
		base := j * n_classes
		lse := logsumexp(inp[base..base + n_classes])
		for c in 0 .. n_classes {
			out[c] += k.couplings[j] * math.exp(inp[base + c] - lse)
		}
	}
	return out
}

// grid4_name is the node name `grid4_nodes` assigns to lattice site (r, c);
// use it to route observations.
pub fn grid4_name(r int, c int) string {
	return 'n_${r}_${c}'
}

// grid4_nodes wires a rows×cols lattice of MRF site kernels with
// 4-neighbourhood feedback edges (order: up, down, left, right — border
// sites simply declare fewer neighbours, and `make_kernel` receives the
// neighbour count so per-site coupling vectors can be sized to match).
// Every site participates in a feedback cycle, so `feedback_cycle_nodes`
// returns the full lattice; inference is one damped relaxation per step of
// `run_recurrent_opts` / `run_residual`.
pub fn grid4_nodes(rows int, cols int, make_kernel fn (r int, c int, n_neighbors int) LikelihoodKernel) map[string]KernelNode {
	if rows < 1 || cols < 1 {
		panic('grid4_nodes: rows/cols must be >= 1 (got ${rows}×${cols})')
	}
	mut nodes := map[string]KernelNode{}
	for r in 0 .. rows {
		for c in 0 .. cols {
			mut fb := []string{}
			if r > 0 {
				fb << grid4_name(r - 1, c)
			}
			if r + 1 < rows {
				fb << grid4_name(r + 1, c)
			}
			if c > 0 {
				fb << grid4_name(r, c - 1)
			}
			if c + 1 < cols {
				fb << grid4_name(r, c + 1)
			}
			nodes[grid4_name(r, c)] = KernelNode{
				kernel:   make_kernel(r, c, fb.len)
				feedback: fb
			}
		}
	}
	return nodes
}
