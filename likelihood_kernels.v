module conger

// likelihood_kernels.v — built-in `LikelihoodKernel` implementations for the
// kernel-graph skeleton (`kernel_graph.v`).
//
// All kernels emit a single scalar per step: the log-likelihood of the routed
// observation under their density. Three pure-f64 kernels cover the common
// cases — fixed diagonal Gaussian, diagonal Gaussian mixture, and a
// conditional Gaussian whose mean is a linear read-in of the concatenated
// feed-forward + feedback inputs (so the *topology* of the graph shapes the
// likelihood). `MixtureSPNKernel` adapts the MLX-backed MixtureSPN: its score
// is the whitened-feature mixture log-likelihood (logsumexp over the
// per-component log joints, whose normalising constants already include the
// mixture weights).
//
// Contract: constructors validate shapes eagerly (explicit panic, not
// `assert` — V strips asserts in `-prod` builds); `step` panics if the
// routed observation / parent widths violate the declared shapes.
import math

// diag_gaussian_ll returns log N(obs; mu, exp(log_var) · I).
fn diag_gaussian_ll(obs []f64, mu []f64, log_var []f64) f64 {
	log2pi := math.log(2.0 * math.pi)
	mut ll := 0.0
	for i, o in obs {
		d := o - mu[i]
		lv := log_var[i]
		ll += -0.5 * (d * d * math.exp(-lv) + lv + log2pi)
	}
	return ll
}

// GaussianKernel scores ctx.obs under a fixed diagonal Gaussian.
pub struct GaussianKernel {
pub:
	mu      []f64
	log_var []f64
}

// new_gaussian_kernel builds a diagonal-Gaussian likelihood kernel.
pub fn new_gaussian_kernel(mu []f64, log_var []f64) GaussianKernel {
	if mu.len == 0 {
		panic('new_gaussian_kernel: mu must be non-empty')
	}
	if mu.len != log_var.len {
		panic('new_gaussian_kernel: mu/log_var length mismatch (${mu.len} vs ${log_var.len})')
	}
	return GaussianKernel{
		mu:      mu
		log_var: log_var
	}
}

pub fn (k GaussianKernel) out_dim() int {
	return 1
}

pub fn (k GaussianKernel) step(ctx KernelContext) []f64 {
	if ctx.obs.len != k.mu.len {
		panic('GaussianKernel.step: obs width ${ctx.obs.len} != dim ${k.mu.len}')
	}
	return [diag_gaussian_ll(ctx.obs, k.mu, k.log_var)]
}

// GaussianMixtureKernel scores ctx.obs under a diagonal Gaussian mixture with
// fixed log-weights (normalised or not — logsumexp handles either).
pub struct GaussianMixtureKernel {
pub:
	log_w    []f64
	mus      [][]f64
	log_vars [][]f64
}

// new_gaussian_mixture_kernel builds a K-component diagonal mixture kernel.
pub fn new_gaussian_mixture_kernel(log_w []f64, mus [][]f64, log_vars [][]f64) GaussianMixtureKernel {
	if log_w.len == 0 {
		panic('new_gaussian_mixture_kernel: need at least one component')
	}
	if mus.len != log_w.len || log_vars.len != log_w.len {
		panic('new_gaussian_mixture_kernel: component count mismatch (log_w ${log_w.len}, mus ${mus.len}, log_vars ${log_vars.len})')
	}
	d := mus[0].len
	if d == 0 {
		panic('new_gaussian_mixture_kernel: components must be non-empty')
	}
	for c in 0 .. log_w.len {
		if mus[c].len != d || log_vars[c].len != d {
			panic('new_gaussian_mixture_kernel: component ${c} dim mismatch')
		}
	}
	return GaussianMixtureKernel{
		log_w:    log_w
		mus:      mus
		log_vars: log_vars
	}
}

pub fn (k GaussianMixtureKernel) out_dim() int {
	return 1
}

pub fn (k GaussianMixtureKernel) step(ctx KernelContext) []f64 {
	if ctx.obs.len != k.mus[0].len {
		panic('GaussianMixtureKernel.step: obs width ${ctx.obs.len} != dim ${k.mus[0].len}')
	}
	mut terms := []f64{len: k.log_w.len}
	for c in 0 .. k.log_w.len {
		terms[c] = k.log_w[c] + diag_gaussian_ll(ctx.obs, k.mus[c], k.log_vars[c])
	}
	return [logsumexp(terms)]
}

// CondGaussianKernel scores ctx.obs under a diagonal Gaussian whose mean is a
// linear read-in of the concatenated parent inputs: mean = bias + W · [feed; back].
// This is the kernel that makes graph topology express conditional structure —
// e.g. a likelihood modulated by another kernel's previous-step estimate via a
// feedback edge, or by several parents via feed-forward concatenation.
pub struct CondGaussianKernel {
pub:
	w       [][]f64 // (D, P) read-in, P = feed_dim + back_dim
	bias    []f64   // (D,)
	log_var []f64   // (D,)
	in_dim  int     // P: expected len(ctx.feed) + len(ctx.back)
}

// new_cond_gaussian_kernel builds a conditional diagonal-Gaussian kernel with
// obs dim D = bias.len and total parent width P = w[0].len.
pub fn new_cond_gaussian_kernel(w [][]f64, bias []f64, log_var []f64) CondGaussianKernel {
	d := bias.len
	if d == 0 {
		panic('new_cond_gaussian_kernel: bias must be non-empty')
	}
	if w.len != d || log_var.len != d {
		panic('new_cond_gaussian_kernel: dim mismatch (w ${w.len}, bias ${bias.len}, log_var ${log_var.len})')
	}
	p := w[0].len
	for i, row in w {
		if row.len != p {
			panic('new_cond_gaussian_kernel: w row ${i} width ${row.len} != ${p}')
		}
	}
	return CondGaussianKernel{
		w:       w
		bias:    bias
		log_var: log_var
		in_dim:  p
	}
}

pub fn (k CondGaussianKernel) out_dim() int {
	return 1
}

pub fn (k CondGaussianKernel) step(ctx KernelContext) []f64 {
	if ctx.obs.len != k.bias.len {
		panic('CondGaussianKernel.step: obs width ${ctx.obs.len} != dim ${k.bias.len}')
	}
	mut inp := []f64{cap: k.in_dim}
	inp << ctx.feed
	inp << ctx.back
	if inp.len != k.in_dim {
		panic('CondGaussianKernel.step: parent width ${inp.len} != declared in_dim ${k.in_dim}')
	}
	mut mu := []f64{len: k.bias.len}
	for i in 0 .. k.bias.len {
		mut m := k.bias[i]
		for j, x in inp {
			m += k.w[i][j] * x
		}
		mu[i] = m
	}
	return [diag_gaussian_ll(ctx.obs, mu, k.log_var)]
}

// MixtureSPNKernel adapts a trained MixtureSPN into the kernel graph: the
// routed observation is a raw feature vector (width = the model's feature
// width), and the score is the whitened-feature mixture log-likelihood
// log Σ_k exp(logq_feat(z)_k). Because logq_feat's per-component constant
// folds in the mixture weight, the logsumexp is the proper mixture density of
// the whitened feature.
pub struct MixtureSPNKernel {
pub:
	net      MixtureSPN
	feat_dim int
}

// new_mixture_spn_kernel wraps a trained net; feat_dim is the raw feature
// width expected in ctx.obs.
pub fn new_mixture_spn_kernel(net MixtureSPN, feat_dim int) MixtureSPNKernel {
	if feat_dim < 1 {
		panic('new_mixture_spn_kernel: feat_dim must be >= 1')
	}
	if net.basis == none {
		panic('new_mixture_spn_kernel: net has no whitening basis (untrained?)')
	}
	return MixtureSPNKernel{
		net:      net
		feat_dim: feat_dim
	}
}

pub fn (k MixtureSPNKernel) out_dim() int {
	return 1
}

pub fn (k MixtureSPNKernel) step(ctx KernelContext) []f64 {
	if ctx.obs.len != k.feat_dim {
		panic('MixtureSPNKernel.step: obs width ${ctx.obs.len} != feat_dim ${k.feat_dim}')
	}
	f := arr32(ctx.obs, [1, k.feat_dim])
	logq := k.net.logq_feat(k.net.z(f))
	ll := axis_logsumexp(logq, 1)
	return [f64(ll.item_f32())]
}
