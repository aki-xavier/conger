module conger

// likelihood_kernels_test.v — hand-computed checks of the built-in likelihood
// kernels (Gaussian / mixture / conditional Gaussian / MixtureSPN adapter),
// plus a recurrent graph in which a smoothing filter feeds a conditional
// likelihood through the topology.
import math
import mlx

const ln2pi = math.log(2.0 * math.pi)

fn approx(a f64, b f64, tol f64) bool {
	return math.abs(a - b) < tol
}

fn test_gaussian_kernel() {
	k := new_gaussian_kernel([0.0, 0.0], [0.0, 0.0])
	assert k.out_dim() == 1
	ll0 := k.step(KernelContext{
		obs: [0.0, 0.0]
	})
	assert approx(ll0[0], -ln2pi, 1e-12)
	ll1 := k.step(KernelContext{
		obs: [1.0, 0.0]
	})
	assert approx(ll1[0], -0.5 - ln2pi, 1e-12)
	// σ² = 4: log N(3; 1, 4) = -0.5·(4/4 + ln 4 + ln 2π)
	k4 := new_gaussian_kernel([1.0], [math.log(4.0)])
	ll4 := k4.step(KernelContext{
		obs: [3.0]
	})
	assert approx(ll4[0], -0.5 * (1.0 + math.log(4.0) + ln2pi), 1e-12)
}

fn test_gaussian_mixture_kernel() {
	k := new_gaussian_mixture_kernel([math.log(0.5), math.log(0.5)], [
		[-1.0],
		[1.0],
	], [
		[0.0],
		[0.0],
	])
	// log(0.5·N(-1;-1,1) + 0.5·N(-1;1,1))
	want := math.log(0.5 * math.exp(-0.5 * ln2pi) + 0.5 * math.exp(-0.5 * (4.0 + ln2pi)))
	got := k.step(KernelContext{
		obs: [-1.0]
	})
	assert approx(got[0], want, 1e-12)
	// at the midpoint both components contribute equally
	mid := k.step(KernelContext{
		obs: [0.0]
	})
	assert approx(mid[0], -0.5 * (1.0 + ln2pi), 1e-12)
}

fn test_cond_gaussian_kernel_readin() {
	// mean = 1 + 2·feed[0] + 0.5·back[0]
	k := new_cond_gaussian_kernel([[2.0, 0.5]], [1.0], [0.0])
	assert k.out_dim() == 1
	ll := k.step(KernelContext{
		obs:  [9.0]
		feed: [3.0]
		back: [4.0]
	})
	assert approx(ll[0], -0.5 * ln2pi, 1e-12)
	// one unit off the conditional mean
	ll_off := k.step(KernelContext{
		obs:  [10.0]
		feed: [3.0]
		back: [4.0]
	})
	assert approx(ll_off[0], -0.5 * (1.0 + ln2pi), 1e-12)
}

// SmoothKernel: first-order filter m_t = 0.5·m_{t-1} + 0.5·feed (self-feedback).
struct SmoothKernel {}

fn (k SmoothKernel) out_dim() int {
	return 1
}

fn (k SmoothKernel) step(ctx KernelContext) []f64 {
	return [0.5 * ctx.back[0] + 0.5 * ctx.feed[0]]
}

fn test_cond_gaussian_in_recurrent_graph() {
	// s → f (smoothing, self-feedback) → lik (conditional Gaussian on f's mean)
	g := KernelGraph{
		nodes: {
			's':   KernelNode{
				kernel: SourceKernel{
					dim: 1
				}
			}
			'f':   KernelNode{
				kernel:   SmoothKernel{}
				parents:  ['s']
				feedback: ['f']
			}
			'lik': KernelNode{
				kernel:  new_cond_gaussian_kernel([[1.0]], [0.0], [0.0])
				parents: ['f']
			}
		}
	}
	mut obs := []map[string][]f64{}
	for _ in 0 .. 3 {
		obs << {
			's':   [1.0]
			'lik': [1.0]
		}
	}
	trace := run_recurrent(g, obs) or { panic(err) }
	// filter: 0.5, 0.75, 0.875 → residuals 0.5, 0.25, 0.125
	assert approx(trace.output(0, 'f')[0], 0.5, 1e-12)
	assert approx(trace.output(1, 'f')[0], 0.75, 1e-12)
	assert approx(trace.output(2, 'f')[0], 0.875, 1e-12)
	assert approx(trace.output(0, 'lik')[0], -0.5 * (0.25 + ln2pi), 1e-12)
	assert approx(trace.output(1, 'lik')[0], -0.5 * (0.0625 + ln2pi), 1e-12)
	assert approx(trace.output(2, 'lik')[0], -0.5 * (0.015625 + ln2pi), 1e-12)
	// likelihoods increase as the filter locks on
	assert trace.output(2, 'lik')[0] > trace.output(1, 'lik')[0]
	assert trace.output(1, 'lik')[0] > trace.output(0, 'lik')[0]
}

fn tiny_spn(feat_dim int) MixtureSPN {
	// 6 samples, 4-dim features, single stratum (matches fit_simple defaults).
	f := mlx.arr32([
		0.0,
		0.1,
		0.2,
		0.3,
		1.0,
		1.1,
		0.9,
		1.2,
		2.0,
		1.9,
		2.1,
		2.2,
		0.2,
		0.0,
		0.3,
		0.1,
		1.2,
		0.9,
		1.1,
		1.0,
		2.1,
		2.2,
		1.9,
		2.0,
	], [6, feat_dim])
	t := mlx.arr32([0.0, 1.0, 2.0, 0.1, 1.1, 2.1], [6, 1])
	stratum := mlx.zeros([6], .int32)
	return fit_simple(f, t, stratum, 0)
}

fn test_mixture_spn_kernel_matches_direct() {
	net := tiny_spn(4)
	k := new_mixture_spn_kernel(net, 4)
	assert k.out_dim() == 1
	obs := [1.0, 1.1, 0.9, 1.2]
	got := k.step(KernelContext{
		obs: obs
	})
	// direct reference: logsumexp over per-component log joints
	logq := net.logq_feat(net.z(mlx.arr32(obs, [1, 4])))
	want := f64(mlx.axis_logsumexp(logq, 1).item_f32())
	assert approx(got[0], want, 1e-5)
}

fn test_mixture_spn_kernel_in_graph() {
	net := tiny_spn(4)
	g := KernelGraph{
		nodes: {
			'spn': KernelNode{
				kernel: new_mixture_spn_kernel(net, 4)
			}
		}
	}
	trace := run_recurrent(g, [
		{
			'spn': [1.0, 1.1, 0.9, 1.2]
		},
		{
			'spn': [5.0, 5.0, 5.0, 5.0]
		},
	]) or { panic(err) }
	// an in-distribution feature must out-score an out-of-distribution one
	assert trace.output(0, 'spn')[0] > trace.output(1, 'spn')[0]
}
