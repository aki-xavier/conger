module conger

// mrf_kernels_test.v — hand-computed checks of the MRF site kernels, exact
// joint-posterior comparisons for the damped relaxation on small lattices,
// Potts smoothing behaviour, and grid wiring invariants.
import math

const ln2pi = math.log(2.0 * math.pi)

fn approx(a f64, b f64, tol f64) bool {
	return math.abs(a - b) < tol
}

fn gmrf_stationary_obs(obs0 map[string][]f64, steps int) []map[string][]f64 {
	mut obs := []map[string][]f64{cap: steps}
	for _ in 0 .. steps {
		obs << obs0
	}
	return obs
}

fn argmax_f64(x []f64) int {
	mut best := 0
	for i in 1 .. x.len {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

fn test_gmrf_kernel_step_handcomputed() {
	// no neighbours: pure observation/prior fusion
	// mu=1, σ²=2, τ²=1, y=3 → c=1, m=(3+0.5)/1.5=7/3
	k := new_gmrf_kernel(1.0, math.log(2.0), 0.0, [])
	assert k.out_dim() == 2
	out := k.step(KernelContext{
		obs: [3.0]
	})
	assert approx(out[0], 7.0 / 3.0, 1e-12)
	assert approx(out[1], -0.5 * (4.0 / 3.0 + math.log(3.0) + ln2pi), 1e-12)
	// one neighbour at estimate 4 with β=0.5 → c = 1 + 0.5·(4-1) = 2.5
	// m = (3 + 2.5/2)/1.5 = 17/6; predictive ll = log N(3; 2.5, σ²+τ²=3)
	k1 := new_gmrf_kernel(1.0, math.log(2.0), 0.0, [0.5])
	out1 := k1.step(KernelContext{
		obs:  [3.0]
		back: [4.0, -99.0] // neighbour emits [estimate, ll]; only the estimate is read
	})
	assert approx(out1[0], 17.0 / 6.0, 1e-12)
	assert approx(out1[1], -0.5 * (0.25 / 3.0 + math.log(3.0) + ln2pi), 1e-12)
}

fn test_gmrf_two_node_exact_posterior() {
	// symmetric chain of two, β=0.4, σ²=τ²=1, μ=0. The joint field precision
	// is Λ = [[1, -β], [-β, 1]]; with y = x + N(0, I) the posterior mean is
	// (Λ + I)⁻¹ y. The damped relaxation must land on it exactly.
	beta := 0.4
	g := KernelGraph{
		nodes: {
			'a': KernelNode{
				kernel:   new_gmrf_kernel(0.0, 0.0, 0.0, [beta])
				feedback: ['b']
			}
			'b': KernelNode{
				kernel:   new_gmrf_kernel(0.0, 0.0, 0.0, [beta])
				feedback: ['a']
			}
		}
	}
	y := {
		'a': [1.5]
		'b': [-0.5]
	}
	trace := run_recurrent_opts(g, gmrf_stationary_obs(y, 300), RecurrentOptions{
		damping: 0.3
		tol:     1e-12
	}) or { panic(err) }
	assert trace.converged
	want := solve_n([[2.0, -beta], [-beta, 2.0]], [1.5, -0.5])
	last := trace.steps.len - 1
	assert approx(trace.output(last, 'a')[0], want[0], 1e-8)
	assert approx(trace.output(last, 'b')[0], want[1], 1e-8)
}

fn test_gmrf_grid_matches_exact_posterior() {
	// 3×3 lattice, β=0.3, σ²=τ²=1, μ=0. Exact reference: posterior mean
	// (Λ + I)⁻¹ y with Λ = I - β·A over the grid adjacency.
	rows, cols := 3, 3
	beta := 0.3
	nodes := grid4_nodes(rows, cols, fn [beta] (r int, c int, n_neighbors int) LikelihoodKernel {
		return new_gmrf_kernel(0.0, 0.0, 0.0, [beta].repeat(n_neighbors))
	})
	mut rng := new_rng(7)
	mut y := map[string][]f64{}
	mut yvec := []f64{cap: rows * cols}
	for r in 0 .. rows {
		for c in 0 .. cols {
			v := rng.normal(0.0, 1.0)
			y[grid4_name(r, c)] = [v]
			yvec << v
		}
	}
	// posterior precision Λ + I and exact solve
	n := rows * cols
	mut prec := [][]f64{len: n, init: []f64{len: n}}
	for r in 0 .. rows {
		for c in 0 .. cols {
			i := r * cols + c
			prec[i][i] = 2.0 // Λ_ii = 1, plus 1/τ²
			for nb in nodes[grid4_name(r, c)].feedback {
				// recover neighbour index from its name suffix
				parts := nb.split('_')
				j := parts[1].int() * cols + parts[2].int()
				prec[i][j] = -beta
			}
		}
	}
	want := solve_n(prec, yvec)
	g := KernelGraph{
		nodes: nodes
	}
	trace := run_recurrent_opts(g, gmrf_stationary_obs(y, 500), RecurrentOptions{
		damping: 0.4
		tol:     1e-11
	}) or { panic(err) }
	assert trace.converged
	last := trace.steps.len - 1
	for r in 0 .. rows {
		for c in 0 .. cols {
			got := trace.output(last, grid4_name(r, c))[0]
			assert approx(got, want[r * cols + c], 1e-6)
		}
	}
}

fn test_potts_kernel_step_handcomputed() {
	// no coupling: output = obs + log_prior
	k := new_potts_kernel([0.1, -0.2], [])
	out := k.step(KernelContext{
		obs: [0.3, 0.1]
	})
	assert approx(out[0], 0.4, 1e-12)
	assert approx(out[1], -0.1, 1e-12)
	// one neighbour (log-posterior [1, 0]) voting with J=2
	p0 := math.exp(1.0) / (math.exp(1.0) + 1.0)
	k1 := new_potts_kernel([0.0, 0.0], [2.0])
	out1 := k1.step(KernelContext{
		obs:  [0.3, 0.1]
		back: [1.0, 0.0]
	})
	assert approx(out1[0], 0.3 + 2.0 * p0, 1e-12)
	assert approx(out1[1], 0.1 + 2.0 * (1.0 - p0), 1e-12)
}

fn test_potts_two_node_smoothing() {
	// a has moderate evidence for class 0, b weak evidence for class 1; with
	// J=0.8 mutual voting must pull b over to class 0.
	g := KernelGraph{
		nodes: {
			'a': KernelNode{
				kernel:   new_potts_kernel([0.0, 0.0], [0.8])
				feedback: ['b']
			}
			'b': KernelNode{
				kernel:   new_potts_kernel([0.0, 0.0], [0.8])
				feedback: ['a']
			}
		}
	}
	obs0 := {
		'a': [0.6, 0.0]
		'b': [0.0, 0.1]
	}
	trace := run_recurrent_opts(g, gmrf_stationary_obs(obs0, 200), RecurrentOptions{
		damping: 0.3
		tol:     1e-10
	}) or { panic(err) }
	assert trace.converged
	last := trace.steps.len - 1
	oa := trace.output(last, 'a')
	ob := trace.output(last, 'b')
	assert argmax_f64(oa) == 0
	assert argmax_f64(ob) == 0
	// outputs are proper log-posteriors up to a constant: softmax sums to 1
	mut s := 0.0
	lse := logsumexp(ob)
	for v in ob {
		s += math.exp(v - lse)
	}
	assert approx(s, 1.0, 1e-12)
	// decoupled (J=0): b keeps its local evidence → class 1
	k0 := new_potts_kernel([0.0, 0.0], [])
	assert argmax_f64(k0.step(KernelContext{
		obs: [0.0, 0.1]
	})) == 1
}

fn test_grid4_nodes_wiring() {
	nodes := grid4_nodes(3, 3, fn (r int, c int, n_neighbors int) LikelihoodKernel {
		return new_gmrf_kernel(0.0, 0.0, 0.0, [0.3].repeat(n_neighbors))
	})
	assert nodes.len == 9
	// 4-neighbour order: up, down, left, right
	assert nodes['n_1_1'].feedback == ['n_0_1', 'n_2_1', 'n_1_0', 'n_1_2']
	assert nodes['n_0_0'].feedback == ['n_1_0', 'n_0_1']
	assert nodes['n_2_2'].feedback == ['n_1_2', 'n_2_1']
	g := KernelGraph{
		nodes: nodes
	}
	order := topo_order(nodes) or { panic(err) }
	assert order.len == 9
	cyc := feedback_cycle_nodes(g)
	assert cyc.len == 9 // every lattice site is on a feedback cycle
}
