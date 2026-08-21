module conger

import math

// kernel_graph_test.v — exercises the likelihood-kernel network skeleton:
// deterministic topo_order (DAG only, feedback ignored), cycle/unknown-ref
// diagnostics, recurrent evaluation with a feedback loop, self-feedback,
// observation routing and the out_dim contract.

// --- toy kernels -------------------------------------------------------------

// GainKernel scales its (single-parent) feed input elementwise.
struct GainKernel {
	dim  int
	gain f64
}

fn (k GainKernel) out_dim() int {
	return k.dim
}

fn (k GainKernel) step(ctx KernelContext) []f64 {
	mut out := []f64{len: k.dim}
	for i in 0 .. k.dim {
		out[i] = k.gain * ctx.feed[i]
	}
	return out
}

// FeedbackAddKernel emits feed[0] + k·back[0] (single ff parent + single fb parent).
struct FeedbackAddKernel {
	k f64
}

fn (k FeedbackAddKernel) out_dim() int {
	return 1
}

fn (k FeedbackAddKernel) step(ctx KernelContext) []f64 {
	return [ctx.feed[0] + k.k * ctx.back[0]]
}

// IntegratorKernel accumulates its observation via self-feedback.
struct IntegratorKernel {}

fn (k IntegratorKernel) out_dim() int {
	return 1
}

fn (k IntegratorKernel) step(ctx KernelContext) []f64 {
	return [ctx.back[0] + ctx.obs[0]]
}

// SwapKernel emits its two feed inputs swapped (probes parent concat order).
struct SwapKernel {}

fn (k SwapKernel) out_dim() int {
	return 2
}

fn (k SwapKernel) step(ctx KernelContext) []f64 {
	return [ctx.feed[1], ctx.feed[0]]
}

// BadKernel declares out_dim 2 but emits 1 value (contract violation probe).
struct BadKernel {}

fn (k BadKernel) out_dim() int {
	return 2
}

fn (k BadKernel) step(ctx KernelContext) []f64 {
	return [0.0]
}

fn source_node(dim int) KernelNode {
	return KernelNode{
		kernel: SourceKernel{
			dim: dim
		}
	}
}

// --- topo_order --------------------------------------------------------------

fn test_topo_order_dag_deterministic() {
	nodes := {
		'c': KernelNode{
			kernel:  GainKernel{
				dim:  2
				gain: 1.0
			}
			parents: ['a', 'b']
		}
		'b': KernelNode{
			kernel:  GainKernel{
				dim:  1
				gain: 1.0
			}
			parents: ['s']
		}
		's': source_node(1)
		'a': KernelNode{
			kernel:  GainKernel{
				dim:  1
				gain: 1.0
			}
			parents: ['s']
		}
	}
	order := topo_order(nodes) or { panic(err) }
	assert order == ['s', 'a', 'b', 'c']
	// deterministic across repeated calls
	assert topo_order(nodes) or { panic(err) } == order
}

fn test_topo_order_ignores_feedback_edges() {
	// b → a is a feedback edge: the ff subgraph stays a DAG s → a → b.
	nodes := {
		's': source_node(1)
		'a': KernelNode{
			kernel:   FeedbackAddKernel{
				k: 0.5
			}
			parents:  ['s']
			feedback: ['b']
		}
		'b': KernelNode{
			kernel:  GainKernel{
				dim:  1
				gain: 2.0
			}
			parents: ['a']
		}
	}
	order := topo_order(nodes) or { panic(err) }
	assert order == ['s', 'a', 'b']
}

fn test_topo_order_feedforward_cycle_is_error() {
	nodes := {
		'a': KernelNode{
			kernel:  GainKernel{
				dim:  1
				gain: 1.0
			}
			parents: ['b']
		}
		'b': KernelNode{
			kernel:  GainKernel{
				dim:  1
				gain: 1.0
			}
			parents: ['a']
		}
	}
	_ := topo_order(nodes) or {
		assert err.msg().contains('cycle')
		return
	}
	assert false, 'expected cycle error'
}

fn test_topo_order_unknown_parent_is_error() {
	nodes := {
		'a': KernelNode{
			kernel:  GainKernel{
				dim:  1
				gain: 1.0
			}
			parents: ['nope']
		}
	}
	_ := topo_order(nodes) or {
		assert err.msg().contains('unknown parent')
		return
	}
	assert false, 'expected unknown-parent error'
}

// --- run_recurrent -----------------------------------------------------------

fn test_run_recurrent_chain() {
	g := KernelGraph{
		nodes: {
			's': source_node(1)
			'g': KernelNode{
				kernel:  GainKernel{
					dim:  1
					gain: 2.0
				}
				parents: ['s']
			}
		}
	}
	trace := run_recurrent(g, [
		{
			's': [3.0]
		},
		{
			's': [4.5]
		},
	]) or { panic(err) }
	assert trace.order == ['s', 'g']
	assert trace.output(0, 's') == [3.0]
	assert trace.output(0, 'g') == [6.0]
	assert trace.output(1, 'g') == [9.0]
	assert trace.output(2, 'g') == []f64{} // out of range
}

fn test_run_recurrent_feedback_loop() {
	// a_t = s_t + 0.5·b_{t-1}; b_t = 2·a_t  →  a: 1, 2, 3, 4 / b: 2, 4, 6, 8
	g := KernelGraph{
		nodes: {
			's': source_node(1)
			'a': KernelNode{
				kernel:   FeedbackAddKernel{
					k: 0.5
				}
				parents:  ['s']
				feedback: ['b']
			}
			'b': KernelNode{
				kernel:  GainKernel{
					dim:  1
					gain: 2.0
				}
				parents: ['a']
			}
		}
	}
	trace := run_recurrent(g, [
		{
			's': [1.0]
		},
		{
			's': [1.0]
		},
		{
			's': [1.0]
		},
		{
			's': [1.0]
		},
	]) or { panic(err) }
	assert trace.output(0, 'a') == [1.0]
	assert trace.output(0, 'b') == [2.0]
	assert trace.output(1, 'a') == [2.0]
	assert trace.output(1, 'b') == [4.0]
	assert trace.output(2, 'a') == [3.0]
	assert trace.output(3, 'b') == [8.0]
}

fn test_run_recurrent_self_feedback_integrator() {
	g := KernelGraph{
		nodes: {
			'i': KernelNode{
				kernel:   IntegratorKernel{}
				feedback: ['i'] // own previous output
			}
		}
	}
	trace := run_recurrent(g, [
		{
			'i': [1.0]
		},
		{
			'i': [1.0]
		},
		{
			'i': [1.0]
		},
	]) or { panic(err) }
	assert trace.output(0, 'i') == [1.0]
	assert trace.output(1, 'i') == [2.0]
	assert trace.output(2, 'i') == [3.0]
}

fn test_run_recurrent_parent_concat_order() {
	// declared parents order (not topo order) drives the feed concatenation.
	g := KernelGraph{
		nodes: {
			'p1': source_node(1)
			'p2': source_node(1)
			'c':  KernelNode{
				kernel:  SwapKernel{}
				parents: ['p2', 'p1']
			}
		}
	}
	trace := run_recurrent(g, [
		{
			'p1': [1.0]
			'p2': [2.0]
		},
	]) or { panic(err) }
	assert trace.output(0, 'c') == [1.0, 2.0] // [feed[1], feed[0]] = [p1, p2]
}

fn test_run_recurrent_out_dim_violation_is_error() {
	g := KernelGraph{
		nodes: {
			'bad': KernelNode{
				kernel: BadKernel{}
			}
		}
	}
	_ := run_recurrent(g, [
		map[string][]f64{},
	]) or {
		assert err.msg().contains('declared out_dim')
		return
	}
	assert false, 'expected out_dim error'
}

fn test_run_recurrent_unknown_obs_target_is_error() {
	g := KernelGraph{
		nodes: {
			's': source_node(1)
		}
	}
	_ := run_recurrent(g, [
		{
			'ghost': [1.0]
		},
	]) or {
		assert err.msg().contains('unknown node')
		return
	}
	assert false, 'expected unknown-observation-target error'
}

fn test_run_recurrent_source_dim_mismatch_is_error() {
	g := KernelGraph{
		nodes: {
			's': source_node(2)
		}
	}
	_ := run_recurrent(g, [
		{
			's': [1.0] // width 1, declared dim 2
		},
	]) or {
		assert err.msg().contains('declared out_dim')
		return
	}
	assert false, 'expected source dim error'
}

// --- RecurrentOptions: damping / tol / residual scheduling -------------------

// AffineBackKernel emits c + k·back[0] (single feedback parent): the minimal
// oscillation/divergence probe. |k| < 1 contracts; k = -1.2 diverges without
// damping but converges with damping 0.5 (effective gain (1-d)·k + d = -0.1).
struct AffineBackKernel {
	c f64
	k f64
}

fn (k AffineBackKernel) out_dim() int {
	return 1
}

fn (k AffineBackKernel) step(ctx KernelContext) []f64 {
	return [k.c + k.k * ctx.back[0]]
}

fn self_loop_graph(k f64) KernelGraph {
	return KernelGraph{
		nodes: {
			'a': KernelNode{
				kernel:   AffineBackKernel{
					c: 1.0
					k: k
				}
				feedback: ['a']
			}
		}
	}
}

fn const_obs(n int) []map[string][]f64 {
	return []map[string][]f64{len: n, init: map[string][]f64{}}
}

fn test_damping_turns_divergence_into_convergence() {
	// k = -1.2: undamped iteration a_{t+1} = 1 - 1.2·a_t diverges
	g := self_loop_graph(-1.2)
	tr_div := run_recurrent_opts(g, const_obs(30), RecurrentOptions{}) or { panic(err) }
	last_div := tr_div.output(29, 'a')[0]
	assert last_div > 100 || last_div < -100
	assert !tr_div.converged
	// damping 0.5: effective gain (1-0.5)·(-1.2) + 0.5 = -0.1 → contracts
	tr_damp := run_recurrent_opts(g, const_obs(30), RecurrentOptions{
		damping: 0.5
		tol:     1e-9
	}) or { panic(err) }
	assert tr_damp.converged
	assert tr_damp.steps.len < 30
	// fixed point of the damped map: a = 0.5·(1 - 1.2a) + 0.5a → a = 0.5/1.1
	last := tr_damp.steps.len - 1
	assert math.abs(tr_damp.output(last, 'a')[0] - 0.5 / 1.1) < 1e-6
}

fn test_tol_early_stop_marks_converged() {
	// a_{t+1} = 1 + 0.5·a_t → a* = 2 (contracts without damping)
	g := self_loop_graph(0.5)
	tr := run_recurrent_opts(g, const_obs(200), RecurrentOptions{
		tol: 1e-9
	}) or { panic(err) }
	assert tr.converged
	assert tr.steps.len < 200
	last := tr.steps.len - 1
	assert math.abs(tr.output(last, 'a')[0] - 2.0) < 1e-6
	// same graph without tol: full length, not marked converged
	tr_full := run_recurrent_opts(g, const_obs(50), RecurrentOptions{}) or { panic(err) }
	assert !tr_full.converged
	assert tr_full.steps.len == 50
}

fn test_feedback_cycle_nodes() {
	g := KernelGraph{
		nodes: {
			'a': KernelNode{
				kernel:   AffineBackKernel{
					c: 1.0
					k: 0.5
				}
				feedback: ['a'] // self-loop
			}
			'b': KernelNode{
				kernel:   AffineBackKernel{
					c: 1.0
					k: 0.5
				}
				feedback: ['c'] // b ↔ c cycle
			}
			'c': KernelNode{
				kernel:   AffineBackKernel{
					c: 1.0
					k: 0.5
				}
				feedback: ['b']
			}
			'd': KernelNode{
				kernel:   AffineBackKernel{
					c: 1.0
					k: 0.5
				}
				feedback: ['a'] // reads the loop but is not part of it
			}
			'e': source_node(1)
		}
	}
	assert feedback_cycle_nodes(g) == ['a', 'b', 'c']
}

fn test_run_residual_converges_on_static_observation() {
	// 'a' self-loop a* = 2; 'g' feed-forward child of source s (2.0 → 4.0)
	g := KernelGraph{
		nodes: {
			'a': KernelNode{
				kernel:   AffineBackKernel{
					c: 1.0
					k: 0.5
				}
				feedback: ['a']
			}
			's': source_node(1)
			'g': KernelNode{
				kernel:  GainKernel{
					dim:  1
					gain: 2.0
				}
				parents: ['s']
			}
		}
	}
	obs := {
		's': [2.0]
	}
	tr := run_residual(g, obs, RecurrentOptions{
		tol: 1e-9
	}, 200) or { panic(err) }
	assert tr.converged
	last := tr.steps.len - 1
	assert math.abs(tr.output(last, 'a')[0] - 2.0) < 1e-6
	assert tr.output(last, 'g')[0] == 4.0 // source passthrough unaffected by scheduling
}

fn test_run_residual_reports_non_convergence() {
	g := self_loop_graph(-1.2)
	obs := map[string][]f64{}
	tr_bad := run_residual(g, obs, RecurrentOptions{
		tol: 1e-9
	}, 50) or { panic(err) }
	assert !tr_bad.converged
	// same graph with damping converges under residual scheduling too
	tr_ok := run_residual(g, obs, RecurrentOptions{
		tol:     1e-9
		damping: 0.5
	}, 200) or { panic(err) }
	assert tr_ok.converged
	last := tr_ok.steps.len - 1
	assert math.abs(tr_ok.output(last, 'a')[0] - 0.5 / 1.1) < 1e-6
}

fn test_run_residual_unknown_obs_target_is_error() {
	g := KernelGraph{
		nodes: {
			's': source_node(1)
		}
	}
	_ := run_residual(g, {
		'ghost': [1.0]
	}, RecurrentOptions{
		tol: 1e-9
	}, 10) or {
		assert err.msg().contains('unknown node')
		return
	}
	assert false, 'expected unknown-observation-target error'
}

// --- feedback_spectral_radius --------------------------------------------------

fn two_node_loop(ka f64, kb f64) KernelGraph {
	return KernelGraph{
		nodes: {
			'a': KernelNode{
				kernel:   AffineBackKernel{
					c: 1.0
					k: ka
				}
				feedback: ['b']
			}
			'b': KernelNode{
				kernel:   AffineBackKernel{
					c: 1.0
					k: kb
				}
				feedback: ['a']
			}
		}
	}
}

fn test_spectral_radius_linear_loop() {
	// J = [[0, ka], [kb, 0]] → eigenvalues ±√(ka·kb)
	r := feedback_spectral_radius(two_node_loop(0.9, 0.9), map[string][]f64{}, 0.0) or {
		panic(err)
	}
	assert math.abs(r - 0.9) < 1e-3
	// ka·kb < 0: complex pair ±i·1.2, |λ| = 1.2 → divergent
	r2 := feedback_spectral_radius(two_node_loop(1.2, -1.2), map[string][]f64{}, 0.0) or {
		panic(err)
	}
	assert math.abs(r2 - 1.2) < 1e-3
	// damping 0.5: |0.5 ± 0.6i| = √0.61 ≈ 0.781 → contraction
	r3 := feedback_spectral_radius(two_node_loop(1.2, -1.2), map[string][]f64{}, 0.5) or {
		panic(err)
	}
	assert math.abs(r3 - math.sqrt(0.61)) < 1e-3
}

fn test_spectral_radius_self_loop_matches_damping_rule() {
	// k = -1.2 self-loop: undamped |λ| = 1.2; damping 0.5 → |(1-d)k + d| = 0.1
	r_div := feedback_spectral_radius(self_loop_graph(-1.2), map[string][]f64{}, 0.0) or {
		panic(err)
	}
	assert r_div > 1.0
	r_ok := feedback_spectral_radius(self_loop_graph(-1.2), map[string][]f64{}, 0.5) or {
		panic(err)
	}
	assert math.abs(r_ok - 0.1) < 1e-3
}
