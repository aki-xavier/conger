module conger

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
