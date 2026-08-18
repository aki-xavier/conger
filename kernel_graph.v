module conger

// kernel_graph.v — generic likelihood-kernel network skeleton.
//
// A kernel network is a directed graph of user-supplied likelihood kernels
// with two edge kinds:
//
//   - feed-forward edges (`KernelNode.parents`): the parent's *current-step*
//     output is concatenated into the child's input; these edges must form a
//     DAG, evaluated in `topo_order`.
//   - feedback edges (`KernelNode.feedback`): the parent's *previous-step*
//     output is concatenated into the child's input (zeros at t=0). Feedback
//     edges may close cycles — this is how recurrent connections between
//     likelihood kernels are expressed, one step of time lag breaking the
//     algebraic loop.
//
// Each kernel sees a `KernelContext` and emits a fixed-dimension vector per
// step (e.g. a log-likelihood vector, a filtered estimate, responsibilities —
// the skeleton is agnostic). `run_recurrent` walks the topo order once per
// step and records every node's output trace.
//
// Kernels are plain f64-vector computations (like `vecmath.v`), independent
// of MLX and of any domain: wiring a `MixtureSPN`, a `ToySeries` expert or a
// custom Gaussian factor into a graph only requires wrapping its prediction
// in the `LikelihoodKernel` protocol.

// KernelContext is everything a kernel sees at one evaluation: the external
// observation routed to this node, the concatenated current-step outputs of
// its feed-forward parents (in declared `parents` order), the concatenated
// previous-step outputs of its feedback parents (in declared `feedback`
// order, zero-filled at t = 0), and the step index.
pub struct KernelContext {
pub:
	t    int
	obs  []f64
	feed []f64
	back []f64
}

// LikelihoodKernel is the unit of compute in a kernel network. `out_dim`
// declares the fixed output width; `step` maps the context to that many
// values. Implementations must be side-effect free w.r.t. the graph: kernels
// that need persistent internal state should hold reference fields and use
// `ctx.t == 0` to (re)initialise.
pub interface LikelihoodKernel {
	out_dim() int
	step(ctx KernelContext) []f64
}

// KernelNode wires one kernel into the graph: `parents` are feed-forward
// dependencies evaluated earlier in the same step (must be acyclic),
// `feedback` are recurrent dependencies injected from the previous step
// (cycles allowed, including self-feedback).
pub struct KernelNode {
pub:
	kernel   LikelihoodKernel
	parents  []string
	feedback []string
}

// KernelGraph is a named collection of kernel nodes.
pub struct KernelGraph {
pub:
	nodes map[string]KernelNode
}

// SourceKernel is the built-in observation pass-through: a node whose kernel
// echoes the externally routed observation (`ctx.obs`). Use it for the
// graph's input nodes; the out_dim check in `run_recurrent` validates the
// routed observation width each step.
pub struct SourceKernel {
pub:
	dim int
}

pub fn (s SourceKernel) out_dim() int {
	return s.dim
}

pub fn (s SourceKernel) step(ctx KernelContext) []f64 {
	return ctx.obs
}

// topo_order returns the node names in a deterministic evaluation order that
// respects the feed-forward edges (Kahn's algorithm; among the ready nodes
// the lexicographically smallest name is emitted first, so the order is
// stable across runs). Feedback edges are ignored — they carry one step of
// time lag and may form cycles. Returns an error on unknown node references
// or on a cycle among feed-forward edges.
pub fn topo_order(nodes map[string]KernelNode) ![]string {
	for name, node in nodes {
		for p in node.parents {
			if p !in nodes {
				return error('topo_order: node "${name}" has unknown parent "${p}"')
			}
		}
		for f in node.feedback {
			if f !in nodes {
				return error('topo_order: node "${name}" has unknown feedback parent "${f}"')
			}
		}
		if name in node.parents {
			return error('topo_order: node "${name}" is its own feed-forward parent (use feedback for self-recurrence)')
		}
	}
	mut indegree := map[string]int{}
	for name, node in nodes {
		indegree[name] = node.parents.len
	}
	mut children := map[string][]string{}
	for name, node in nodes {
		for p in node.parents {
			children[p] << name
		}
	}
	mut ready := []string{}
	for name, deg in indegree {
		if deg == 0 {
			ready << name
		}
	}
	mut order := []string{cap: nodes.len}
	for ready.len > 0 {
		ready.sort()
		name := ready[0]
		ready.delete(0)
		order << name
		for c in children[name] {
			indegree[c]--
			if indegree[c] == 0 {
				ready << c
			}
		}
	}
	if order.len != nodes.len {
		mut cyclic := []string{}
		for name, deg in indegree {
			if deg > 0 {
				cyclic << name
			}
		}
		cyclic.sort()
		return error('topo_order: feed-forward cycle involving ${cyclic}')
	}
	return order
}

// RecurrentTrace captures one recurrent run: the topo order used and, per
// step, every node's output vector.
pub struct RecurrentTrace {
pub:
	order []string
	steps []map[string][]f64
}

// output returns node `name`'s output vector at step `step` (empty if absent).
pub fn (r RecurrentTrace) output(step int, name string) []f64 {
	if step < 0 || step >= r.steps.len {
		return []f64{}
	}
	return r.steps[step][name] or { []f64{} }
}

// run_recurrent evaluates the graph for `obs.len` steps. `obs[t]` routes
// external observations to nodes by name (typically `SourceKernel` nodes).
// Each step walks the feed-forward topo order once; feedback inputs come from
// the previous step's outputs (zeros at t = 0). Returns an error on unknown
// observation targets, graph mis-wiring (see `topo_order`), or a kernel
// violating its declared `out_dim`.
pub fn run_recurrent(g KernelGraph, obs []map[string][]f64) !RecurrentTrace {
	order := topo_order(g.nodes)!
	for t, step_obs in obs {
		for name in step_obs.keys() {
			if name !in g.nodes {
				return error('run_recurrent: observation routed to unknown node "${name}" (step ${t})')
			}
		}
	}
	mut prev := map[string][]f64{}
	mut steps := []map[string][]f64{cap: obs.len}
	for t in 0 .. obs.len {
		mut cur := map[string][]f64{}
		for name in order {
			node := g.nodes[name]
			mut feed := []f64{}
			for p in node.parents {
				feed << cur[p]
			}
			mut back := []f64{}
			for f in node.feedback {
				if t == 0 {
					back << []f64{len: g.nodes[f].kernel.out_dim()}
				} else {
					back << prev[f]
				}
			}
			out := node.kernel.step(KernelContext{
				t:    t
				obs:  obs[t][name] or { []f64{} }
				feed: feed
				back: back
			})
			want := node.kernel.out_dim()
			if out.len != want {
				return error('run_recurrent: kernel "${name}" produced ${out.len} values, declared out_dim ${want} (step ${t})')
			}
			cur[name] = out
		}
		steps << cur.clone()
		prev = cur.move()
	}
	return RecurrentTrace{
		order: order
		steps: steps
	}
}
