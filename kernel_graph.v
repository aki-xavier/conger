module conger

// kernel_graph.v — generic likelihood-kernel network skeleton.
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
import math

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
// step, every node's output vector. `converged` is true when the run stopped
// early because the per-step change fell below the tolerance (see
// RecurrentOptions.tol); it is false for fixed-length runs.
pub struct RecurrentTrace {
pub:
	order     []string
	steps     []map[string][]f64
	converged bool
}

// output returns node `name`'s output vector at step `step` (empty if absent).
pub fn (r RecurrentTrace) output(step int, name string) []f64 {
	if step < 0 || step >= r.steps.len {
		return []f64{}
	}
	return r.steps[step][name] or { []f64{} }
}

// RecurrentOptions tunes a recurrent run (loopy-BP-style machinery).
//
//   - `damping` (0 = off): nodes with feedback edges blend their computed
//     output with their own previous-step output,
//     out = (1-damping)·computed + damping·previous, which suppresses the
//     oscillation that feedback loops can cause. Only feedback-carrying
//     nodes are damped — feed-forward chains and observation sources are
//     exact. Ignored at t = 0 (no previous output).
//   - `tol` (0 = off): early stop. After each step, if the largest absolute
//     change of any node's output versus the previous step is below `tol`,
//     the run stops and the trace is marked `converged`.
pub struct RecurrentOptions {
pub:
	damping f64
	tol     f64
}

// run_recurrent evaluates the graph for `obs.len` steps. `obs[t]` routes
// external observations to nodes by name (typically `SourceKernel` nodes).
// Each step walks the feed-forward topo order once; feedback inputs come from
// the previous step's outputs (zeros at t = 0). Returns an error on unknown
// observation targets, graph mis-wiring (see `topo_order`), or a kernel
// violating its declared `out_dim`.
pub fn run_recurrent(g KernelGraph, obs []map[string][]f64) !RecurrentTrace {
	return run_recurrent_opts(g, obs, RecurrentOptions{})
}

// sweep_once evaluates one synchronous step: feed inputs come from the
// in-sweep (topo-ordered) values, feedback inputs from `prev`, damping blends
// against `prev` exactly like run_recurrent_opts at t >= 1.
fn sweep_once(g KernelGraph, order []string, prev map[string][]f64, obs map[string][]f64, damping f64) map[string][]f64 {
	mut cur := map[string][]f64{}
	for name in order {
		node := g.nodes[name]
		mut feed := []f64{}
		for p in node.parents {
			feed << cur[p]
		}
		mut back := []f64{}
		for f in node.feedback {
			back << prev[f]
		}
		mut out := node.kernel.step(KernelContext{
			t:    1
			obs:  obs[name] or { []f64{} }
			feed: feed
			back: back
		})
		if damping > 0 && node.feedback.len > 0 {
			pv := prev[name]
			for i in 0 .. out.len {
				out[i] = (1.0 - damping) * out[i] + damping * pv[i]
			}
		}
		cur[name] = out
	}
	return cur
}

// feedback_spectral_radius estimates the spectral radius of the Jacobian of
// the one-step synchronous iteration Φ (the map run_recurrent applies each
// step) near the state reached from a cold start, by finite differences +
// power iteration. `damping` folds into Φ exactly as in run_recurrent_opts,
// so callers can compare radii across damping values. Radius < 1 means the
// recurrent iteration is a local contraction (converges); radius >= 1 warns
// of oscillation/divergence *before* running. The Jacobian covers the full
// output state of every node (auxiliary outputs such as log-likelihoods
// included), so for kernels emitting diagnostics alongside estimates the
// radius is a conservative bound. Cost: O(total output dim) sweeps for the
// Jacobian plus the power iterations.
pub fn feedback_spectral_radius(g KernelGraph, obs map[string][]f64, damping f64) !f64 {
	order := topo_order(g.nodes)!
	for name in obs.keys() {
		if name !in g.nodes {
			return error('feedback_spectral_radius: observation routed to unknown node "${name}"')
		}
	}
	// cold-start state: feedback reads zeros, then two sweeps toward the
	// attractor to get a representative operating point
	mut prev := map[string][]f64{}
	for name in order {
		prev[name] = []f64{len: g.nodes[name].kernel.out_dim()}
	}
	mut s := sweep_once(g, order, prev, obs, damping)
	s = sweep_once(g, order, s, obs, damping)
	// index layout for the flattened state vector
	mut idx := map[string]int{}
	mut dim := 0
	for name in order {
		idx[name] = dim
		dim += s[name].len
	}
	base := sweep_once(g, order, s, obs, damping)
	eps := 1e-6
	mut jac := [][]f64{len: dim}
	for i in 0 .. dim {
		jac[i] = []f64{len: dim}
	}
	for name in order {
		for k in 0 .. s[name].len {
			mut sp := s.clone()
			sp[name] = s[name].clone()
			sp[name][k] += eps
			pb := sweep_once(g, order, sp, obs, damping)
			col := idx[name] + k
			for name2 in order {
				for m in 0 .. base[name2].len {
					jac[idx[name2] + m][col] = (pb[name2][m] - base[name2][m]) / eps
				}
			}
		}
	}
	// power iteration (norm ratio converges to the spectral radius even for
	// complex dominant pairs)
	mut rng := new_rng(1)
	mut v := []f64{len: dim}
	for i in 0 .. dim {
		v[i] = rng.normal(0.0, 1.0)
	}
	mut radius := 0.0
	for _ in 0 .. 500 {
		mut w := []f64{len: dim}
		for i in 0 .. dim {
			mut acc := 0.0
			for j in 0 .. dim {
				acc += jac[i][j] * v[j]
			}
			w[i] = acc
		}
		mut nrm := 0.0
		for x in w {
			nrm += x * x
		}
		nrm = math.sqrt(nrm)
		if nrm < 1e-300 {
			return 0.0
		}
		mut vn := []f64{len: dim}
		for i in 0 .. dim {
			vn[i] = w[i] / nrm
		}
		change := math.abs(nrm - radius)
		v = vn.clone()
		radius = nrm
		if change < 1e-10 {
			break
		}
	}
	return radius
}

// run_recurrent_opts is `run_recurrent` with damping / early-stop options.
pub fn run_recurrent_opts(g KernelGraph, obs []map[string][]f64, opts RecurrentOptions) !RecurrentTrace {
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
	mut converged := false
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
			mut out := node.kernel.step(KernelContext{
				t:    t
				obs:  obs[t][name] or { []f64{} }
				feed: feed
				back: back
			})
			want := node.kernel.out_dim()
			if out.len != want {
				return error('run_recurrent: kernel "${name}" produced ${out.len} values, declared out_dim ${want} (step ${t})')
			}
			if opts.damping > 0 && t > 0 && node.feedback.len > 0 {
				pv := prev[name]
				for i in 0 .. out.len {
					out[i] = (1.0 - opts.damping) * out[i] + opts.damping * pv[i]
				}
			}
			cur[name] = out
		}
		steps << cur.clone()
		if opts.tol > 0 && t >= 1 {
			mut max_change := 0.0
			for name in order {
				for i, v in cur[name] {
					d := math.abs(v - prev[name][i])
					if d > max_change {
						max_change = d
					}
				}
			}
			prev = cur.move()
			if max_change < opts.tol {
				converged = true
				break
			}
		} else {
			prev = cur.move()
		}
	}
	return RecurrentTrace{
		order:     order
		steps:     steps
		converged: converged
	}
}

// feedback_cycle_nodes returns the names of nodes that participate in a
// feedback cycle (including self-loops): the nodes whose outputs can, after
// one or more steps of time-lagged feedback, influence themselves. These are
// the nodes where oscillation / non-convergence can occur and where damping
// (RecurrentOptions.damping) applies. Nodes in purely feed-forward positions
// and nodes whose feedback chains dead-end are not listed.
pub fn feedback_cycle_nodes(g KernelGraph) []string {
	// A feedback edge y→x (x reads y's previous-step output) closes a cycle
	// iff y is reachable from x along feed-forward edges (x →ff* y): then x's
	// output influences y within a sweep and y's lagged output influences x.
	// Cycle members are the nodes on any such x →ff* y path, plus pure
	// feedback chains that return to their start (b↔c mutual reads, self-loops).
	mut on_cycle := map[string]bool{}
	for name, node in g.nodes {
		mut seen := map[string]bool{}
		mut queue := node.feedback.clone()
		for queue.len > 0 {
			cur := queue[0]
			queue.delete(0)
			if cur == name {
				on_cycle[name] = true
				break
			}
			if cur in seen {
				continue
			}
			seen[cur] = true
			if cn := g.nodes[cur] {
				queue << cn.feedback
			}
		}
	}
	mut children := map[string][]string{}
	for name, node in g.nodes {
		for p in node.parents {
			children[p] << name
		}
	}
	// reach_from returns all nodes reachable from `start` (inclusive) by
	// following `adj`.
	reach_from := fn (adj map[string][]string, start string) map[string]bool {
		mut seen := map[string]bool{}
		mut queue := [start]
		for queue.len > 0 {
			cur := queue[0]
			queue.delete(0)
			if cur in seen {
				continue
			}
			seen[cur] = true
			queue << adj[cur]
		}
		return seen
	}
	for name, node in g.nodes {
		for y in node.feedback {
			if y == name {
				on_cycle[name] = true
				continue
			}
			desc := reach_from(children, name)
			if y !in desc {
				continue
			}
			// Nodes on ff paths name →* y: descendants of name that also
			// reach y (BFS over parents from y).
			mut parents_of := map[string][]string{}
			for n2, nd2 in g.nodes {
				parents_of[n2] = nd2.parents.clone()
			}
			anc := reach_from(parents_of, y)
			for n2, _ in desc {
				if n2 in anc {
					on_cycle[n2] = true
				}
			}
		}
	}
	mut out := []string{cap: on_cycle.len}
	for name, _ in on_cycle {
		out << name
	}
	out.sort()
	return out
}

// run_residual evaluates the graph on a *static* observation with
// residual-driven asynchronous scheduling (in the spirit of residual belief
// propagation): instead of sweeping every node once per step, each update
// recomputes only the node whose output changed most in the previous round,
// so the nodes driving the iteration move first. Stops when the largest
// residual falls below `opts.tol` (trace marked converged) or after
// `max_updates` single-node updates.
//
// Semantics per single-node update: feed = parents' current values, back =
// feedback parents' current values (in asynchronous mode the one-step delay
// is approximated by the latest available value), then optionally damped
// against the node's own current value (RecurrentOptions.damping).
// SourceKernel nodes are never updated — their output is the routed
// observation. Each trace step records the full state after one node update.
pub fn run_residual(g KernelGraph, obs map[string][]f64, opts RecurrentOptions, max_updates int) !RecurrentTrace {
	order := topo_order(g.nodes)!
	for name in obs.keys() {
		if name !in g.nodes {
			return error('run_residual: observation routed to unknown node "${name}"')
		}
	}
	// initial full sweep in topo order (feedback reads zeros, then latest)
	mut cur := map[string][]f64{}
	for name in order {
		node := g.nodes[name]
		mut feed := []f64{}
		for p in node.parents {
			feed << cur[p] or { []f64{len: g.nodes[p].kernel.out_dim()} }
		}
		mut back := []f64{}
		for f in node.feedback {
			back << cur[f] or { []f64{len: g.nodes[f].kernel.out_dim()} }
		}
		out := node.kernel.step(KernelContext{
			t:    0
			obs:  obs[name] or { []f64{} }
			feed: feed
			back: back
		})
		if out.len != node.kernel.out_dim() {
			return error('run_residual: kernel "${name}" produced ${out.len} values, declared out_dim ${node.kernel.out_dim()}')
		}
		cur[name] = out
	}
	mut steps := []map[string][]f64{cap: max_updates + 1}
	steps << cur.clone()
	mut converged := false
	mut res := residual_map(g, order, cur, obs, opts)
	for _ in 0 .. max_updates {
		// pick the updatable node with the largest residual
		mut best := ''
		mut best_res := 0.0
		for name in order {
			if g.nodes[name].kernel is SourceKernel {
				continue
			}
			if res[name] > best_res || (res[name] == best_res && best == '') {
				best_res = res[name]
				best = name
			}
		}
		if best == '' || best_res < opts.tol {
			converged = true
			break
		}
		node := g.nodes[best]
		mut feed := []f64{}
		for p in node.parents {
			feed << cur[p]
		}
		mut back := []f64{}
		for f in node.feedback {
			back << cur[f]
		}
		mut out := node.kernel.step(KernelContext{
			t:    steps.len
			obs:  obs[best] or { []f64{} }
			feed: feed
			back: back
		})
		if out.len != node.kernel.out_dim() {
			return error('run_residual: kernel "${best}" produced ${out.len} values, declared out_dim ${node.kernel.out_dim()}')
		}
		if opts.damping > 0 {
			for i in 0 .. out.len {
				out[i] = (1.0 - opts.damping) * out[i] + opts.damping * cur[best][i]
			}
		}
		cur[best] = out
		steps << cur.clone()
		res = residual_map(g, order, cur, obs, opts)
	}
	return RecurrentTrace{
		order:     order
		steps:     steps
		converged: converged
	}
}

// residual_map recomputes every updatable node's would-be output change
// (max absolute elementwise difference) against the current state.
fn residual_map(g KernelGraph, order []string, cur map[string][]f64, obs map[string][]f64, opts RecurrentOptions) map[string]f64 {
	mut res := map[string]f64{}
	for name in order {
		node := g.nodes[name]
		if node.kernel is SourceKernel {
			continue
		}
		mut feed := []f64{}
		for p in node.parents {
			feed << cur[p]
		}
		mut back := []f64{}
		for f in node.feedback {
			back << cur[f]
		}
		mut out := node.kernel.step(KernelContext{
			t:    1
			obs:  obs[name] or { []f64{} }
			feed: feed
			back: back
		})
		if opts.damping > 0 {
			for i in 0 .. out.len {
				out[i] = (1.0 - opts.damping) * out[i] + opts.damping * cur[name][i]
			}
		}
		mut r := 0.0
		for i, v in out {
			d := math.abs(v - cur[name][i])
			if d > r {
				r = d
			}
		}
		res[name] = r
	}
	return res
}
