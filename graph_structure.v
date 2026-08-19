module conger

// graph_structure.v — structure learning over the kernel-graph topology
// itself: greedy edge birth / pruning for GMRF kernel lattices, gated by
// likelihood gain (the same proposal → gate → accept philosophy as
// `generic_structure_gate.v` / `structure_birth.v`, specialised to graph
// topology edits).
//
// The parameter fit is *homogeneous* (one shared coupling β for every edge,
// like `mrf_learning.v`): a free per-edge parameterisation has ≈2N couplings
// for N sites, and its pseudo-likelihood regression is ill-posed at lattice
// scale (coefficients blow up with cancellation; L1 soft-thresholding does
// not rescue it either — verified on synthetic fields). Structure decisions
// therefore compare *refit* pseudo-likelihoods, and the unit of a structure
// move is a whole *offset class* (all pairs with the same (|Δr|, |Δc|)
// pattern): per-edge local PL deltas only rank classes — single pairs carry
// one sample's evidence each and are noise-dominated — while a class
// aggregates 100+ pairs, and acceptance is gated by a full EM refit:
//
//	birth (first): add every missing pair of the best aggregate-gain class
//	prune (second): remove every edge of the smallest aggregate-loss class
//
// Both accept iff the refit PL improves by > `min_gain` (strict
// hill-climbing on the predictive PL; see run() for the objective caveat).
// The shared (μ, β, σ²) refit runs through `EMLoop` (E step = kernel-network
// relaxation, M step = pseudo-likelihood OLS).
import math

// GMRFEdge is one undirected lattice edge (i < j, flat row-major indices).
pub struct GMRFEdge {
pub:
	i int
	j int
}

// GMRFTopo is a mutable lattice topology: symmetric per-site neighbour lists
// over flat row-major site indices.
pub struct GMRFTopo {
pub:
	rows int
	cols int
pub mut:
	nbs [][]int
}

fn gmrf_blank_topo(rows int, cols int) GMRFTopo {
	mut nbs := [][]int{len: rows * cols}
	for i in 0 .. rows * cols {
		nbs[i] = []int{}
	}
	return GMRFTopo{
		rows: rows
		cols: cols
		nbs:  nbs
	}
}

// gmrf_topo4 returns the standard 4-neighbourhood lattice.
pub fn gmrf_topo4(rows int, cols int) GMRFTopo {
	mut t := gmrf_blank_topo(rows, cols)
	for i in 0 .. rows * cols {
		for j in grid4_nb_idx(i, rows, cols) {
			if i < j {
				t.add_edge(i, j)
			}
		}
	}
	return t
}

// gmrf_topo8 returns the 8-neighbourhood lattice (4-neighbourhood plus the
// four diagonals) — the standard "over-connected" starting point for
// structure pruning.
pub fn gmrf_topo8(rows int, cols int) GMRFTopo {
	mut t := gmrf_topo4(rows, cols)
	for r in 0 .. rows {
		for c in 0 .. cols {
			i := r * cols + c
			for d in [[1, 1], [1, -1]] {
				r2, c2 := r + d[0], c + d[1]
				if r2 >= 0 && r2 < rows && c2 >= 0 && c2 < cols {
					t.add_edge(i, r2 * cols + c2)
				}
			}
		}
	}
	return t
}

// gmrf_topo_horizontal returns a lattice with only left/right edges — the
// standard "under-connected" starting point for structure birth.
pub fn gmrf_topo_horizontal(rows int, cols int) GMRFTopo {
	mut t := gmrf_blank_topo(rows, cols)
	for r in 0 .. rows {
		for c in 0 .. cols - 1 {
			t.add_edge(r * cols + c, r * cols + c + 1)
		}
	}
	return t
}

// has_edge reports whether sites i and j are neighbours.
pub fn (t GMRFTopo) has_edge(i int, j int) bool {
	for nb in t.nbs[i] {
		if nb == j {
			return true
		}
	}
	return false
}

// add_edge inserts edge (i,j) (no-op if present).
pub fn (mut t GMRFTopo) add_edge(i int, j int) {
	if i == j || t.has_edge(i, j) {
		return
	}
	t.nbs[i] << j
	t.nbs[j] << i
}

// remove_edge deletes edge (i,j) (no-op if absent).
pub fn (mut t GMRFTopo) remove_edge(i int, j int) {
	t.nbs[i] = t.nbs[i].filter(it != j)
	t.nbs[j] = t.nbs[j].filter(it != i)
}

// edge_count returns the number of undirected edges.
pub fn (t GMRFTopo) edge_count() int {
	mut s := 0
	for nb in t.nbs {
		s += nb.len
	}
	return s / 2
}

// clone returns a deep copy (the search keeps backups for reverts).
pub fn (t GMRFTopo) clone() GMRFTopo {
	mut nbs := [][]int{len: t.nbs.len}
	for i in 0 .. t.nbs.len {
		nbs[i] = t.nbs[i].clone()
	}
	return GMRFTopo{
		rows: t.rows
		cols: t.cols
		nbs:  nbs
	}
}

// edges returns the canonical edge list (i < j, sorted by (i, j)).
pub fn (t GMRFTopo) edges() []GMRFEdge {
	mut out := []GMRFEdge{}
	for i in 0 .. t.nbs.len {
		mut js := t.nbs[i].filter(it > i)
		js.sort()
		for j in js {
			out << GMRFEdge{
				i: i
				j: j
			}
		}
	}
	return out
}

// GMRFTopoFit is a homogeneous parameter fit on a topology: one shared
// coupling `beta` for every edge.
pub struct GMRFTopoFit {
pub mut:
	mu      f64
	log_var f64
	beta    f64
}

// edge_key maps an unordered site pair to a canonical map key.
fn edge_key(i int, j int) string {
	return if i < j { '${i}_${j}' } else { '${j}_${i}' }
}

fn gmrf_site_ll(y f64, c f64, v f64) f64 {
	d := y - c
	return -0.5 * (d * d / v + math.log(v) + math.log(2.0 * math.pi))
}

fn gmrf_cond_mean(vals []f64, i int, topo GMRFTopo, beta f64, mu f64) f64 {
	mut c := mu
	for j in topo.nbs[i] {
		c += beta * (vals[j] - mu)
	}
	return c
}

// pl evaluates the observation pseudo-likelihood PL(y; θ) = Σ_i
// log N(y_i; c_i(y_∂), σ²+τ²) of this fit on `topo`.
pub fn (f GMRFTopoFit) pl(obs []f64, topo GMRFTopo, obs_log_var f64) f64 {
	v := math.exp(f.log_var) + math.exp(obs_log_var)
	mut ll := 0.0
	for i, y in obs {
		ll += gmrf_site_ll(y, gmrf_cond_mean(obs, i, topo, f.beta, f.mu), v)
	}
	return ll
}

// GMRFTopoLearner is the EMLoop model behind `gmrf_fit_topo` (homogeneous
// coupling; params layout [mu, log_var, beta]).
struct GMRFTopoLearner {
	topo        GMRFTopo
	obs_log_var f64
}

fn (l GMRFTopoLearner) responsibilities(params []f64, observation []f64, _temperature f64) GMRFMeans {
	n := l.topo.rows * l.topo.cols
	if observation.len != n {
		panic('GMRFTopoLearner.responsibilities: observation width ${observation.len} != lattice ${n}')
	}
	mu, log_var, beta := params[0], params[1], params[2]
	mut nodes := map[string]KernelNode{}
	mut obs0 := map[string][]f64{}
	for i in 0 .. n {
		mut fb := []string{cap: l.topo.nbs[i].len}
		for j in l.topo.nbs[i] {
			fb << 's${j}'
		}
		nodes['s${i}'] = KernelNode{
			kernel:   new_gmrf_kernel(mu, log_var, l.obs_log_var, [beta].repeat(l.topo.nbs[i].len))
			feedback: fb
		}
		obs0['s${i}'] = [observation[i]]
	}
	mut obs := []map[string][]f64{cap: 200}
	for _ in 0 .. 200 {
		obs << obs0
	}
	trace := run_recurrent_opts(KernelGraph{
		nodes: nodes
	}, obs, RecurrentOptions{
		damping: 0.4
		tol:     1e-9
	}) or { panic(err) }
	last := trace.steps.len - 1
	mut means := []f64{cap: n}
	for i in 0 .. n {
		means << trace.output(last, 's${i}')[0]
	}
	return GMRFMeans{
		means: means
	}
}

fn (l GMRFTopoLearner) maximize(resp GMRFMeans, _observation []f64, params []f64, damping f64) []f64 {
	m := resp.means
	n := m.len
	mut mu := 0.0
	for v in m {
		mu += v
	}
	mu /= f64(n)
	var2 := math.exp(params[1])
	tau2 := math.exp(l.obs_log_var)
	mut num := 0.0
	mut den := 0.0
	for i in 0 .. n {
		mut vi := 0.0
		for j in l.topo.nbs[i] {
			vi += m[j] - mu
		}
		num += (m[i] - mu) * vi
		den += vi * vi
	}
	mut beta := params[2]
	if den > 1e-12 {
		beta = gmrf_beta_clamp(num / den, var2, tau2)
	}
	mut rss := 0.0
	for i in 0 .. n {
		d := m[i] - gmrf_cond_mean(m, i, l.topo, beta, mu)
		rss += d * d
	}
	// same mean-field variance correction as GMRFLearner.maximize
	r := rss / f64(n)
	mut sig2 := (r + math.sqrt(r * r + 4.0 * r * tau2)) / 2.0
	sig2 = math.max(sig2, 1e-12)
	return [
		(1.0 - damping) * mu + damping * params[0],
		(1.0 - damping) * math.log(sig2) + damping * params[1],
		(1.0 - damping) * beta + damping * params[2],
	]
}

fn (l GMRFTopoLearner) log_likelihood(params []f64, observation []f64) f64 {
	fit := GMRFTopoFit{
		mu:      params[0]
		log_var: params[1]
		beta:    params[2]
	}
	return fit.pl(observation, l.topo, l.obs_log_var)
}

// gmrf_fit_topo fits the homogeneous (μ, β, σ²) model on `topo` by EM
// (E step = kernel-network relaxation, M step = pseudo-likelihood OLS).
pub fn gmrf_fit_topo(obs []f64, topo GMRFTopo, obs_log_var f64, em_iters int) GMRFTopoFit {
	mut mu := 0.0
	for v in obs {
		mu += v
	}
	mu /= f64(obs.len)
	mut var0 := 0.0
	for v in obs {
		var0 += (v - mu) * (v - mu)
	}
	var0 = math.max(var0 / f64(obs.len), 1e-3)
	learner := GMRFTopoLearner{
		topo:        topo
		obs_log_var: obs_log_var
	}
	mut loop := EMLoop[GMRFTopoLearner, []f64, GMRFMeans]{
		max_iters: em_iters
		tol:       1e-4
		damping:   0.2
		model:     learner
	}
	res := loop.run(obs, [mu, math.log(var0), 0.05])
	return GMRFTopoFit{
		mu:      res.params[0]
		log_var: res.params[1]
		beta:    res.params[2]
	}
}

// GMRFTopoSearch configures the greedy topology search.
pub struct GMRFTopoSearch {
pub:
	obs_log_var f64
	min_gain    f64 = 0.05 // PL gate (nats): a class move is accepted only if the refit PL improves by more
	max_rounds  int = 12
	em_iters    int = 15
}

// GMRFTopoResult captures the search outcome: final topology, final fit,
// its PL, and the edit log (in order).
pub struct GMRFTopoResult {
pub:
	topo  GMRFTopo
	fit   GMRFTopoFit
	pl    f64
	edits []string
}

// edge_removal_loss returns PL(with edge) − PL(without edge) under the
// current fit. Used only to *rank* prune candidates; the acceptance
// decision is the refit PL gate in GMRFTopoSearch.run.
fn edge_removal_loss(obs []f64, topo GMRFTopo, fit GMRFTopoFit, obs_log_var f64, target GMRFEdge) f64 {
	v := math.exp(fit.log_var) + math.exp(obs_log_var)
	mut loss := 0.0
	for site in [target.i, target.j] {
		other := if site == target.i { target.j } else { target.i }
		c_full := gmrf_cond_mean(obs, site, topo, fit.beta, fit.mu)
		c_cut := c_full - fit.beta * (obs[other] - fit.mu)
		loss += gmrf_site_ll(obs[site], c_full, v) - gmrf_site_ll(obs[site], c_cut, v)
	}
	return loss
}

// birth_gain scores one candidate non-edge: the locally optimal single-edge
// coupling from the current residuals and the PL gain it would bring.
fn birth_gain(obs []f64, topo GMRFTopo, fit GMRFTopoFit, obs_log_var f64, i int, j int) (f64, f64) {
	v := math.exp(fit.log_var) + math.exp(obs_log_var)
	ri := obs[i] - gmrf_cond_mean(obs, i, topo, fit.beta, fit.mu)
	rj := obs[j] - gmrf_cond_mean(obs, j, topo, fit.beta, fit.mu)
	di := obs[i] - fit.mu
	dj := obs[j] - fit.mu
	den := dj * dj + di * di
	if den < 1e-12 {
		return 0.0, 0.0
	}
	beta := gmrf_beta_clamp((ri * dj + rj * di) / den, math.exp(fit.log_var), math.exp(obs_log_var))
	gain := gmrf_site_ll(obs[i], obs[i] - ri + beta * dj, v) + gmrf_site_ll(obs[j], obs[j] - rj +
		beta * di, v) - gmrf_site_ll(obs[i], obs[i] - ri, v) - gmrf_site_ll(obs[j], obs[j] - rj, v)
	return beta, gain
}

// offset_class labels a lattice pair by its unordered (|Δrow|, |Δcol|)
// pattern — the unit of structure moves.
fn offset_class(i int, j int, cols int) string {
	dr := math.abs(i / cols - j / cols)
	dc := math.abs(i % cols - j % cols)
	a := math.min(dr, dc)
	b := math.max(dr, dc)
	return '${a}_${b}'
}

// run executes the greedy class-move search from `init`. Structure moves are
// whole offset classes (all pairs with the same (|Δr|, |Δc|) pattern): a
// single pair carries one sample's worth of evidence, so per-edge decisions
// are noise-dominated, but a class aggregates 100+ pairs — cumulative
// evidence makes the decision robust. Each round ranks classes by the sum of
// cheap local PL deltas, then tries moves best-first, accepting only if the
// full EM refit improves PL by > min_gain (strict hill-climbing):
//
//	birth (first): add every missing pair of the best-gain class
//	prune (second): remove every edge of the smallest-loss class
//
// Note the objective is *predictive* PL, not ground-truth recovery; the
// found topology is the best-predicting one in the candidate family.
pub fn (s GMRFTopoSearch) run(obs []f64, init GMRFTopo) GMRFTopoResult {
	mut topo := init
	mut fit := gmrf_fit_topo(obs, topo, s.obs_log_var, s.em_iters)
	mut pl := fit.pl(obs, topo, s.obs_log_var)
	mut edits := []string{}
	for _ in 0 .. s.max_rounds {
		mut changed := false
		// birth attempt: best aggregate-gain offset class first
		mut class_gain := map[string]f64{}
		mut class_pairs := map[string][]GMRFEdge{}
		mut class_order := []string{}
		for i in 0 .. obs.len {
			r, c := i / topo.cols, i % topo.cols
			for dr in -2 .. 3 {
				for dc in -2 .. 3 {
					if dr == 0 && dc == 0 {
						continue
					}
					r2, c2 := r + dr, c + dc
					if r2 < 0 || r2 >= topo.rows || c2 < 0 || c2 >= topo.cols {
						continue
					}
					j := r2 * topo.cols + c2
					if j <= i || topo.has_edge(i, j) {
						continue
					}
					_, g := birth_gain(obs, topo, fit, s.obs_log_var, i, j)
					k := offset_class(i, j, topo.cols)
					if k !in class_pairs {
						class_order << k
					}
					class_gain[k] += g
					class_pairs[k] << GMRFEdge{
						i: i
						j: j
					}
				}
			}
		}
		class_order.sort_with_compare(fn [class_gain] (a &string, b &string) int {
			return if class_gain[*a] > class_gain[*b] {
				-1
			} else if class_gain[*a] < class_gain[*b] {
				1
			} else {
				0
			}
		})
		for k in class_order {
			backup := topo.clone()
			for e in class_pairs[k] {
				topo.add_edge(e.i, e.j)
			}
			fit2 := gmrf_fit_topo(obs, topo, s.obs_log_var, s.em_iters)
			pl2 := fit2.pl(obs, topo, s.obs_log_var)
			if pl2 > pl + s.min_gain {
				edits << 'birth class (${k}) +${class_pairs[k].len} edges'
				fit, pl = fit2, pl2
				changed = true
				break
			}
			topo = backup
		}
		// prune attempt: smallest aggregate-loss offset class first
		mut class_loss := map[string]f64{}
		mut class_edges := map[string][]GMRFEdge{}
		mut prune_order := []string{}
		for e in topo.edges() {
			k := offset_class(e.i, e.j, topo.cols)
			if k !in class_edges {
				prune_order << k
			}
			class_loss[k] += edge_removal_loss(obs, topo, fit, s.obs_log_var, e)
			class_edges[k] << e
		}
		prune_order.sort_with_compare(fn [class_loss] (a &string, b &string) int {
			return if class_loss[*a] < class_loss[*b] {
				-1
			} else if class_loss[*a] > class_loss[*b] {
				1
			} else {
				0
			}
		})
		for k in prune_order {
			backup := topo.clone()
			for e in class_edges[k] {
				topo.remove_edge(e.i, e.j)
			}
			fit2 := gmrf_fit_topo(obs, topo, s.obs_log_var, s.em_iters)
			pl2 := fit2.pl(obs, topo, s.obs_log_var)
			if pl2 > pl + s.min_gain {
				edits << 'prune class (${k}) -${class_edges[k].len} edges'
				fit, pl = fit2, pl2
				changed = true
				break
			}
			topo = backup
		}
		if !changed {
			break
		}
	}
	return GMRFTopoResult{
		topo:  topo
		fit:   fit
		pl:    pl
		edits: edits
	}
}
