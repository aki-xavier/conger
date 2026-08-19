module conger

// graph_structure_test.v — greedy topology birth/prune for GMRF kernel
// lattices: pruning an over-connected (8-neighbourhood) start must recover
// the true 4-neighbourhood, and birth from an under-connected
// (horizontal-only) start must grow the missing vertical edges, both with
// improving pseudo-likelihood.
import math

fn approx(a f64, b f64, tol f64) bool {
	return math.abs(a - b) < tol
}

fn gibbs_field(rows int, cols int, mu f64, beta f64, sig2 f64, sweeps int, seed u64) []f64 {
	mut rng := new_rng(seed)
	n := rows * cols
	mut x := []f64{cap: n}
	for _ in 0 .. n {
		x << rng.normal(mu, 1.0)
	}
	sd := math.sqrt(sig2)
	for _ in 0 .. sweeps {
		for i in 0 .. n {
			mut c := mu
			for j in grid4_nb_idx(i, rows, cols) {
				c += beta * (x[j] - mu)
			}
			x[i] = c + rng.normal(0.0, sd)
		}
	}
	return x
}

fn noisify(x []f64, tau2 f64, seed u64) []f64 {
	mut rng := new_rng(seed)
	mut y := []f64{cap: x.len}
	for v in x {
		y << v + rng.normal(0.0, math.sqrt(tau2))
	}
	return y
}

// count diagonal edges (|dr|=1, |dc|=1) and vertical edges (|dr|=1, dc=0)
fn count_topo(t GMRFTopo, want_dr int, want_dc int) int {
	mut n := 0
	for e in t.edges() {
		dr := math.abs(e.i / t.cols - e.j / t.cols)
		dc := math.abs(e.i % t.cols - e.j % t.cols)
		if dr == want_dr && dc == want_dc {
			n++
		}
	}
	return n
}

fn test_topo_prune_recovers_4nb() {
	rows, cols := 12, 12
	tau2 := 0.25
	field := gibbs_field(rows, cols, 0.0, 0.2, 0.5, 400, 5)
	y := noisify(field, tau2, 6)
	init := gmrf_topo8(rows, cols)
	fit0 := gmrf_fit_topo(y, init, math.log(tau2), 25)
	pl0 := fit0.pl(y, init, math.log(tau2))
	s := GMRFTopoSearch{
		obs_log_var: math.log(tau2)
		min_gain:    0.05
		max_rounds:  12
		em_iters:    15
	}
	res := s.run(y, init)
	assert res.pl > pl0
	// diagonal classes (spurious) must all be pruned
	assert count_topo(res.topo, 1, 1) == 0
	// true 4-nb edges must be fully retained
	n4 := count_topo(res.topo, 1, 0) + count_topo(res.topo, 0, 1)
	assert n4 == 2 * rows * cols - rows - cols
	println('prune: edges ${init.edge_count()} -> ${res.topo.edge_count()}, PL ${pl0:.2f} -> ${res.pl:.2f}')
}

fn test_topo_birth_grows_vertical_edges() {
	rows, cols := 12, 12
	tau2 := 0.25
	field := gibbs_field(rows, cols, 0.0, 0.2, 0.5, 400, 5)
	y := noisify(field, tau2, 6)
	init := gmrf_topo_horizontal(rows, cols)
	fit0 := gmrf_fit_topo(y, init, math.log(tau2), 25)
	pl0 := fit0.pl(y, init, math.log(tau2))
	s := GMRFTopoSearch{
		obs_log_var: math.log(tau2)
		min_gain:    0.05
		max_rounds:  12
		em_iters:    15
	}
	res := s.run(y, init)
	assert res.pl > pl0
	// the whole vertical class must be born
	assert count_topo(res.topo, 1, 0) == (rows - 1) * cols
	// horizontal edges retained
	assert count_topo(res.topo, 0, 1) == rows * (cols - 1)
	// no spurious (diagonal or distance-2) classes born
	spurious := res.topo.edge_count() - count_topo(res.topo, 1, 0) - count_topo(res.topo, 0, 1)
	assert spurious == 0
	println('birth: edges ${init.edge_count()} -> ${res.topo.edge_count()}, PL ${pl0:.2f} -> ${res.pl:.2f}')
}

fn test_topo_helpers() {
	t4 := gmrf_topo4(3, 3)
	assert t4.edge_count() == 12
	assert t4.has_edge(0, 1) && t4.has_edge(0, 3) && !t4.has_edge(0, 4)
	t8 := gmrf_topo8(3, 3)
	assert t8.edge_count() == 12 + 8
	th := gmrf_topo_horizontal(3, 3)
	assert th.edge_count() == 6
	mut t := gmrf_topo_horizontal(3, 3)
	t.add_edge(0, 4)
	assert t.has_edge(0, 4) && t.has_edge(4, 0) && t.edge_count() == 7
	t.add_edge(0, 4) // no-op
	assert t.edge_count() == 7
	t.remove_edge(0, 4)
	assert !t.has_edge(0, 4) && t.edge_count() == 6
}

fn test_topo_fit_recovers_homogeneous_coupling() {
	// homogeneous fit on the true topology must recover β ≈ 0.2
	rows, cols := 12, 12
	tau2 := 0.01
	field := gibbs_field(rows, cols, 0.5, 0.2, 0.5, 400, 9)
	y := noisify(field, tau2, 10)
	topo := gmrf_topo4(rows, cols)
	fit := gmrf_fit_topo(y, topo, math.log(tau2), 40)
	assert approx(fit.mu, 0.5, 0.2)
	assert fit.beta > 0.08 && fit.beta < 0.32
	s2 := math.exp(fit.log_var)
	assert s2 > 0.1 && s2 < 1.5
	println('topo fit: mu=${fit.mu:.3f} beta=${fit.beta:.3f} sig2=${s2:.3f}')
}
