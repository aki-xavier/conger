module conger

// causal_invariance_test.v — held-out lighting probe + marginalisation tests.
import math
import mlx

fn test_lighting_holdout_partitions_grid() {
	h := lighting_holdout_split(3, 3, 2, 2)
	mut n_in := 0
	mut n_out := 0
	for lc in 0 .. 3 {
		for ld in 0 .. 3 {
			if h.in_support(lc, ld) {
				n_in++
			}
			if h.holdout(lc, ld) {
				n_out++
			}
		}
	}
	assert n_in == 4
	assert n_out == 5
	assert n_in + n_out == 9
}

fn test_marginal_appearance_sums_to_one() {
	rng := mlx.random_key(0)
	logp := mlx.random_normal([54], .float32, 0.0, 1.0, rng)
	posterior := logp.subtract(logp.logsumexp()).exp()
	for factor in ['hue', 'lcol', 'ldir'] {
		m := sr_marginal_appearance(posterior, factor)
		assert math.abs(f64(m.sum().item_f32()) - 1.0) < 1e-5
	}
}

fn test_marginal_appearance_recovers_dominant_hue() {
	mut posterior := mlx.zeros([54], .float32)
	// hue=1, lcol=2, ldir=1 → index 1*9 + 2*3 + 1 = 16
	mut data := posterior.data_f32()
	// rebuild with one-hot
	mut onehot := []f64{len: 54}
	onehot[16] = 1.0
	posterior = arr32(onehot, [54])
	marg_hue := sr_marginal_appearance(posterior, 'hue')
	marg_lcol := sr_marginal_appearance(posterior, 'lcol')
	marg_ldir := sr_marginal_appearance(posterior, 'ldir')
	assert marg_hue.argmax().item_i32() == 1
	assert marg_lcol.argmax().item_i32() == 2
	assert marg_ldir.argmax().item_i32() == 1
}

fn test_invariance_score_is_worst_group() {
	assert invariance_score([1.0, 1.0, 0.7]) == 0.7
	assert invariance_score([]f64{}) == 0.0
}

fn test_summarize_computes_gap_and_invariance() {
	h := lighting_holdout_split(3, 3, 2, 2)
	mut groups := map[string][][2]int{}
	groups['0,0'] = [[0, 0]!, [1, 1]!]
	groups['1,1'] = [[2, 2]!, [3, 3]!]
	groups['2,0'] = [[4, 0]!, [5, 1]!]
	groups['0,2'] = [[0, 3]!, [1, 4]!]
	rep := summarize_invariance(groups, 'hue', h)
	assert rep.in_support_accuracy == 1.0
	assert rep.holdout_accuracy == 0.0
	assert rep.gap == 1.0
	assert rep.invariance_score == 0.0
	assert rep.n_groups == 4
}

fn test_summarize_fully_invariant() {
	h := lighting_holdout_split(3, 3, 2, 2)
	mut groups := map[string][][2]int{}
	groups['0,0'] = [[0, 0]!]
	groups['1,1'] = [[1, 1]!]
	groups['2,2'] = [[2, 2]!]
	rep := summarize_invariance(groups, 'hue', h)
	assert rep.invariance_score == 1.0
	assert rep.gap == 0.0
}

fn test_invariance_probe_render_recovers_hue() {
	holdout := lighting_holdout_split(3, 3, 2, 2)
	cb := new_codebook(InverseConfig{ scene_family: 'single' })
	in_row := [0.0, 72.0, 72.0, 0.45, 3.2, 2.0, 0.0, 0.0]
	out_row := [0.0, 80.0, 70.0, 0.5, 3.0, 4.0, 2.0, 0.0]
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	report := invariance_probe_run(cb, [in_row, out_row], holdout, 'hue', mut renderer, cam_l,
		cam_r)
	assert report.n_groups == 2
	assert report.in_support_accuracy == 1.0
	assert report.holdout_accuracy == 1.0
	assert report.invariance_score == 1.0
	assert report.gap == 0.0
}
