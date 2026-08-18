module conger

// layered_test.v — layered occlusion scene family tests.
import math
import mlx

fn test_layered_sampling_and_scene() {
	cb := new_layered_codebook(InverseConfig{ scene_family: 'layered' })
	p := lcb_sample(1, 123, false)
	assert p.dim(0) == n_combo_layered && p.dim(1) == 14
	z0 := p.take_axis(sel1(4), 1).data_f32()
	z1 := p.take_axis(sel1(10), 1).data_f32()
	for i in 0 .. z0.len {
		assert z0[i] > z1[i]
	}
	combos := p.take_axis(mlx.array_i32([i32(0), i32(6), i32(5), i32(11), i32(12), i32(13)], [
		6,
	]), 1).astype(.int32).data_i32()
	mut seen := map[string]bool{}
	for i in 0 .. combos.len / 6 {
		key := '${combos[i * 6]},${combos[i * 6 + 1]},${combos[i * 6 + 2]},${combos[i * 6 + 3]},${combos[
			i * 6 + 4]},${combos[i * 6 + 5]}'
		seen[key] = true
	}
	assert seen.len == n_combo_layered
	p0 := p.data_f32()
	mut params := []f64{len: 14}
	for i in 0 .. 14 {
		params[i] = f64(p0[i])
	}
	scene := cb.to_scene(params)
	assert scene.objects.len == 2 && scene.lights.len == 2
}

fn test_layered_targets_and_decoding() {
	cb := new_layered_codebook(InverseConfig{ scene_family: 'layered' })
	p := lcb_sample(1, 7, false)
	t := db_targets(p)
	c := db_scene_classes(p)
	assert t.dim(0) == p.dim(0) && t.dim(1) == 8
	assert c.dim(0) == p.dim(0) && c.dim(1) == 6
	// one-hot cat_p from c[0]
	cv := c.data_i32()
	mut onehot := []f64{len: 24}
	mut lo := 0
	for j, sz in lcb_cat_sizes {
		onehot[lo + cv[j]] = 1.0
		lo += sz
	}
	cat_p := arr32(onehot, [1, 24])
	prm := lrc_params_raw(t.take_axis(sel1(0), 0), cat_p)[0]
	p0 := p.data_f32()
	for i in 0 .. 14 {
		assert math.abs(prm[i] - f64(p0[i])) < 1e-5
	}
	est := StructuredHypothesis{
		scene:            cb.to_scene(prm)
		params:           prm
		spn_posterior:    cat_p.take_axis(sel1(0), 0)
		candidate_params: [prm]
		factor_sizes:     [3, 3, 6, 6, 3, 3]
		factor_indices:   [0, 6, 5, 11, 12, 13]
	}
	marginals := est.factor_marginals()
	assert marginals.len == 6
	for m in marginals {
		assert math.abs(f64(m.sum().item_f32()) - 1.0) < 1e-6
	}
}

fn test_layered_residual_roundtrip() {
	cb := new_layered_codebook(InverseConfig{ scene_family: 'layered' })
	p := lcb_sample(1, 11, false).take_axis(mlx.arange(0.0, 4.0, 1.0, .int32), 0)
	t := db_targets(p)
	c := db_scene_classes(p)
	pd := p.data_f32()
	cv := c.data_i32()
	mut stats_flat := []f64{}
	for i in 0 .. 4 {
		mut row := []f64{len: 14}
		mut cls := []int{len: 6}
		for j in 0 .. 14 {
			row[j] = f64(pd[i * 14 + j])
		}
		for j in 0 .. 6 {
			cls[j] = cv[i * 6 + j]
		}
		stats_flat << layered_make_stats(row, cls)
	}
	stats := mlx.array_f32(f32s(stats_flat), [4, 8])
	rt := lrc_residual_targets(t, c, stats, lrc_residual_scale)
	mut onehots := []f64{}
	for i in 0 .. 4 {
		mut lo := 0
		for j, sz in lcb_cat_sizes {
			for k in 0 .. sz {
				if k == cv[i * 6 + j] {
					onehots << 1.0
				} else {
					onehots << 0.0
				}
			}
			lo += sz
		}
	}
	cat_p := arr32(onehots, [4, 24])
	got := lrc_params(rt, cat_p, stats, lrc_residual_scale)
	got_t := lrc_targets_from_params(got)
	gt := t.data_f32()
	for i in 0 .. 4 {
		for j in 0 .. 8 {
			assert math.abs(f64(got_t.data_f32()[i * 8 + j]) - f64(gt[i * 8 + j])) < 1e-5
		}
	}
}
