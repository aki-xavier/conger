module conger

// composite_test.v — explicit attached composite template tests (tractable parts).
import math
import mlx

const crc_scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

fn test_composite_sampling_and_scene() {
	p := ccb_sample(1, 123, false)
	assert p.dim(0) == n_combo_layered && p.dim(1) == 14
	v0 := p.take_axis(sel1(2), 1).data_f32()
	v1 := p.take_axis(sel1(8), 1).data_f32()
	for i in 0 .. v0.len {
		assert v0[i] > v1[i]
	}
	s0 := p.take_axis(sel1(3), 1).data_f32()
	s1 := p.take_axis(sel1(9), 1).data_f32()
	z0 := p.take_axis(sel1(4), 1).data_f32()
	z1 := p.take_axis(sel1(10), 1).data_f32()
	for i in 0 .. s0.len {
		r := f64(s1[i]) / f64(s0[i])
		assert r >= 0.35 - 1e-6 && r <= 0.75 + 1e-6
		assert math.abs(f64(z1[i]) - f64(z0[i])) <= 0.060001
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
	scene := new_composite_codebook(InverseConfig{}).to_scene(params)
	assert scene.objects.len == 2 && scene.lights.len == 2
	pe := ccb_sample(1, 124, true)
	assert pe.dim(0) == n_combo_layered && pe.dim(1) == 14
	pe_v0 := pe.take_axis(sel1(2), 1).data_f32()
	pe_v1 := pe.take_axis(sel1(8), 1).data_f32()
	for i in 0 .. pe_v0.len {
		assert pe_v0[i] > pe_v1[i]
	}
}

fn test_composite_decoding_roundtrip() {
	p := ccb_sample(1, 123, false).take_axis(sel1(0), 0)
	t := db_targets(p)
	c := db_scene_classes(p)
	mut onehot := []f64{len: 24}
	cv := c.data_i32()
	mut lo := 0
	for j, sz in lcb_cat_sizes {
		onehot[lo + cv[j]] = 1.0
		lo += sz
	}
	cat_p := arr32(onehot, [1, 24])
	prm := lrc_params_raw(t, cat_p)[0]
	p0 := p.data_f32()
	for i in 0 .. 14 {
		assert math.abs(prm[i] - f64(p0[i])) < 1e-5
	}
	est := StructuredHypothesis{
		scene:            new_composite_codebook(InverseConfig{}).to_scene(prm)
		params:           prm
		spn_posterior:    cat_p.take_axis(sel1(0), 0)
		candidate_params: [prm]
		factor_sizes:     [3, 3, 6, 6, 3, 3]
		factor_indices:   [0, 6, 5, 11, 12, 13]
	}
	assert est.factor_marginals().len == 6
}

fn test_composite_geometry_recovers_parts() {
	cb := new_composite_codebook(InverseConfig{ scene_family: 'composite' })
	p := ccb_sample(1, 123, false)
	mut prm := []f64{len: 14}
	pd := p.data_f32()
	for i in 0 .. 14 {
		prm[i] = f64(pd[i])
	}
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	scene := cb.to_scene(prm)
	fl := renderer.render(scene, cam_l)
	fr := renderer.render(scene, cam_r)
	st := cg_estimate(fl, fr)
	assert math.abs(st[0] - prm[1]) < 8.0
	assert math.abs(st[4] - prm[7]) < 8.0
	assert st[1] > st[5]
	assert st[3] > st[7]
	assert 2.0 < st[2] && st[2] < 4.5
	assert 2.0 < st[6] && st[6] < 4.5
}

fn test_composite_residual_roundtrip() {
	p := ccb_sample(1, 123, false).take_axis(mlx.arange(0.0, 4.0, 1.0, .int32), 0)
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
	rt := lrc_residual_targets(t, c, stats, crc_scale)
	mut onehots := []f64{}
	for i in 0 .. 4 {
		for j, sz in lcb_cat_sizes {
			for k in 0 .. sz {
				if k == cv[i * 6 + j] {
					onehots << 1.0
				} else {
					onehots << 0.0
				}
			}
		}
	}
	cat_p := arr32(onehots, [4, 24])
	got := lrc_params(rt, cat_p, stats, crc_scale)
	got_t := lrc_targets_from_params(got)
	gt := t.data_f32()
	for i in 0 .. 4 {
		for j in 0 .. 8 {
			assert math.abs(f64(got_t.data_f32()[i * 8 + j]) - f64(gt[i * 8 + j])) < 1e-5
		}
	}
}
