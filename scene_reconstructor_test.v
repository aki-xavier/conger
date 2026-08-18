module conger

// scene_reconstructor_test.v — full cga.Scene reconstructor tests.

import math

import cga

import mlx

fn test_scene_param_decoding() {
	cat_p := mlx.concatenate([
		arr32([0.0, 3.0, 1.0], [1, 3]),
		arr32([1.0, 0.0, 4.0, 0.0, 0.0, 0.0], [1, 6]),
		arr32([1.0, 5.0, 0.0], [1, 3]),
		arr32([0.0, 0.0, 2.0], [1, 3]),
	], 1)
	t := arr32([72.0, 70.0, 0.01, 0.02], [1, 4])
	stats := arr32([3.25, 5.0, 1000.0], [1, 3])
	prm := sr_params(t, cat_p, stats)[0]
	assert prm[0] == 1.0
	assert prm[5] == 2.0 && prm[6] == 1.0 && prm[7] == 2.0
	assert math.abs(prm[1] - 72.0) < 1e-6
	assert math.abs(prm[2] - 70.0) < 1e-6
	assert math.abs(prm[4] - 3.27) < 1e-6
	expected_s := 0.01 + f64(sr_s_proxy(1, stats).item_f32())
	assert math.abs(prm[3] - expected_s) < 1e-6
}

fn test_single_object_s_z_floor_clamp() {
	cat_p := mlx.concatenate([
		arr32([0.0, 3.0, 1.0], [1, 3]),
		arr32([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1, 6]),
		arr32([1.0, 0.0, 0.0], [1, 3]),
		arr32([1.0, 0.0, 0.0], [1, 3]),
	], 1)
	t := arr32([72.0, 70.0, -1.0, 99.0], [1, 4])
	stats := arr32([3.25, 5.0, 0.1], [1, 3])
	prm := sr_params(t, cat_p, stats)[0]
	assert prm[3] >= s_floor
	assert z_min <= prm[4] && prm[4] <= z_max
}

fn test_kind_conditioned_size_proxy() {
	stats := arr32([3.0, 6.0, 1600.0], [1, 3])
	round_s := f64(sr_s_proxy(0, stats).item_f32())
	box_s := f64(sr_s_proxy(2, stats).item_f32())
	q := (40.0 * (cam_z - 3.0) / fx)
	assert math.abs(round_s - q / math.sqrt(math.pi)) < 1e-6
	assert math.abs(box_s - q * 0.5) < 1e-6
}

fn test_scene_reconstruction_contains_light() {
	cb := new_codebook(InverseConfig{})
	scenes := sr_scenes([[1.0, 72.0, 72.0, 0.45, 3.2, 2.0, 1.0, 2.0]], cb)
	scene := scenes[0]
	mut has_dir := false
	mut has_amb := false
	for l in scene.lights {
		if l.kind == .directional {
			has_dir = true
		}
		if l.kind == .ambient {
			has_amb = true
		}
	}
	assert has_dir && has_amb
	assert scene.objects.len > 0
}

fn test_render_residual_recovers_appearance() {
	cb := new_codebook(InverseConfig{})
	gt := [1.0, 72.0, 72.0, 0.45, 3.2, 2.0, 1.0, 2.0]
	wrong := [gt[0], gt[1], gt[2], gt[3], gt[4], 0.0, 0.0, 0.0]
	mut renderer, cam_l, cam_r := sr_rig()
	scene := cb.to_scene(gt)
	fl := renderer.render(scene, cam_l)
	fr := renderer.render(scene, cam_r)
	pred, score, scores := sr_refine_appearance(cb, wrong, fl, fr, mut renderer,
		cam_l, cam_r)
	assert pred == gt
	assert score < 1e-6
	assert scores.dim(0) == 54
}

fn test_topk_structure_refinement_and_marginals() {
	cb := new_codebook(InverseConfig{})
	gt := [2.0, 72.0, 72.0, 0.45, 3.2, 2.0, 1.0, 2.0]
	mut renderer, cam_l, cam_r := sr_rig()
	scene := cb.to_scene(gt)
	fl := renderer.render(scene, cam_l)
	fr := renderer.render(scene, cam_r)
	z_hat, d, area := StereoDepth{}.estimate(fl, fr)
	stats := arr32([z_hat, d, area], [1, 3])
	s_resid := gt[3] - f64(sr_s_proxy(2, stats).item_f32())
	wrong_s := f64(sr_s_proxy(0, stats).item_f32()) + s_resid
	wrong := [0.0, gt[1], gt[2], wrong_s, gt[4], 0.0, 0.0, 0.0]
	kind_p := arr32([0.4, 0.1, 0.5], [3])
	pred, candidates, scores, posterior, temperature := sr_refine_scene(cb, wrong,
		kind_p, stats, fl, fr, 2, mut renderer, cam_l, cam_r)
	assert pred[0] == gt[0] && pred[1] == gt[1] && pred[2] == gt[2]
	assert math.abs(pred[3] - gt[3]) < 1e-6
	assert pred[4] == gt[4] && pred[5] == gt[5] && pred[6] == gt[6] && pred[7] == gt[7]
	assert candidates.len == 108
	assert scores.dim(0) == 108 && posterior.dim(0) == 108
	assert temperature > 0.0
	assert math.abs(f64(posterior.sum().item_f32()) - 1.0) < 1e-5
	estimate := StructuredHypothesis{
		scene: cb.to_scene(pred)
		params: pred
		spn_posterior: mlx.zeros([15], .float32)
		candidate_params: candidates
		candidate_scores: scores
		candidate_posterior: posterior
		candidate_temperature: temperature
		factor_sizes: [3, 6, 3, 3]
		factor_indices: [0, 5, 6, 7]
	}
	marg := estimate.factor_marginals()
	assert f64(marg[0].data_f32()[2]) > 0.9
	assert f64(marg[1].data_f32()[2]) > 0.9
	assert f64(marg[2].data_f32()[1]) > 0.9
	assert f64(marg[3].data_f32()[2]) > 0.9
}

fn test_novelty_metrics_contract() {
	mut onehot := []f64{len: 15}
	onehot[0] = 1.0
	onehot[3] = 1.0
	onehot[9] = 1.0
	onehot[12] = 1.0
	cat := arr32(onehot, [15])
	r0 := arr32([1.0, 0.0, 0.0], [1, 3])
	rn0, ent0, nov0 := sr_novelty_metrics(cat, r0, none)
	assert rn0 < 1e-6 && ent0 < 1e-6 && nov0 < 1e-6
	r1 := mlx.full_value([1, 4], 0.25, .float32)
	_, ent1, nov1 := sr_novelty_metrics(cat, r1, 9.0)
	assert ent1 == 0.0
	assert nov1 > nov0
}

fn test_evaluator_full_scene_contract() {
	p_gt := arr32([0.0, 10.0, 20.0, 0.4, 3.0, 1.0, 0.0, 2.0, 2.0, 30.0, 40.0,
		0.5, 3.5, 5.0, 2.0, 1.0], [2, 8])
	t_pred := p_gt.take_axis(mlx.arange(1.0, 5.0, 1.0, .int32), 1)
	scene_pred := [[0.0, 10.0, 20.0, 0.4, 3.0, 1.0, 0.0, 2.0], [2.0, 30.0, 40.0,
		0.5, 3.5, 5.0, 2.0, 1.0]]
	out := Evaluator{}.report('合成', p_gt, t_pred, scene_pred, p_gt)
	assert out['kind'] == 1.0
	assert out['hue'] == 1.0
	assert out['lcol'] == 1.0
	assert out['ldir'] == 1.0
	assert out['u_rmse'] == 0.0
	assert out['z_r2'] == 1.0
}
