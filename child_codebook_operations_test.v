module conger

// child_codebook_operations_test.v — V port of tests/test_child_codebook_operations.py.

import math

import mlx

fn ccf_test_spec(operation string, name string) ChildTemplateSpec {
	family := if operation == 'layer' { 'layered' } else { 'composite' }
	mut constraints := map[string]MetaValue{}
	constraints['relation'] = operation
	constraints['scale_ratio'] = [0.4, 0.6]
	constraints['part_kinds'] = [1.0]
	constraints['part_hues'] = [2.0]
	if operation == 'layer' {
		constraints['lateral_ratio'] = [-0.1, 0.1]
		constraints['depth_gap'] = [0.7, 0.9]
	} else {
		constraints['period_ratio'] = [0.18, 0.22]
	}
	return ChildTemplateSpec{
		name: name
		family: family
		parent_family: 'composite'
		operation: operation
		constraints: constraints
		complexity: 1.5
		generation: 3
		evidence_count: 2
		residual_mean: 1.0
		score_mean: 2.5
	}
}

fn test_layer_child_codebook_constraints() {
	cb := ccf_build(ccf_test_spec('layer', 'layer_child'))
	assert cb.operation == 'layer'
	assert cb.n_combo() == 3 * 1 * 6 * 1 * 3 * 3
	app := new_inverse_app_cb(InverseConfig{scene_family: 'layered'}, cb)
	assert app.layered_reconstructor() == 'constrained'
	assert cb.lateral_lo == 0.35 && cb.lateral_hi == 0.7
	assert meta_list(cb.lineage.delta, 'lateral_ratio')[0] == 0.35
	assert meta_list(cb.lineage.delta, 'lateral_ratio')[1] == 0.7

	mut rng := new_rng(1)
	vals := ccf_sample_layer(mut rng, cb)
	assert vals[6] / vals[2] >= 0.4 - 1e-6 && vals[6] / vals[2] <= 0.6 + 1e-6
	assert vals[3] - vals[7] >= 0.7 - 1e-6 && vals[3] - vals[7] <= 0.9 + 1e-6
	ex := extent * vals[2] * fx / (cam_z - vals[3]) + extent * vals[6] * fx / (cam_z - vals[7])
	assert vals[4] - vals[0] >= 0.35 * ex - 1e-6 && vals[4] - vals[0] <= 0.7 * ex + 1e-6
}

fn test_lateral_child_codebooks_constraints() {
	ops := ['mirror', 'repeat']
	spacings := [5.0, 7.5]
	for i, op in ops {
		spacing := spacings[i]
		cb := ccf_build(ccf_test_spec(op, '${op}_child'))
		assert cb.operation == op
		assert cb.spacing == spacing
		assert cb.n_combo() == 1 * 1 * 3 * 3
		mut rng := new_rng(2)
		vals := ccf_sample_lateral(mut rng, cb)
		assert vals[4] > vals[0]
		assert vals[7] == vals[3]
		assert vals[6] / vals[2] >= 0.4 - 1e-6 && vals[6] / vals[2] <= 0.6 + 1e-6
	}
}

fn test_delta_learner_feeds_all_child_factories() {
	for op in ['layer', 'mirror', 'repeat'] {
		mut delta := map[string]MetaValue{}
		delta['relation'] = op
		delta['ratio'] = 0.5
		delta['lateral_ratio'] = if op == 'layer' { 0.0 } else { 0.2 }
		delta['part_kind'] = 1
		delta['part_hue'] = 2
		if op == 'layer' {
			delta['depth_gap'] = 0.8
		}
		mut params := []f64{len: 14}
		for i in 0 .. 14 {
			params[i] = f64(i)
		}
		proposal := TemplateProposal{
			family: if op == 'layer' { 'layered' } else { 'composite' }
			operation: op
			params: params
			residual: 1.0
			complexity: 1.5
			score: 2.5
			parent_family: 'composite'
			delta: delta
		}
		req := StructureBirthRequest{
			residual_mean: 1.0
			best_posterior_mean: 0.5
			reason: 'test'
			proposals: [proposal, proposal]
		}
		specs := TemplateDeltaLearner{min_evidence: 2}.tdl_learn([req], map[string]TemplateLineage{})
		assert specs.len == 1
		cb := ccf_build(specs[0])
		assert cb.lineage.operation == op
	}
}

fn test_lateral_geometry_and_frame_features() {
	cb := ccf_build(ccf_test_spec('mirror', 'mirror_geom'))
	cfg := InverseConfig{scene_family: 'composite'}
	app := new_inverse_app_cb(cfg, cb)
	prm := cb.sample(1, 7, false).take_axis(sel1(0), 0).data_f32()
	mut prm_f := []f64{len: 14}
	for i in 0 .. 14 {
		prm_f[i] = f64(prm[i])
	}
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	scene := cb.to_scene(prm_f)
	fl := renderer.render(scene, cam_l)
	fr := renderer.render(scene, cam_r)
	st := lgc_estimate(fl, fr)
	assert st[4] > st[0]
	assert math.abs(st[2] - st[6]) < 0.3
	vec, stats, _ := sr_frame_features(app, fl, fr, none)
	assert vec.dim(1) == cfg.n_feat()
	assert stats.dim(0) == 1 && stats.dim(1) == 8
}
