module conger

// template_delta_learner_test.v — V port of tests/test_template_delta_learner.py.
import math
import os

fn tdl_parent_expert() SceneExpert {
	return SceneExpert{
		name: 'layered'
		app:  new_inverse_app(InverseConfig{ scene_family: 'layered' })
	}
}

fn tdl_test_proposal(ratio f64, lateral f64) TemplateProposal {
	mut delta := map[string]MetaValue{}
	delta['relation'] = 'attach'
	delta['ratio'] = ratio
	delta['lateral_ratio'] = lateral
	delta['part_kind'] = 1
	delta['part_hue'] = 2
	mut params := []f64{len: 14}
	for i in 0 .. 14 {
		params[i] = f64(i)
	}
	return TemplateProposal{
		family:        'composite'
		operation:     'attach'
		params:        params
		residual:      10.0 + ratio
		complexity:    1.5
		score:         11.5 + ratio
		parent_family: 'layered'
		delta:         delta
	}
}

fn test_template_delta_learning_groups_and_ranges() {
	tdl := TemplateDeltaLearner{
		min_evidence: 2
	}
	req := StructureBirthRequest{
		residual_mean:       10.0
		best_posterior_mean: 0.4
		reason:              'test'
		proposals:           [tdl_test_proposal(0.4, -0.1), tdl_test_proposal(0.6, 0.1)]
	}
	lineages := {
		'layered': layered_lineage()
	}
	specs := tdl.tdl_learn([req], lineages)
	assert specs.len == 1
	spec := specs[0]
	assert spec.parent_family == 'layered'
	assert spec.operation == 'attach'
	assert spec.generation == 2
	assert spec.evidence_count == 2
	sr := meta_list(spec.constraints, 'scale_ratio')
	assert sr.len == 2
	assert math.abs(sr[0] - 0.38) < 1e-12
	assert math.abs(sr[1] - 0.62) < 1e-12
	lateral := meta_list(spec.constraints, 'lateral_ratio')
	assert math.abs(lateral[0] + 0.12) < 1e-12
	assert math.abs(lateral[1] - 0.12) < 1e-12
	pks := meta_list(spec.constraints, 'part_kinds')
	assert pks.len == 1 && pks[0] == 1.0
}

fn tdl_request() StructureBirthRequest {
	return StructureBirthRequest{
		residual_mean:       10.0
		best_posterior_mean: 0.4
		reason:              'test'
		proposals:           [tdl_test_proposal(0.4, -0.1), tdl_test_proposal(0.6, 0.1)]
	}
}

fn test_child_codebook_factory_and_cache_variant() {
	tdl := TemplateDeltaLearner{
		min_evidence: 2
	}
	spec := tdl.tdl_learn([tdl_request()], map[string]TemplateLineage{})[0]
	cb := ccf_build(spec)
	assert math.abs(cb.scale_lo - 0.38) < 1e-12
	assert math.abs(cb.scale_hi - 0.62) < 1e-12
	assert math.abs(cb.lateral_lo + 0.12) < 1e-12
	assert math.abs(cb.lateral_hi - 0.12) < 1e-12
	assert cb.part_kinds.len == 1 && cb.part_kinds[0] == 1
	assert cb.n_combo() == 3 * 1 * 6 * 1 * 3 * 3
	assert cb.lineage.parent_family == 'layered'

	cfg := InverseConfig{
		scene_family: 'composite'
	}
	app := new_inverse_app_cb(cfg, cb)
	assert app.codebook.template_variant() == spec.name
	assert app.data.cache_tag().contains(spec.name)
}

fn test_layer_child_lateral_guarantees_back_visibility() {
	mut constraints := map[string]MetaValue{}
	constraints['relation'] = 'layer'
	constraints['scale_ratio'] = [0.43, 0.62]
	constraints['lateral_ratio'] = [-0.02, 0.02]
	constraints['depth_gap'] = [0.78, 0.82]
	constraints['part_kinds'] = [1.0]
	constraints['part_hues'] = [2.0]
	spec := ChildTemplateSpec{
		name:           'layered_layer_test'
		family:         'layered'
		parent_family:  'layered'
		operation:      'layer'
		constraints:    constraints
		complexity:     2.0
		generation:     2
		evidence_count: 2
	}
	cb := ccf_build(spec)
	assert cb.lateral_lo == 0.35 && cb.lateral_hi == 0.7
	for seed in [1, 2, 3, 4, 5] {
		row := cb.sample(1, u64(seed), false).take_axis(sel1(0), 0).data_f32()
		u0 := f64(row[1])
		s0 := f64(row[3])
		z0 := f64(row[4])
		u1 := f64(row[7])
		s1 := f64(row[9])
		z1 := f64(row[10])
		r0 := s0 * fx / (cam_z - z0)
		r1 := s1 * fx / (cam_z - z1)
		assert math.abs(u1 - u0) + r1 > r0 + 0.5
	}
}

fn test_child_template_train_and_register() {
	tdl := TemplateDeltaLearner{
		min_evidence: 2
	}
	spec := tdl.tdl_learn([tdl_request()], map[string]TemplateLineage{})[0]
	child_cls := ccf_build(spec)
	mut registry := new_expert_registry({
		'old': tdl_parent_expert()
	})
	cfg := InverseConfig{
		scene_family: 'composite'
	}
	expert := registry.train_and_register(spec.name, cfg, os.temp_dir(), child_cls)
	assert expert.lineage().family == spec.name
	ch := registry.children_of('layered')
	assert ch.len == 1 && ch[0] == spec.name
}
