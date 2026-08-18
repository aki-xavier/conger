module conger

// child_template_workflow_test.v — V port of tests/test_child_template_workflow.py
// (the render-proposal → explicit registration path).

import os

import mlx

fn ctw_parent_expert() SceneExpert {
	return SceneExpert{
		name: 'layered'
		app: new_inverse_app(InverseConfig{scene_family: 'layered'})
	}
}

fn ctw_rendered_request() (StructureBirthRequest, []f64) {
	base := [0.0, 72.0, 90.0, 0.45, 3.2, 1.0, 0.0, 1.0]
	proposer := CompositeTemplateProposer{
		ratios: [0.45]
		lateral_ratios: [0.0]
		part_kinds: [1]
		part_hues: [2]
		max_proposals: 2
		grammar: new_template_grammar(['attach'], 2, [0, 1, 2])
		codebook: new_composite_codebook(InverseConfig{scene_family: 'composite'})
	}
	gt := ctp_attach(base, 1, 2, 0.45, 0.0)
	cb := new_composite_codebook(InverseConfig{scene_family: 'composite'})
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	scene := cb.to_scene(gt)
	fl := renderer.render(scene, cam_l)
	fr := renderer.render(scene, cam_r)
	case_ := StructureCase{
		fl: fl
		fr: fr
		residuals: {
			'layered': 1000.0
		}
		posterior: {
			'layered': 1.0
		}
		params: base
		structure_id: 'layered'
	}
	return StructureBirthRequest{
		cases: [case_, case_]
		residual_mean: 1000.0
		best_posterior_mean: 1.0
		reason: 'test'
		proposals: proposer.ctp_propose([case_, case_])
	}, gt
}

fn test_child_template_workflow_end_to_end() {
	request, _ := ctw_rendered_request()
	mut registry := new_expert_registry({
		'layered': ctw_parent_expert()
	})
	registry.enable_child_template_learning()
	pending := registry.observe_birth_request(request)
	assert pending.len == 1
	reg := registry.confirm_child_template(pending[0].name, os.temp_dir())
	assert reg.spec.parent_family == 'layered'
	assert reg.spec.operation == 'attach'
	assert reg.spec.evidence_count == 2
	assert reg.expert.lineage().parent_family == 'layered'
	assert reg.spec.name in registry.lineages()
}

fn test_dynamic_child_uses_composite_geometry_family() {
	// a dynamic child with geometry_family="composite" must inherit the composite
	// geometry evidence (not fall back to 0 for the unknown expert name).
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	p := ccb_sample(1, 42, false).take_axis(sel1(0), 0).data_f32()
	mut prm := []f64{len: 14}
	for i in 0 .. 14 {
		prm[i] = f64(p[i])
	}
	cb := new_composite_codebook(InverseConfig{scene_family: 'composite'})
	scene := cb.to_scene(prm)
	fl := renderer.render(scene, cam_l)
	fr := renderer.render(scene, cam_r)
	composite_cost := sg_costs(fl, fr)['composite']
	estimate := StructuredHypothesis{
		scene: scene
		structure_id: 'child'
		geometry_family: 'composite'
		params: [0.0]
		residual: 1.0
	}
	out := new_structure_gate().decide({
		'child': estimate
	}, fl, fr)
	assert out.estimate.geometry_cost == composite_cost
}
