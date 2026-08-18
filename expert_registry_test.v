module conger

// expert_registry_test.v — V port of tests/test_expert_registry.py
// (registration/loading semantics; the render-decide path is covered by
// structure_gate_test / child_template_workflow_test).

import os

fn er_parent_expert() SceneExpert {
	return SceneExpert{
		name: 'layered'
		app: new_inverse_app(InverseConfig{scene_family: 'layered'})
	}
}

fn test_default_registry_includes_three_structure_families() {
	registry := default_expert_registry()
	assert registry.experts.len == 3
	assert 'single' in registry.experts
	assert 'layered' in registry.experts
	assert 'composite' in registry.experts
}

fn test_missing_expert_model_fails_closed() {
	dir := os.join_path(os.temp_dir(), 'conger_missing_model')
	os.mkdir_all(dir, os.MkdirParams{}) or {}
	cfg := InverseConfig{scene_family: 'single'}
	mut failed := false
	scene_expert_from_config('missing', cfg, dir) or { failed = true }
	assert failed
}

fn test_train_and_register_workflow() {
	mut registry := new_expert_registry({
		'old': er_parent_expert()
	})
	cfg := InverseConfig{scene_family: 'single'}
	expert := registry.train_and_register('new', cfg, os.temp_dir(), none)
	assert 'new' in registry.experts
	assert expert.app.cfg.family() == 'single'
}
