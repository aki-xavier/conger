module conger

// registry_manifest_test.v — V port of tests/test_registry_manifest.py.

import os

fn rm_test_spec(name string) ChildTemplateSpec {
	mut constraints := map[string]MetaValue{}
	constraints['relation'] = 'attach'
	constraints['scale_ratio'] = [0.4, 0.6]
	constraints['lateral_ratio'] = [-0.1, 0.1]
	constraints['part_kinds'] = [1.0]
	constraints['part_hues'] = [2.0]
	return ChildTemplateSpec{
		name: name
		family: 'composite'
		parent_family: 'layered'
		operation: 'attach'
		constraints: constraints
		complexity: 1.5
		generation: 2
		evidence_count: 2
		residual_mean: 10.0
		score_mean: 11.5
	}
}

fn test_registry_manifest_roundtrip() {
	registered := rm_test_spec('registered_child')
	pending := rm_test_spec('pending_child')
	path := os.join_path(os.temp_dir(), 'conger_registry_manifest_test.json')
	rm_save(RegistryManifest{
		children: [RegisteredChildTemplate{
			spec: registered
			model_path: 'child.safetensors'
		}]
		pending: [pending]
	}, path)
	out := rm_load(path)
	assert out.children.len == 1
	assert out.children[0].spec.name == 'registered_child'
	assert out.children[0].model_path == 'child.safetensors'
	assert out.pending.len == 1
	pks := meta_list(out.pending[0].constraints, 'part_kinds')
	assert pks.len == 1 && pks[0] == 1.0
	os.rm(path) or {}
}

fn rm_parent_expert() SceneExpert {
	return SceneExpert{
		name: 'layered'
		app: new_inverse_app(InverseConfig{scene_family: 'layered'})
	}
}

fn test_registry_manifest_restores_child_expert() {
	spec := rm_test_spec('restored_spec')
	child_cls := ccf_build(spec)
	cfg := InverseConfig{scene_family: 'composite'}
	dir := os.join_path(os.temp_dir(), 'conger_manifest_dir')
	os.mkdir_all(dir, os.MkdirParams{}) or {}
	model_path := new_inverse_app_cb(cfg, child_cls).default_model_path(dir)
	os.write_file(model_path, '') or {}
	mut registry := new_expert_registry({
		'layered': rm_parent_expert()
	})
	registry.child_specs[spec.name] = spec
	registry.child_model_paths[spec.name] = model_path
	manifest_path := registry.save_manifest(os.join_path(dir, 'manifest.json'))

	mut restored := new_expert_registry({
		'layered': rm_parent_expert()
	})
	restored.load_manifest(manifest_path, dir, false) or { panic(err) }
	assert spec.name in restored.experts
	assert restored.experts[spec.name].lineage().family == spec.name
	ch := restored.children_of('layered')
	assert ch.len == 1 && ch[0] == spec.name
	os.rm(os.join_path(dir, 'manifest.json')) or {}
	os.rm(model_path) or {}
}

fn test_registry_manifest_missing_model_policy() {
	spec := rm_test_spec('missing_child')
	dir := os.join_path(os.temp_dir(), 'conger_manifest_dir2')
	os.mkdir_all(dir, os.MkdirParams{}) or {}
	path := os.join_path(dir, 'manifest.json')
	rm_save(RegistryManifest{
		children: [RegisteredChildTemplate{
			spec: spec
			model_path: os.join_path(dir, 'none.safetensors')
		}]
	}, path)
	mut registry := new_expert_registry({
		'layered': rm_parent_expert()
	})
	registry.load_manifest(path, dir, true) or { panic(err) }
	assert spec.name !in registry.experts
	if _ := registry.load_manifest(path, dir, false) {
		assert false, 'expected missing-model failure'
	}
	os.rm(path) or {}
}
