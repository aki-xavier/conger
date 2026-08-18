module conger

// template_lineage_test.v — template parent/child lineage contract tests.

fn test_codebook_template_lineages() {
	single := single_lineage()
	layered := layered_lineage()
	composite := composite_lineage()
	assert single.is_root()
	assert layered.parent_family == 'single'
	assert composite.parent_family == 'layered'
	assert meta_str(composite.delta, 'relation') == 'attached_on_top'
	assert single.generation == 0 && layered.generation == 1 && composite.generation == 2
}

fn test_registry_lineage_tree() {
	mut experts := map[string]SceneExpert{}
	experts['single'] = SceneExpert{
		name: 'single'
		app: new_inverse_app(InverseConfig{scene_family: 'single'})
	}
	experts['layered'] = SceneExpert{
		name: 'layered'
		app: new_inverse_app(InverseConfig{scene_family: 'layered'})
	}
	experts['composite'] = SceneExpert{
		name: 'composite'
		app: new_inverse_app(InverseConfig{scene_family: 'composite'})
	}
	registry := new_expert_registry(experts)
	lineages := registry.lineages()
	assert lineages['composite'].signature() == 'layered->composite:attach'
	assert registry.children_of('single') == ['layered']
	assert registry.children_of('layered') == ['composite']
	assert registry.children_of('composite') == []
}
