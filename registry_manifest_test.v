module conger

// registry_manifest_test.v — JSON persistence round-trip of the typed manifest.
import os

fn manifest_spec() ChildTemplateSpec {
	return ChildTemplateSpec{
		name:           'layered_attach_abc'
		family:         'composite'
		parent_family:  'layered'
		operation:      'attach'
		constraints:    TemplateConstraints{
			relation:    'attach'
			scale_ratio: [0.4, 0.6]
			part_kinds:  [0, 1]
			depth_gap:   [0.1, 0.9]
			n_objects:   2
		}
		complexity:     1.5
		generation:     2
		evidence_count: 12
		residual_mean:  0.3
		score_mean:     1.2
	}
}

fn test_manifest_roundtrip() {
	spec := manifest_spec()
	mf := RegistryManifest{
		children: [
			RegisteredChildTemplate{
				spec:       spec
				model_path: '/tmp/foo.safetensors'
			},
		]
		pending:  [spec]
		version:  1
	}
	path := os.temp_dir() + '/conger_registry_manifest_test.json'
	rm_save(mf, path) or { panic(err) }
	loaded := rm_load(path) or { panic(err) }
	assert loaded.version == 1
	assert loaded.children.len == 1
	assert loaded.pending.len == 1

	c := loaded.children[0]
	assert c.model_path == '/tmp/foo.safetensors'
	assert c.spec.name == spec.name
	assert c.spec.family == spec.family
	assert c.spec.parent_family == spec.parent_family
	assert c.spec.operation == spec.operation
	assert c.spec.generation == 2
	assert c.spec.evidence_count == 12
	assert c.spec.complexity == 1.5

	// typed constraints round-trip exactly
	assert c.spec.constraints.relation == 'attach'
	assert c.spec.constraints.scale_ratio.len == 2
	assert c.spec.constraints.scale_ratio[0] == 0.4
	assert c.spec.constraints.scale_ratio[1] == 0.6
	assert c.spec.constraints.part_kinds == [0, 1]
	assert c.spec.constraints.depth_gap == [0.1, 0.9]
	assert (c.spec.constraints.n_objects or { 0 }) == 2
}

fn test_rm_spec_name_and_complexity_roundtrip() {
	spec := manifest_spec()
	mf := RegistryManifest{
		pending: [spec]
		version: 1
	}
	path := os.temp_dir() + '/conger_registry_manifest_pending_test.json'
	rm_save(mf, path) or { panic(err) }
	loaded := rm_load(path) or { panic(err) }
	assert loaded.pending.len == 1
	assert loaded.pending[0].name == 'layered_attach_abc'
	assert loaded.pending[0].score_mean == 1.2
	assert loaded.pending[0].constraints.part_kinds == [0, 1]
}
