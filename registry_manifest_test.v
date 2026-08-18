module conger

// registry_manifest_test.v — JSON persistence round-trip + total MetaValue readers.
import json2
import os

fn manifest_spec() ChildTemplateSpec {
	return ChildTemplateSpec{
		name:           'layered_attach_abc'
		family:         'composite'
		parent_family:  'layered'
		operation:      'attach'
		constraints:    {
			'relation':      MetaValue('attach')
			'scale_ratio':   MetaValue([0.4, 0.6])
			'part_kinds':    MetaValue([0.0, 1.0])
			'depth_gap':     MetaValue(0.1)
			'evidence_kind': MetaValue(7)
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
	rm_save(mf, path)
	loaded := rm_load(path)
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

	// constraints round-trip: string, []f64 and numeric values survive and are
	// read back without panicking (json2 decodes numbers as f64, so ints come
	// back as f64 — the total readers below must still return the right value).
	assert meta_str(c.spec.constraints, 'relation') == 'attach'
	scale := meta_list(c.spec.constraints, 'scale_ratio')
	assert scale.len == 2 && scale[0] == 0.4 && scale[1] == 0.6
	assert meta_f64(c.spec.constraints, 'depth_gap') == 0.1
	assert meta_int(c.spec.constraints, 'evidence_kind') == 7
}

fn test_meta_value_readers_are_total() {
	mut inner := map[string]f64{}
	inner['a'] = 1.0
	m := {
		'i': MetaValue(3)
		'f': MetaValue(2.5)
		's': MetaValue('x')
		'l': MetaValue([1.0, 2.0])
		'm': MetaValue(inner)
	}
	assert meta_int(m, 'i') == 3
	assert meta_int(m, 'f') == 2 // f64 truncated toward zero
	assert meta_int(m, 'missing') == 0
	assert meta_f64(m, 'f') == 2.5
	assert meta_f64(m, 'i') == 3.0
	assert meta_str(m, 's') == 'x'
	assert meta_str(m, 'i') == '3'
	assert meta_list(m, 'l')[1] == 2.0
	assert meta_map(m, 'm')['a'] == 1.0
	// type mismatches fall back to zero values instead of panicking
	assert meta_int(m, 's') == 0
	assert meta_f64(m, 's') == 0.0
	assert meta_str(m, 'm') == ''
	assert meta_list(m, 'i').len == 0
	assert meta_map(m, 'i').len == 0
}

fn test_rm_meta_from_any_preserves_int_in_process() {
	// A non-JSON Any tree (e.g. built directly) preserves integer variants.
	assert rm_meta_from_any(json2.Any(i64(7))) == MetaValue(7)
	assert rm_meta_from_any(json2.Any(2.5)) == MetaValue(2.5)
	assert rm_meta_from_any(json2.Any('hi')) == MetaValue('hi')
	assert rm_meta_from_any(json2.Any([json2.Any(1.0), json2.Any(2.0)])) == MetaValue([
		1.0,
		2.0,
	])
}

fn test_rm_spec_name_and_complexity_roundtrip() {
	spec := manifest_spec()
	mf := RegistryManifest{
		pending: [spec]
		version: 1
	}
	path := os.temp_dir() + '/conger_registry_manifest_pending_test.json'
	rm_save(mf, path)
	loaded := rm_load(path)
	assert loaded.pending.len == 1
	assert loaded.pending[0].name == 'layered_attach_abc'
	assert loaded.pending[0].score_mean == 1.2
}
