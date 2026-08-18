module conger

// registry_manifest.v — JSON persistence for dynamic child templates and the
// expert registry (V port of src/registry_manifest.py).
import json2
import os

// RegisteredChildTemplate records a trained dynamic child template.
pub struct RegisteredChildTemplate {
pub:
	spec       ChildTemplateSpec
	model_path string
}

// RegistryManifest is the strong-typed registry_manifest.json.
pub struct RegistryManifest {
pub:
	children []RegisteredChildTemplate
	pending  []ChildTemplateSpec
	version  int = 1
}

// rm_meta_to_any converts a MetaValue to a json2.Any tree node.
pub fn rm_meta_to_any(v MetaValue) json2.Any {
	match v {
		string {
			return json2.Any(v)
		}
		int {
			return json2.Any(i64(v))
		}
		f64 {
			return json2.Any(v)
		}
		[]f64 {
			mut arr := []json2.Any{}
			for x in v {
				arr << json2.Any(x)
			}
			return json2.Any(arr)
		}
		map[string]f64 {
			mut m := map[string]json2.Any{}
			for k, x in v {
				m[k] = json2.Any(x)
			}
			return json2.Any(m)
		}
	}
}

// rm_meta_from_any converts a json2.Any tree node back to a MetaValue.
//
// Note: json2 decodes every JSON number as f64, so an `int` MetaValue written
// through `rm_meta_to_any` comes back as f64 after a JSON round-trip. The
// integer branches below still preserve the type for in-process (non-JSON)
// Any trees; the total `meta_*` readers in types.v handle the f64 form.
pub fn rm_meta_from_any(a json2.Any) MetaValue {
	if a is string {
		return MetaValue(a as string)
	}
	if a is []json2.Any {
		mut out := []f64{}
		for x in a as []json2.Any {
			out << x.f64()
		}
		return MetaValue(out)
	}
	if a is map[string]json2.Any {
		mut m := map[string]f64{}
		for k, x in a as map[string]json2.Any {
			m[k] = x.f64()
		}
		return MetaValue(m)
	}
	// signed / unsigned integer variants → int (in-process preservation)
	if a is i64 || a is int || a is i32 || a is i16 || a is i8 || a is u64 || a is u32 || a is u16
		|| a is u8 {
		return MetaValue(int(a.i64()))
	}
	// f32/f64 (and, as a safe default, bool/null/time) → f64
	return MetaValue(a.f64())
}

// rm_spec_to_any serialises a ChildTemplateSpec.
pub fn rm_spec_to_any(s ChildTemplateSpec) map[string]json2.Any {
	mut c := map[string]json2.Any{}
	for k, v in s.constraints {
		c[k] = rm_meta_to_any(v)
	}
	return {
		'name':           json2.Any(s.name)
		'family':         json2.Any(s.family)
		'parent_family':  json2.Any(s.parent_family)
		'operation':      json2.Any(s.operation)
		'constraints':    json2.Any(c)
		'complexity':     json2.Any(s.complexity)
		'generation':     json2.Any(i64(s.generation))
		'evidence_count': json2.Any(i64(s.evidence_count))
		'residual_mean':  json2.Any(s.residual_mean)
		'score_mean':     json2.Any(s.score_mean)
	}
}

// rm_spec_from_any deserialises a ChildTemplateSpec.
pub fn rm_spec_from_any(m map[string]json2.Any) ChildTemplateSpec {
	mut constraints := map[string]MetaValue{}
	cm := m['constraints'] or { json2.Any(map[string]json2.Any{}) }.as_map()
	for k, v in cm {
		constraints[k] = rm_meta_from_any(v)
	}
	return ChildTemplateSpec{
		name:           (m['name'] or { json2.Any('') }).str()
		family:         (m['family'] or { json2.Any('') }).str()
		parent_family:  (m['parent_family'] or { json2.Any('') }).str()
		operation:      (m['operation'] or { json2.Any('') }).str()
		constraints:    constraints
		complexity:     (m['complexity'] or { json2.Any(f64(0)) }).f64()
		generation:     (m['generation'] or { json2.Any(i64(0)) }).int()
		evidence_count: (m['evidence_count'] or { json2.Any(i64(0)) }).int()
		residual_mean:  (m['residual_mean'] or { json2.Any(f64(0)) }).f64()
		score_mean:     (m['score_mean'] or { json2.Any(f64(0)) }).f64()
	}
}

// rm_manifest_to_any serialises the whole manifest.
pub fn rm_manifest_to_any(mf RegistryManifest) map[string]json2.Any {
	mut children := []json2.Any{}
	for c in mf.children {
		children << json2.Any({
			'spec':       json2.Any(rm_spec_to_any(c.spec))
			'model_path': json2.Any(c.model_path)
		})
	}
	mut pending := []json2.Any{}
	for p in mf.pending {
		pending << json2.Any(rm_spec_to_any(p))
	}
	return {
		'version':  json2.Any(i64(mf.version))
		'children': json2.Any(children)
		'pending':  json2.Any(pending)
	}
}

// rm_manifest_from_any deserialises the whole manifest.
pub fn rm_manifest_from_any(m map[string]json2.Any) RegistryManifest {
	mut children := []RegisteredChildTemplate{}
	ca := m['children'] or { json2.Any([]json2.Any{}) }.as_array()
	for child in ca {
		cm := child.as_map()
		model_path := (cm['model_path'] or { json2.Any('') }).str()
		children << RegisteredChildTemplate{
			spec:       rm_spec_from_any((cm['spec'] or { json2.Any(map[string]json2.Any{}) }).as_map())
			model_path: model_path
		}
	}
	mut pending := []ChildTemplateSpec{}
	pa := m['pending'] or { json2.Any([]json2.Any{}) }.as_array()
	for p in pa {
		pending << rm_spec_from_any(p.as_map())
	}
	return RegistryManifest{
		children: children
		pending:  pending
		version:  (m['version'] or { json2.Any(i64(1)) }).int()
	}
}

// rm_save writes the manifest to path (creating parent directories).
pub fn rm_save(mf RegistryManifest, path string) {
	dir := os.dir(path)
	if dir != '.' && dir != '' && !os.exists(dir) {
		os.mkdir_all(dir, os.MkdirParams{}) or { panic(err) }
	}
	content := json2.encode(rm_manifest_to_any(mf), json2.EncoderOptions{})
	os.write_file(path, content) or { panic(err) }
}

// rm_load reads and parses the manifest at path.
pub fn rm_load(path string) RegistryManifest {
	content := os.read_file(path) or { panic(err) }
	decoded := json2.decode[map[string]json2.Any](content) or { panic(err) }
	return rm_manifest_from_any(decoded)
}
