module conger

// types.v — shared heterogeneous value type for the free-form `dict[str, Any]`
// maps in the Python reference (template delta / metadata / constraints).

pub type MetaValue = int | f64 | string | map[string]f64 | []f64

// The `meta_*` readers are total: they never panic on a type mismatch, so a
// MetaValue produced by one subsystem (e.g. an int delta) can be read by
// another (e.g. as f64) without aborting the whole process. Numeric variants
// coerce to the requested type; the wrong structural variant yields the zero
// value.

// meta_str reads a string MetaValue ('' if absent or non-string); numeric
// variants are stringified for convenience.
pub fn meta_str(m map[string]MetaValue, key string) string {
	v := m[key] or { return '' }
	return match v {
		string { v }
		int { v.str() }
		f64 { v.str() }
		else { '' }
	}
}

// meta_int reads an int MetaValue (0 if absent; f64 is truncated).
pub fn meta_int(m map[string]MetaValue, key string) int {
	v := m[key] or { return 0 }
	return match v {
		int { v }
		f64 { int(v) }
		else { 0 }
	}
}

// meta_f64 reads an f64 MetaValue (0.0 if absent; int is promoted).
pub fn meta_f64(m map[string]MetaValue, key string) f64 {
	v := m[key] or { return 0.0 }
	return match v {
		f64 { v }
		int { f64(v) }
		else { 0.0 }
	}
}

// meta_map reads a nested map[string]f64 MetaValue (empty map if absent).
pub fn meta_map(m map[string]MetaValue, key string) map[string]f64 {
	v := m[key] or { return map[string]f64{} }
	return match v {
		map[string]f64 {
			v
		}
		else {
			map[string]f64{}
		}
	}
}

// meta_list reads a []f64 MetaValue (empty slice if absent).
pub fn meta_list(m map[string]MetaValue, key string) []f64 {
	v := m[key] or { return []f64{} }
	return match v {
		[]f64 { v }
		else { []f64{} }
	}
}
