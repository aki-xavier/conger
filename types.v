module conger

// types.v — shared heterogeneous value type for the free-form `dict[str, Any]`
// maps in the Python reference (template delta / metadata / constraints).

pub type MetaValue = int | f64 | string | map[string]f64 | []f64

// meta_str reads a string MetaValue ('' if absent).
pub fn meta_str(m map[string]MetaValue, key string) string {
	v := m[key] or { return '' }
	return v as string
}

// meta_int reads an int MetaValue (0 if absent).
pub fn meta_int(m map[string]MetaValue, key string) int {
	v := m[key] or { return 0 }
	return v as int
}

// meta_f64 reads an f64 MetaValue (0.0 if absent).
pub fn meta_f64(m map[string]MetaValue, key string) f64 {
	v := m[key] or { return 0.0 }
	return v as f64
}

// meta_map reads a nested map[string]f64 MetaValue (empty map if absent).
pub fn meta_map(m map[string]MetaValue, key string) map[string]f64 {
	v := m[key] or { return map[string]f64{} }
	return v as map[string]f64
}

// meta_list reads a []f64 MetaValue (empty slice if absent).
pub fn meta_list(m map[string]MetaValue, key string) []f64 {
	v := m[key] or { return []f64{} }
	return v as []f64
}
