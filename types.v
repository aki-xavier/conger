module conger

// types.v — typed template delta / metadata / constraint records. The V port
// replaces the Python reference's free-form `dict[str, Any]` maps with these
// compile-time-checked structs, so a mistyped key becomes a compile error
// instead of a silent zero value.

// TemplateDelta carries a proposal's per-observation delta evidence (point
// estimates extracted from one concrete birth case).
pub struct TemplateDelta {
pub mut:
	relation      string
	base_kind     ?int
	part_kind     ?int
	part_hue      ?int
	ratio         ?f64
	lateral_ratio ?f64
	depth_gap     ?f64
	depth_jitter  []f64 // empty, or [lo, hi]
}

// TemplateMetadata carries a proposal's provenance and observed evidence.
pub struct TemplateMetadata {
pub mut:
	relation      string
	signature     string
	base_kind     ?int
	part_kind     ?int
	part_hue      ?int
	ratio         ?f64
	lateral_ratio ?f64
	case_index    ?int
	env           ?int
	seed          ?int
	n_cases       ?int
	residual_gain ?f64
	observed      map[string]f64
}

// TemplateConstraints carries the learned/declared constraint ranges and
// discrete support sets of a (child) template. Shared by
// ChildTemplateSpec.constraints, TemplateLineage.delta and
// StructuredHypothesis.template_delta.
pub struct TemplateConstraints {
pub mut:
	relation      string
	scale_ratio   []f64 // empty, or [lo, hi]
	lateral_ratio []f64 // empty, or [lo, hi]
	period_ratio  []f64 // empty, or [lo, hi]
	depth_gap     []f64 // empty, or [lo, hi]
	depth_jitter  []f64 // empty, or [lo, hi]
	part_kinds    []int
	part_hues     []int
	n_objects     ?int
}
