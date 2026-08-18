module conger

// template_lineage.v — structure-template lineage/inheritance contract
// (V port of src/template_lineage.py).

struct TemplateLineage {
	family        string
	parent_family string // '' = None
	operation     string
	complexity    f64
	generation    int = 0
	delta         map[string]MetaValue
}

fn (t TemplateLineage) is_root() bool {
	return t.parent_family == ''
}

fn (t TemplateLineage) signature() string {
	parent := if t.parent_family != '' { t.parent_family } else { 'root' }
	return '${parent}->${t.family}:${t.operation}'
}

// ChildTemplateSpec is a candidate child-template constraint estimated from
// multiple template proposals.
struct ChildTemplateSpec {
	name           string
	family         string
	parent_family  string
	operation      string
	constraints    map[string]MetaValue
	complexity     f64
	generation     int
	evidence_count int
	residual_mean  f64
	score_mean     f64
}

// lineage returns the registrable lineage object for this spec.
fn (c ChildTemplateSpec) lineage() TemplateLineage {
	mut delta := map[string]MetaValue{}
	for k, v in c.constraints {
		delta[k] = v
	}
	return TemplateLineage{
		family: c.name
		parent_family: c.parent_family
		operation: c.operation
		complexity: c.complexity
		generation: c.generation
		delta: delta
	}
}

// --- the built-in template lineages (single → layered → composite) ------------

fn single_lineage() TemplateLineage {
	return TemplateLineage{
		family: 'single'
		parent_family: ''
		operation: 'primitive'
		complexity: 1.0
		generation: 0
	}
}

fn layered_lineage() TemplateLineage {
	mut delta := map[string]MetaValue{}
	delta['relation'] = 'independent_front_back'
	delta['n_objects'] = 2
	return TemplateLineage{
		family: 'layered'
		parent_family: 'single'
		operation: 'layer'
		complexity: 2.0
		generation: 1
		delta: delta
	}
}

fn composite_lineage() TemplateLineage {
	mut delta := map[string]MetaValue{}
	delta['relation'] = 'attached_on_top'
	return TemplateLineage{
		family: 'composite'
		parent_family: 'layered'
		operation: 'attach'
		complexity: 1.5
		generation: 2
		delta: delta
	}
}

fn lateral_lineage() TemplateLineage {
	mut delta := map[string]MetaValue{}
	delta['relation'] = 'mirror'
	return TemplateLineage{
		family: 'lateral'
		parent_family: 'composite'
		operation: 'mirror'
		complexity: 1.4
		generation: 3
		delta: delta
	}
}
