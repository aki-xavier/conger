module conger

// template_lineage.v — structure-template lineage/inheritance contract.

pub struct TemplateLineage {
pub:
	family        string
	parent_family string // '' = None
	operation     string
	complexity    f64
	generation    int
	delta         TemplateConstraints
}

pub fn (t TemplateLineage) is_root() bool {
	return t.parent_family == ''
}

pub fn (t TemplateLineage) signature() string {
	parent := if t.parent_family != '' { t.parent_family } else { 'root' }
	return '${parent}->${t.family}:${t.operation}'
}

// ChildTemplateSpec is a candidate child-template constraint estimated from
// multiple template proposals.
pub struct ChildTemplateSpec {
pub:
	name           string
	family         string
	parent_family  string
	operation      string
	constraints    TemplateConstraints
	complexity     f64
	generation     int
	evidence_count int
	residual_mean  f64
	score_mean     f64
}

// lineage returns the registrable lineage object for this spec.
pub fn (c ChildTemplateSpec) lineage() TemplateLineage {
	return TemplateLineage{
		family:        c.name
		parent_family: c.parent_family
		operation:     c.operation
		complexity:    c.complexity
		generation:    c.generation
		delta:         c.constraints
	}
}

// --- the built-in template lineages (single → layered → composite) ------------

pub fn single_lineage() TemplateLineage {
	return TemplateLineage{
		family:        'single'
		parent_family: ''
		operation:     'primitive'
		complexity:    1.0
		generation:    0
	}
}

pub fn layered_lineage() TemplateLineage {
	return TemplateLineage{
		family:        'layered'
		parent_family: 'single'
		operation:     'layer'
		complexity:    2.0
		generation:    1
		delta:         TemplateConstraints{
			relation:  'independent_front_back'
			n_objects: 2
		}
	}
}

pub fn composite_lineage() TemplateLineage {
	return TemplateLineage{
		family:        'composite'
		parent_family: 'layered'
		operation:     'attach'
		complexity:    1.5
		generation:    2
		delta:         TemplateConstraints{
			relation: 'attached_on_top'
		}
	}
}

pub fn lateral_lineage() TemplateLineage {
	return TemplateLineage{
		family:        'lateral'
		parent_family: 'composite'
		operation:     'mirror'
		complexity:    1.4
		generation:    3
		delta:         TemplateConstraints{
			relation: 'mirror'
		}
	}
}
