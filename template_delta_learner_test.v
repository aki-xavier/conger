module conger

// template_delta_learner_test.v — TemplateDeltaLearner aggregation + the
// built-in template lineages (both previously untested in this repo).
import math

fn tdl_proposal(parent string, op string, ratio f64, residual f64, score f64) TemplateProposal {
	mut delta := map[string]MetaValue{}
	delta['ratio'] = ratio
	mut metadata := map[string]MetaValue{}
	metadata['env'] = 0
	return TemplateProposal{
		family:        '${parent}_${op}_x'
		operation:     op
		residual:      residual
		complexity:    1.5
		score:         score
		parent_family: parent
		delta:         delta
		metadata:      metadata
	}
}

fn test_tdl_learn_groups_and_builds_spec() {
	requests := [
		StructureBirthRequest{
			proposals: [
				tdl_proposal('layered', 'attach', 0.4, 0.3, 1.8),
				tdl_proposal('layered', 'attach', 0.6, 0.5, 2.2),
			]
		},
	]
	lineages := {
		'layered': layered_lineage()
	}
	tdl := TemplateDeltaLearner{
		min_evidence: 2
	}
	specs := tdl.tdl_learn(requests, lineages)
	assert specs.len == 1
	s := specs[0]
	assert s.parent_family == 'layered'
	assert s.operation == 'attach'
	assert s.generation == 2 // layered lineage generation (1) + 1
	assert s.evidence_count == 2
	assert math.abs(s.residual_mean - 0.4) < 1e-12
	assert math.abs(s.score_mean - 2.0) < 1e-12
	// scale_ratio is the [min,max] evidence range padded by range_margin
	scale := meta_list(s.constraints, 'scale_ratio')
	assert scale[0] < 0.4 && scale[1] > 0.6
	assert meta_str(s.constraints, 'relation') == 'attach'
}

fn test_tdl_learn_skips_insufficient_evidence() {
	requests := [
		StructureBirthRequest{
			proposals: [
				tdl_proposal('layered', 'attach', 0.4, 0.3, 1.8),
			]
		},
	]
	tdl := TemplateDeltaLearner{
		min_evidence: 2
	}
	assert tdl.tdl_learn(requests, {
		'layered': layered_lineage()
	}).len == 0
}

fn test_tdl_repr_distinguishes_values() {
	assert tdl_repr(MetaValue(1)) == 'i:1'
	assert tdl_repr(MetaValue('x')) == 's:x'
	// lists are order-sensitive
	assert tdl_repr(MetaValue([1.0, 2.0])) != tdl_repr(MetaValue([2.0, 1.0]))
	// maps are order-independent and include their contents
	mut m1 := map[string]f64{}
	m1['a'] = 1.0
	m1['b'] = 2.0
	mut m2 := map[string]f64{}
	m2['a'] = 1.0
	m2['c'] = 2.0
	assert tdl_repr(MetaValue(m1)) != tdl_repr(MetaValue(m2))
	mut m3 := map[string]f64{}
	m3['b'] = 2.0
	m3['a'] = 1.0
	assert tdl_repr(MetaValue(m1)) == tdl_repr(MetaValue(m3))
}

fn test_tdl_range_margin() {
	tdl := TemplateDeltaLearner{
		range_margin: 0.10
	}
	r := tdl.tdl_range([0.4, 0.6])
	assert math.abs(r[0] - 0.38) < 1e-12
	assert math.abs(r[1] - 0.62) < 1e-12
}

fn test_builtin_lineages() {
	assert single_lineage().is_root()
	assert single_lineage().signature() == 'root->single:primitive'
	assert !layered_lineage().is_root()
	assert layered_lineage().generation == 1
	assert composite_lineage().parent_family == 'layered'
	assert lateral_lineage().operation == 'mirror'
	assert lateral_lineage().generation == 3
}

fn test_child_spec_lineage() {
	mut constraints := map[string]MetaValue{}
	constraints['relation'] = MetaValue('attach')
	spec := ChildTemplateSpec{
		name:           'layered_attach_abc'
		family:         'composite'
		parent_family:  'layered'
		operation:      'attach'
		constraints:    constraints
		complexity:     1.5
		generation:     2
		evidence_count: 3
		residual_mean:  0.3
		score_mean:     1.2
	}
	lin := spec.lineage()
	assert lin.family == spec.name
	assert lin.parent_family == spec.parent_family
	assert lin.operation == spec.operation
	assert lin.generation == spec.generation
	assert lin.complexity == spec.complexity
	assert meta_str(lin.delta, 'relation') == 'attach'
}
