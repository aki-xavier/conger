module conger

// template_grammar_test.v — bounded composition space + operator constraints.

fn test_template_grammar_depth_bounds() {
	ops := ['attach', 'layer', 'mirror', 'repeat']
	g1 := new_template_grammar(ops, 1, [])
	assert g1.rules().len == 3
	for r in g1.rules() {
		assert r.operation == 'primitive'
	}
	g2 := new_template_grammar(ops, 2, [])
	assert g2.primitives().len == 3
	assert g2.composites().len == 24
	assert g2.rules().len == 27
}

fn test_template_grammar_operator_constraints() {
	g := new_template_grammar(['mirror', 'repeat'], 2, [])
	for r in g.composites() {
		assert r.base_kind == r.part_kind
	}
}
