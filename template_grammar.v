module conger

// template_grammar.v — bounded geometric template grammar (V port of
// src/template_grammar.py).

struct TemplateRule {
	operation  string
	base_kind  int
	part_kind  int = -1 // -1 = None
	complexity f64 = 1.0
	depth      int = 1
}

fn (r TemplateRule) signature() string {
	if r.part_kind < 0 {
		return 'primitive:${r.base_kind}'
	}
	return '${r.operation}:${r.base_kind}:${r.part_kind}'
}

struct TemplateGrammar {
	operations []string = ['attach']
	max_depth  int      = 2
	kinds      []int
}

fn new_template_grammar(operations []string, max_depth int, kinds []int) TemplateGrammar {
	for op in operations {
		if op != 'attach' && op != 'layer' && op != 'mirror' && op != 'repeat' {
			panic('unknown template operation: ${op}')
		}
	}
	assert max_depth >= 1
	mut ks := kinds.clone()
	if ks.len == 0 {
		ks = [0, 1, 2]
	}
	return TemplateGrammar{
		operations: operations
		max_depth: max_depth
		kinds: ks
	}
}

fn (g TemplateGrammar) op_complexity(op string) (f64, bool) {
	match op {
		'attach' { return 1.5, false }
		'layer' { return 2.0, false }
		'mirror' { return 1.4, true }
		'repeat' { return 1.3, true }
		else { return 1.0, false }
	}
}

// primitives returns the depth=1 primitive rules.
fn (g TemplateGrammar) primitives() []TemplateRule {
	mut out := []TemplateRule{}
	for k in g.kinds {
		out << TemplateRule{
			operation: 'primitive'
			base_kind: k
			part_kind: -1
			complexity: 1.0
			depth: 1
		}
	}
	return out
}

// composites returns the depth=2 primitive∘primitive rules.
fn (g TemplateGrammar) composites() []TemplateRule {
	mut out := []TemplateRule{}
	for op in g.operations {
		complexity, same_kind := g.op_complexity(op)
		for base in g.kinds {
			parts := if same_kind { [base] } else { g.kinds }
			for part in parts {
				out << TemplateRule{
					operation: op
					base_kind: base
					part_kind: part
					complexity: complexity
					depth: 2
				}
			}
		}
	}
	return out
}

// rules returns the bounded search space (depth1 primitives + optional depth2).
fn (g TemplateGrammar) rules() []TemplateRule {
	mut out := g.primitives()
	if g.max_depth >= 2 {
		out << g.composites()
	}
	return out
}
