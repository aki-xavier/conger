module conger

// template_proposal_test.v — template proposal + birth-request integration
// (renderer-free subset; the renderer-driven compositor test is ported with the
// visual codebook modules).
import mlx

struct StaticProposer {}

fn (s StaticProposer) propose(cases []StructureCase) []TemplateProposal {
	mut params := []f64{len: 14}
	for i in 0 .. 14 {
		params[i] = f64(i)
	}
	return [
		TemplateProposal{
			family:        'composite'
			operation:     'attach'
			params:        params
			residual:      1.0
			complexity:    1.5
			score:         2.5
			parent_family: 'layered'
			delta:         TemplateDelta{
				relation: 'attach'
			}
			metadata:      TemplateMetadata{
				n_cases: cases.len
			}
		},
	]
}

fn test_birth_request_carries_template_proposals() {
	estimate := StructuredHypothesis[voidptr]{
		structure_id: 'single'
		params:       [0.0]
		residual:     10.0
	}
	decision := GenericStructureDecision[voidptr]{
		estimate:            estimate
		posterior:           {
			'single': 0.4
		}
		residuals:           {
			'single': 10.0
		}
		scores:              {
			'single': 11.0
		}
		needs_new_structure: true
	}
	mut controller := StructureBirthController{
		min_cases: 2
		proposer:  StaticProposer{}
	}
	assert controller.observe(decision, mlx.zeros([2], .float32), mlx.zeros([2], .float32)) == none
	req := controller.observe(decision, mlx.zeros([2], .float32), mlx.zeros([2], .float32)) or {
		panic('expected request')
	}
	assert req.proposals.len == 1
	assert req.cases[0].structure_id == 'single'
	assert req.proposals[0].parent_family == 'layered'
	assert req.proposals[0].delta.relation == 'attach'
	assert (req.proposals[0].metadata.n_cases or { 0 }) == 2
	assert req.reason.contains('1 个模板提案')
}
