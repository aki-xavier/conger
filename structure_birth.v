module conger

// structure_birth.v — unknown-structure sample queue, birth request and
// candidate training registration (V port of src/structure_birth.py).
import mlx

pub struct StructureCase {
pub:
	fl           mlx.Array
	fr           mlx.Array
	residuals    map[string]f64
	posterior    map[string]f64
	params       []f64
	structure_id string = 'unknown'
}

pub struct StructureBirthRequest {
pub:
	cases               []StructureCase
	residual_mean       f64
	best_posterior_mean f64
	reason              string
	proposals           []TemplateProposal
}

pub struct StructureBirthController {
pub:
	min_cases int = 3
	max_cases int = 128
	proposer  ?TemplateProposer
pub mut:
	cases []StructureCase
}

// observe records one gate result; returns none until the evidence threshold is
// reached, then a birth request (and clears the accumulated cases).
pub fn (mut b StructureBirthController) observe(decision GenericStructureDecision, fl mlx.Array, fr mlx.Array) ?StructureBirthRequest {
	if !decision.needs_new_structure {
		return none
	}
	b.cases << StructureCase{
		fl:           fl
		fr:           fr
		residuals:    decision.residuals
		posterior:    decision.posterior
		params:       decision.estimate.params
		structure_id: decision.estimate.structure_id
	}
	if b.cases.len > b.max_cases {
		b.cases = b.cases[b.cases.len - b.max_cases..]
	}
	if b.cases.len < b.min_cases {
		return none
	}
	cases := b.cases.clone()
	mut rsum := 0.0
	mut psum := 0.0
	for c in cases {
		rsum += min_of(c.residuals)
		psum += max_of(c.posterior)
	}
	residual_mean := rsum / f64(cases.len)
	best_posterior_mean := psum / f64(cases.len)
	b.cases = []StructureCase{}
	mut proposals := []TemplateProposal{}
	if p := b.proposer {
		proposals = p.propose(cases)
	}
	return StructureBirthRequest{
		cases:               cases
		residual_mean:       residual_mean
		best_posterior_mean: best_posterior_mean
		reason:              '${cases.len} 个样本在所有结构专家中均不兼容; 已生成 ${proposals.len} 个模板提案; 请提供新的可渲染结构族并训练注册'
		proposals:           proposals
	}
}
