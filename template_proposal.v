module conger

// template_proposal.v — unified description of a structure-birth candidate
// (V port of src/template_proposal.py).

struct TemplateProposal {
	family        string
	operation     string
	params        []f64
	residual      f64
	complexity    f64
	score         f64
	parent_family string
	delta         map[string]MetaValue
	metadata      map[string]MetaValue
}

// TemplateProposer generates scorable new-template candidates from birth evidence.
interface TemplateProposer {
	propose(cases []StructureCase) []TemplateProposal
}
