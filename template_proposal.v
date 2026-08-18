module conger

// template_proposal.v — unified description of a structure-birth candidate.

pub struct TemplateProposal {
pub:
	family        string
	operation     string
	params        []f64
	residual      f64
	complexity    f64
	score         f64
	parent_family string
	delta         TemplateDelta
	metadata      TemplateMetadata
}

// TemplateProposer generates scorable new-template candidates from birth evidence.
pub interface TemplateProposer {
	propose(cases []StructureCase) []TemplateProposal
}
