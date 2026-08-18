module conger

// child_template_workflow.v — orchestration from birth proposals to explicit
// child-template training/registration (V port of src/child_template_workflow.py).

struct ChildTemplateRegistration {
	spec         ChildTemplateSpec
	codebook_cls ChildCodebook
	expert       SceneExpert
}

struct ChildTemplateWorkflow {
	learner TemplateDeltaLearner
}

fn new_child_template_workflow() ChildTemplateWorkflow {
	return ChildTemplateWorkflow{
		learner: TemplateDeltaLearner{
			min_evidence: 2
		}
	}
}

// ctw_learn learns candidate child specs from birth requests.
fn (w ChildTemplateWorkflow) ctw_learn(requests []StructureBirthRequest, lineages map[string]TemplateLineage) []ChildTemplateSpec {
	return w.learner.tdl_learn(requests, lineages)
}

// ctw_materialize materialises a spec into a constraint-bearing child codebook.
fn (w ChildTemplateWorkflow) ctw_materialize(spec ChildTemplateSpec) ChildCodebook {
	return ccf_build(spec)
}
