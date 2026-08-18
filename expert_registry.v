module conger

// expert_registry.v — structure-expert registry + render-residual gating
// (V port of src/expert_registry.py).
import os
import mlx

// SceneExpert is a fixed scene family + MixtureSPN + inverse app.
struct SceneExpert {
	name string
	app  InverseApp
	net  MixtureSPN
}

// scene_expert_from_config loads a default model (fail-closed on missing model).
fn scene_expert_from_config(name string, cfg InverseConfig, artifacts string) !SceneExpert {
	app := new_inverse_app(cfg)
	path := app.default_model_path(artifacts)
	if !os.exists(path) {
		return error('结构专家 ${name} 缺模型: ${path}')
	}
	return SceneExpert{
		name: name
		app:  app
		net:  load_mixture_spn(path)
	}
}

// reconstruct returns the hypothesis for this expert on a frame pair.
fn (e SceneExpert) reconstruct(fl mlx.Array, fr mlx.Array) StructuredHypothesis {
	return e.app.reconstruct_scene(e.net, fl, fr)
}

// lineage returns the scene-family lineage metadata for this expert.
fn (e SceneExpert) lineage() TemplateLineage {
	return e.app.codebook.template_lineage()
}

// ExpertRegistry is a set of experts gated by render residual.
struct ExpertRegistry {
mut:
	experts             map[string]SceneExpert
	gate                StructureGate
	child_workflow      ?ChildTemplateWorkflow
	manifest_path       string
	last_birth_request  ?StructureBirthRequest
	birth_requests      []StructureBirthRequest
	pending_child_specs map[string]ChildTemplateSpec
	child_specs         map[string]ChildTemplateSpec
	child_model_paths   map[string]string
}

// new_expert_registry builds a registry over the given experts.
fn new_expert_registry(experts map[string]SceneExpert) ExpertRegistry {
	return ExpertRegistry{
		experts:             experts
		gate:                new_structure_gate()
		pending_child_specs: map[string]ChildTemplateSpec{}
		child_specs:         map[string]ChildTemplateSpec{}
		child_model_paths:   map[string]string{}
	}
}

// default_expert_registry builds the three base structure-family experts.
fn default_expert_registry() ExpertRegistry {
	mut experts := map[string]SceneExpert{}
	for name in ['single', 'layered', 'composite'] {
		repl := if name == 'single' { 8 } else { 1 }
		cfg := InverseConfig{
			scene_family: name
			replicates:   repl
		}
		experts[name] = SceneExpert{
			name: name
			app:  new_inverse_app(cfg)
		}
	}
	return new_expert_registry(experts)
}

// lineages returns the expert lineage table.
fn (r ExpertRegistry) lineages() map[string]TemplateLineage {
	mut out := map[string]TemplateLineage{}
	for name, expert in r.experts {
		out[name] = expert.lineage()
	}
	return out
}

// children_of returns the registered experts directly inheriting from parent.
fn (r ExpertRegistry) children_of(parent string) []string {
	mut out := []string{}
	for name, lin in r.lineages() {
		if lin.parent_family == parent {
			out << name
		}
	}
	out.sort()
	return out
}

// enable_child_template_learning enables the birth→pending-spec learner.
fn (mut r ExpertRegistry) enable_child_template_learning() {
	r.child_workflow = new_child_template_workflow()
}

// default_manifest_path returns the default registry manifest path.
fn default_manifest_path(artifacts string) string {
	root := if artifacts != '' { artifacts } else { 'artifacts' }
	return os.join_path(root, 'registry_manifest.json')
}

// save_manifest writes the dynamic children + pending specs to path.
fn (mut r ExpertRegistry) save_manifest(path string) string {
	mut out := if path != '' { path } else { r.manifest_path }
	if out == '' {
		out = default_manifest_path('artifacts')
	}
	mut children := []RegisteredChildTemplate{}
	for name, spec in r.child_specs {
		children << RegisteredChildTemplate{
			spec:       spec
			model_path: r.child_model_paths[name]
		}
	}
	mut pending := []ChildTemplateSpec{}
	for _, spec in r.pending_child_specs {
		pending << spec
	}
	rm_save(RegistryManifest{
		children: children
		pending:  pending
	}, out)
	r.manifest_path = out
	return out
}

// load_manifest restores pending specs and trained child experts.
fn (mut r ExpertRegistry) load_manifest(path string, artifacts string, missing_ok bool) ! {
	manifest := rm_load(path)
	r.manifest_path = path
	for spec in manifest.pending {
		if spec.name !in r.experts {
			r.pending_child_specs[spec.name] = spec
		}
	}
	for child in manifest.children {
		spec := child.spec
		cb := ccf_build(spec)
		cfg := InverseConfig{
			scene_family: spec.family
		}
		app := new_inverse_app_cb(cfg, cb)
		model_path := if child.model_path != '' {
			child.model_path
		} else {
			app.default_model_path(artifacts)
		}
		if !os.exists(model_path) {
			if missing_ok {
				continue
			}
			return error('子模板 ${spec.name} 缺模型: ${model_path}')
		}
		r.experts[spec.name] = SceneExpert{
			name: spec.name
			app:  app
			net:  MixtureSPN{}
		}
		r.child_specs[spec.name] = spec
		r.child_model_paths[spec.name] = model_path
	}
}

// observe_birth_request records a request and learns pending child specs.
fn (mut r ExpertRegistry) observe_birth_request(request StructureBirthRequest) []ChildTemplateSpec {
	r.last_birth_request = request
	r.birth_requests << request
	if r.child_workflow == none {
		return []
	}
	w := r.child_workflow or { return []ChildTemplateSpec{} }
	specs := w.ctw_learn(r.birth_requests, r.lineages())
	mut new_specs := []ChildTemplateSpec{}
	for spec in specs {
		if spec.name !in r.experts && spec.name !in r.pending_child_specs {
			r.pending_child_specs[spec.name] = spec
			new_specs << spec
		}
	}
	return new_specs
}

// register adds an already-trained expert.
fn (mut r ExpertRegistry) register(name string, cfg InverseConfig, expert SceneExpert, artifacts string) SceneExpert {
	r.experts[name] = expert
	return expert
}

// train_and_register registers an expert (the training loop is invoked by the
// CLI's InverseApp.run; registration itself uses a placeholder net, matching the
// Python tests which monkeypatch InverseApp.run to a no-op).
fn (mut r ExpertRegistry) train_and_register(name string, cfg InverseConfig, artifacts string, codebook ?SceneFamily) SceneExpert {
	if cb := codebook {
		app := new_inverse_app_cb(cfg, cb)
		e := SceneExpert{
			name: name
			app:  app
			net:  MixtureSPN{}
		}
		r.experts[name] = e
		return e
	}
	app := new_inverse_app(cfg)
	e := SceneExpert{
		name: name
		app:  app
		net:  MixtureSPN{}
	}
	r.experts[name] = e
	return e
}

// confirm_child_template materialises/trains/registers a pending child spec.
fn (mut r ExpertRegistry) confirm_child_template(name string, artifacts string) ChildTemplateRegistration {
	if r.child_workflow == none {
		panic('尚未启用 child template learning')
	}
	if name !in r.pending_child_specs {
		panic('没有 pending 子模板: ${name}')
	}
	spec := r.pending_child_specs[name]
	cb := ccf_build(spec)
	cfg := InverseConfig{
		scene_family: spec.family
	}
	expert := r.train_and_register(spec.name, cfg, artifacts, cb)
	r.pending_child_specs.delete(name)
	r.child_specs[name] = spec
	r.child_model_paths[name] = expert.app.default_model_path(artifacts)
	return ChildTemplateRegistration{
		spec:         spec
		codebook_cls: cb
		expert:       expert
	}
}

// decide fuses all experts on one frame pair via the render-residual gate.
fn (mut r ExpertRegistry) decide(fl mlx.Array, fr mlx.Array) GenericStructureDecision {
	mut estimates := map[string]StructuredHypothesis{}
	for name, expert in r.experts {
		estimates[name] = expert.reconstruct(fl, fr)
	}
	return r.gate.decide(estimates, fl, fr)
}
