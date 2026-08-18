module conger

// inverse_app.v — inverse-rendering app wiring (V port of the config/codebook/
// data/cache surface of src/inverse_app.py; the full training loop is not part
// of the ported test surface).
import os
import mlx

// InverseApp holds the scene family + feature extractor + data builder wiring.
struct InverseApp {
	cfg       InverseConfig
	codebook  SceneFamily
	data      DataBuilder
	extractor FeatureExtractor
}

// default_codebook builds the base scene family for a config.
fn default_codebook(cfg InverseConfig) SceneFamily {
	fam := cfg.family()
	if fam == 'single' {
		return new_codebook(cfg)
	}
	if fam == 'layered' {
		return new_layered_codebook(cfg)
	}
	if fam == 'composite' {
		return new_composite_codebook(cfg)
	}
	panic('unknown scene_family: ${fam}')
}

// new_inverse_app builds the app with the default codebook for the config.
fn new_inverse_app(cfg InverseConfig) InverseApp {
	cb := default_codebook(cfg)
	ex := new_feature_extractor(cfg)
	return InverseApp{
		cfg:       cfg
		codebook:  cb
		data:      DataBuilder{
			cfg:       cfg
			codebook:  cb
			extractor: ex
		}
		extractor: ex
	}
}

// new_inverse_app_cb builds the app with an explicit (child) codebook.
fn new_inverse_app_cb(cfg InverseConfig, codebook SceneFamily) InverseApp {
	ex := new_feature_extractor(cfg)
	return InverseApp{
		cfg:       cfg
		codebook:  codebook
		data:      DataBuilder{
			cfg:       cfg
			codebook:  codebook
			extractor: ex
		}
		extractor: ex
	}
}

// layered_reconstructor returns 'constrained' for child layer templates (full
// residual learning) and 'layered' for the base family (anchor-only).
fn (app InverseApp) layered_reconstructor() string {
	if app.codebook.template_variant() != '' {
		return 'constrained'
	}
	return 'layered'
}

// default_model_path returns the model path for the current expert config.
fn (app InverseApp) default_model_path(artifacts string) string {
	root := if artifacts != '' { artifacts } else { 'artifacts' }
	prefix := match app.cfg.family() {
		'single' { 'spn_kindgeo' }
		'layered' { 'spn_layered_anchor' }
		else { 'spn_composite' }
	}
	return os.join_path(root, '${prefix}_${app.data.cache_tag()}.safetensors')
}

// run executes the full train → predict → refine → evaluate loop.
fn (app InverseApp) run(artifacts string) {
	cfg := app.cfg
	root := if artifacts != '' { artifacts } else { 'artifacts' }
	n_tr := app.codebook.n_combo() * cfg.replicates
	data := app.data.db_build(cfg.replicates)
	f_tr := data.f_tr
	p_tr := data.p_tr
	f_ti := data.f_ti
	p_ti := data.p_ti
	f_te := data.f_te
	p_te := data.p_te
	s_tr := data.s_tr
	s_ti := data.s_ti
	s_te := data.s_te

	mut t_tr := db_targets(p_tr)
	c_tr := db_scene_classes(p_tr)
	if cfg.family() == 'single' {
		t_tr = sr_residual_targets(t_tr, c_tr, s_tr)
	} else if cfg.family() == 'composite' {
		t_tr = lrc_residual_targets(t_tr, c_tr, s_tr, cr_residual_scale)
	} else {
		scale := if app.layered_reconstructor() == 'constrained' {
			lrc_residual_scale_constrained
		} else {
			lrc_residual_scale
		}
		t_tr = lrc_residual_targets(t_tr, c_tr, s_tr, scale)
	}

	model_path := app.default_model_path(root)
	stratum := c_tr.take_axis(sel1(0), 1).squeeze_axis(1)
	cat_sizes := if cfg.family() == 'single' { sr_cat_sizes(cfg.n_textures) } else { lcb_cat_sizes }
	mut net := MixtureSPN{}
	if os.exists(model_path) {
		net = load_mixture_spn(model_path)
		if net.f_mu.dim(0) < n_tr {
			net.add(f_tr.take_axis(mlx.arange(f64(net.f_mu.dim(0)), f64(n_tr), 1.0, .int32), 0), t_tr.take_axis(mlx.arange(f64(net.f_mu.dim(0)),
				f64(n_tr), 1.0, .int32), 0), stratum.take(mlx.arange(f64(net.f_mu.dim(0)),
				f64(n_tr), 1.0, .int32)), c_tr.take_axis(mlx.arange(f64(net.f_mu.dim(0)),
				f64(n_tr), 1.0, .int32), 0))
			net.save(model_path)
		}
	} else {
		net = fit_mixture_spn(f_tr, t_tr, stratum, cfg.sigma_rel_floor, c_tr, cat_sizes,
			cfg.basis_dim)
		net.save(model_path)
	}

	ti_raw, ci_p, _ := net.predict(f_ti)
	te_raw, ce_p, _ := net.predict(f_te)
	mut ti_pred := ti_raw
	mut te_pred := te_raw
	mut ci_pred := [][]f64{}
	mut ce_pred := [][]f64{}
	if cfg.family() == 'single' {
		ki0 :=
			ci_p.take_axis(mlx.arange(0.0, f64(n_kind), 1.0, .int32), 1).argmax_axis(1, false).astype(.int32)
		ke0 :=
			ce_p.take_axis(mlx.arange(0.0, f64(n_kind), 1.0, .int32), 1).argmax_axis(1, false).astype(.int32)
		ti_pred = sr_physical_targets(ti_raw, s_ti, ki0)
		te_pred = sr_physical_targets(te_raw, s_te, ke0)
		ci_pred = sr_params(ti_raw, ci_p, s_ti)
		ce_pred = sr_params(te_raw, ce_p, s_te)
		if cfg.refine_appearance {
			ci_pred = app.refine_scenes(ci_pred, ci_p, s_ti, p_ti)
			ce_pred = app.refine_scenes(ce_pred, ce_p, s_te, p_te)
			ti_pred = sr_targets_from_params(ci_pred)
			te_pred = sr_targets_from_params(ce_pred)
		}
	} else if cfg.family() == 'composite' {
		scale := cr_residual_scale
		ci_pred = lrc_params(ti_raw, ci_p, s_ti, scale)
		ce_pred = lrc_params(te_raw, ce_p, s_te, scale)
		ti_pred = lrc_targets_from_params(ci_pred)
		te_pred = lrc_targets_from_params(ce_pred)
		if cfg.refine_composite {
			ci_pred = app.refine_composite_scenes(ci_pred, ci_p, p_ti)
			ce_pred = app.refine_composite_scenes(ce_pred, ce_p, p_te)
			ti_pred = lrc_targets_from_params(ci_pred)
			te_pred = lrc_targets_from_params(ce_pred)
		}
	} else {
		scale := if app.layered_reconstructor() == 'constrained' {
			lrc_residual_scale_constrained
		} else {
			lrc_residual_scale
		}
		ci_pred = lrc_params(ti_raw, ci_p, s_ti, scale)
		ce_pred = lrc_params(te_raw, ce_p, s_te, scale)
		ti_pred = lrc_targets_from_params(ci_pred)
		te_pred = lrc_targets_from_params(ce_pred)
	}

	mi := Evaluator{}.report('插值', p_ti, ti_pred, ci_pred, p_tr)
	me := Evaluator{}.report('外推', p_te, te_pred, ce_pred, p_tr)
	app.self_check(mi, me)
}

// refine_scenes re-renders each single-family prediction via render residuals.
fn (app InverseApp) refine_scenes(scene_pred [][]f64, cat_p mlx.Array, stats mlx.Array, p_gt mlx.Array) [][]f64 {
	cb := app.codebook as Codebook
	mut renderer, cam_l, cam_r := sr_rig()
	mut out := [][]f64{}
	pg := p_gt.data_f32()
	cp := cat_p.data_f32()
	for i, prm in scene_pred {
		mut gt := []f64{len: p_gt.dim(1)}
		for j in 0 .. p_gt.dim(1) {
			gt[j] = f64(pg[i * p_gt.dim(1) + j])
		}
		scene_gt := cb.to_scene(gt)
		fl := renderer.render(scene_gt, cam_l)
		fr := renderer.render(scene_gt, cam_r)
		kind_p := mlx.array_f32(cp[i * cat_p.dim(1)..(i + 1) * cat_p.dim(1)], [
			cat_p.dim(1)]).take_axis(mlx.arange(0.0, f64(n_kind), 1.0, .int32), 0)
		st := mlx.array_f32(stats.data_f32()[i * stats.dim(1)..(i + 1) * stats.dim(1)], [
			1,
			stats.dim(1),
		])
		refined, _, _, _, _ := sr_refine_scene(cb, prm, kind_p, st, fl, fr, app.cfg.kind_topk, mut
			renderer, cam_l, cam_r)
		final_prm, _ := sr_em_refine(app, refined, fl, fr)
		out << final_prm
	}
	return out
}

// refine_composite_scenes re-renders each composite prediction.
fn (app InverseApp) refine_composite_scenes(scene_pred [][]f64, cat_p mlx.Array, p_gt mlx.Array) [][]f64 {
	cb := app.codebook as CompositeCodebook
	mut renderer, cam_l, cam_r := sr_rig()
	mut out := [][]f64{}
	pg := p_gt.data_f32()
	for i, prm in scene_pred {
		mut gt := []f64{len: p_gt.dim(1)}
		for j in 0 .. p_gt.dim(1) {
			gt[j] = f64(pg[i * p_gt.dim(1) + j])
		}
		scene_gt := cb.to_scene(gt)
		fl := renderer.render(scene_gt, cam_l)
		fr := renderer.render(scene_gt, cam_r)
		refined, _, _, _, _ := cr_refine_scene(cb, prm,
			cat_p.take_axis(sel1(i), 0).squeeze_axis(0), fl, fr, 2, 1, 1)
		out << refined
	}
	return out
}

// self_check asserts the single-family metric thresholds.
fn (app InverseApp) self_check(mi map[string]f64, _ map[string]f64) {
	if app.cfg.family() != 'single' {
		return
	}
	kind_floor := if app.cfg.refine_appearance { 0.65 } else { 0.45 }
	assert mi['kind'] > kind_floor
	assert mi['u_rmse'] < 9.0
	assert mi['v_rmse'] < 9.0
	assert mi['hue'] > 0.9
	assert mi['lcol'] > 0.85
	assert mi['ldir'] > 0.7
	assert mi['z_r2'] > 0.6
	s_fl := if app.cfg.refine_appearance { 0.4 } else { 0.2 }
	assert mi['s_r2'] > s_fl
}

// reconstruct_scene decodes a frame pair into a StructuredHypothesis.
fn (app InverseApp) reconstruct_scene(net MixtureSPN, fl mlx.Array, fr mlx.Array) StructuredHypothesis {
	fam := app.cfg.family()
	if fam == 'layered' {
		scale := if app.layered_reconstructor() == 'constrained' {
			lrc_residual_scale_constrained
		} else {
			lrc_residual_scale
		}
		return lrc_from_frames(app, net, fl, fr, none, scale)
	}
	if fam == 'composite' {
		return cr_from_frames(app, net, fl, fr, none, app.cfg.refine_composite)
	}
	return sr_from_frames(app, net, fl, fr, none, app.cfg.refine_appearance, app.cfg.kind_topk)
}
