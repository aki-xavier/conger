module conger

// scene_reconstructor.v — feature/frame-pair → full cga.Scene reconstruction
// (V port of src/scene_reconstructor.py; the InverseApp-driven `from_frames` path
// is added with the app module).

import math

import cga

import mlx

const cat_sizes_ = [n_kind, n_hue, light_colors_len, light_dirs_len]
const s_floor = 0.05
const z_min = 0.5
const z_max = cam_z - 0.5

struct SceneReconstructor {}

// novelty_metrics returns (responsibility novelty, posterior entropy, novelty score).
fn sr_novelty_metrics_sized(cat_p mlx.Array, r mlx.Array, render_residual ?f64, sizes []int) (f64, f64, f64) {
	max_r := f64(r.max().item_f32()) + 1e-12
	rn := -math.log(max_r) / math.log(f64(r.dim(1)))
	mut ent := 0.0
	mut lo := 0
	for sz in sizes {
		p := cat_p.take_axis(mlx.arange(f64(lo), f64(lo + sz), 1.0, .int32), 0)
		ent -= f64(p.multiply(p.maximum(mlx.f32_scalar(1e-12)).log()).sum().item_f32()) / math.log(f64(sz))
		lo += sz
	}
	post_ent := ent / f64(sizes.len)
	render_term := if rr := render_residual { math.log(1.0 + rr) } else { 0.0 }
	return rn, post_ent, rn + post_ent + render_term
}

// novelty_metrics uses the single-family factor sizes.
fn sr_novelty_metrics(cat_p mlx.Array, r mlx.Array, render_residual ?f64) (f64, f64, f64) {
	return sr_novelty_metrics_sized(cat_p, r, render_residual, cat_sizes_)
}

// sr_s_proxy returns the kind-conditioned size proxy for an int kind.
fn sr_s_proxy(kind int, stats mlx.Array) mlx.Array {
	sqrt_area := stats.take_axis(sel1(2), 1).sqrt()
	depth := mlx.f32_scalar(f32(cam_z)).subtract(stats.take_axis(sel1(0), 1))
	q := sqrt_area.multiply(depth).divide(mlx.f32_scalar(f32(fx)))
	coef := if kind == 2 { 0.5 } else { 1.0 / math.sqrt(math.pi) }
	return q.multiply(mlx.f32_scalar(f32(coef)))
}

// sr_s_proxy_arr returns the kind-conditioned size proxy for an array kind.
fn sr_s_proxy_arr(kind mlx.Array, stats mlx.Array) mlx.Array {
	sqrt_area := stats.take_axis(sel1(2), 1).sqrt()
	depth := mlx.f32_scalar(f32(cam_z)).subtract(stats.take_axis(sel1(0), 1))
	q := sqrt_area.multiply(depth).divide(mlx.f32_scalar(f32(fx)))
	coef := mlx.where(kind.astype(.int32).equal(mlx.int_scalar(2)), mlx.f32_scalar(0.5),
		mlx.f32_scalar(f32(1.0 / math.sqrt(math.pi))))
	return q.multiply(coef)
}

// sr_split_cat_argmax splits the concatenated posterior into per-factor argmax.
fn sr_split_cat_argmax(cat_p mlx.Array) []mlx.Array {
	mut out := []mlx.Array{}
	mut lo := 0
	for sz in cat_sizes_ {
		out << cat_p.take_axis(mlx.arange(f64(lo), f64(lo + sz), 1.0, .int32), 1).argmax_axis(1,
			false).astype(.int32)
		lo += sz
	}
	return out
}

// sr_params decodes model output into full scene params.
fn sr_params(t_pred mlx.Array, cat_p mlx.Array, stats mlx.Array) [][]f64 {
	probs := sr_split_cat_argmax(cat_p)
	s := t_pred.take_axis(sel1(2), 1).add(sr_s_proxy_arr(probs[0], stats)).maximum(mlx.f32_scalar(f32(s_floor)))
	z := t_pred.take_axis(sel1(3), 1).add(stats.take_axis(sel1(0), 1)).clip(mlx.f32_scalar(f32(z_min)),
		mlx.f32_scalar(f32(z_max)))
	n := t_pred.dim(0)
	kind := probs[0].data_i32()
	hue := probs[1].data_i32()
	lcol := probs[2].data_i32()
	ldir := probs[3].data_i32()
	u := t_pred.take_axis(sel1(0), 1).data_f32()
	v := t_pred.take_axis(sel1(1), 1).data_f32()
	sv := s.data_f32()
	zv := z.data_f32()
	mut rows := [][]f64{len: n}
	for i in 0 .. n {
		rows[i] = [f64(kind[i]), f64(u[i]), f64(v[i]), f64(sv[i]), f64(zv[i]),
			f64(hue[i]), f64(lcol[i]), f64(ldir[i])]
	}
	return rows
}

// sr_scenes builds cga.Scene objects for each param tuple.
fn sr_scenes(params [][]f64, cb Codebook) []cga.Scene {
	mut out := []cga.Scene{}
	for p in params {
		out << cb.to_scene(p)
	}
	return out
}

// sr_masked_mse returns the foreground-weighted RGB MSE.
fn sr_masked_mse(observed mlx.Array, candidate mlx.Array, weights mlx.Array) f64 {
	d := observed.take_axis(mlx.arange(0.0, 3.0, 1.0, .int32), -1).astype(.float32).subtract(candidate.take_axis(mlx.arange(0.0,
		3.0, 1.0, .int32), -1).astype(.float32))
	num := weights.multiply(d.multiply(d).sum_axis(2, false)).sum().item_f32()
	den := weights.sum().maximum(mlx.f32_scalar(1e-8)).item_f32()
	return f64(num) / f64(den)
}

// sr_appearance_candidates returns the 54 hue×lcol×ldir candidates.
fn sr_appearance_candidates(base []f64) [][]f64 {
	mut out := [][]f64{}
	for hue in 0 .. n_hue {
		for lcol in 0 .. light_colors_len {
			for ldir in 0 .. light_dirs_len {
				out << [base[0], base[1], base[2], base[3], base[4], f64(hue),
					f64(lcol), f64(ldir)]
			}
		}
	}
	return out
}

// sr_refine_appearance refines hue/lcol/ldir via render residuals.
fn sr_refine_appearance(cb Codebook, base []f64, fl mlx.Array, fr mlx.Array, mut renderer cga.Renderer, cam_l cga.PerspectiveCamera, cam_r cga.PerspectiveCamera) ([]f64, f64, mlx.Array) {
	wl := foreground_weights(fl)
	wr := foreground_weights(fr)
	candidates := sr_appearance_candidates(base)
	mut scores := []f32{}
	for prm in candidates {
		sc := cb.to_scene(prm)
		cl := renderer.render(sc, cam_l)
		cr := renderer.render(sc, cam_r)
		scores << f32(0.5 * (sr_masked_mse(fl, cl, wl) + sr_masked_mse(fr, cr, wr)))
	}
	score_arr := mlx.array_f32(scores, [scores.len])
	best_i := score_arr.argmin().item_i32()
	return candidates[best_i], f64(score_arr.data_f32()[best_i]), score_arr
}

// sr_refine_scene returns the top-k kind × appearance joint render posterior.
fn sr_refine_scene(cb Codebook, base []f64, kind_p mlx.Array, stats mlx.Array, fl mlx.Array, fr mlx.Array, kind_topk int, mut renderer cga.Renderer, cam_l cga.PerspectiveCamera, cam_r cga.PerspectiveCamera) ([]f64, [][]f64, mlx.Array, mlx.Array, f64) {
	ktopk := max_i(1, min_i(kind_topk, n_kind))
	ov := kind_p.argsort().data_i32()
	mut order := []int{}
	for i := kind_p.dim(0) - 1; i >= 0 && order.len < ktopk; i-- {
		order << ov[i]
	}
	s_resid := base[3] - f64(sr_s_proxy(int(base[0]), stats).item_f32())
	mut params := [][]f64{}
	mut score_blocks := []mlx.Array{}
	mut weights := []f32{}
	kp := kind_p.data_f32()
	for kk in order {
		kbase := [f64(kk), base[1], base[2], base[3], base[4], base[5], base[6],
			base[7]]
		_, _, block_scores := sr_refine_appearance(cb, kbase, fl, fr, mut renderer,
			cam_l, cam_r)
		for p in sr_appearance_candidates(kbase) {
			params << p
		}
		score_blocks << block_scores
		for _ in 0 .. block_scores.dim(0) {
			weights << kp[kk]
		}
	}
	score_arr := mlx.concatenate(score_blocks, 0)
	weight_arr := mlx.array_f32(weights, [weights.len]).maximum(mlx.f32_scalar(1e-12))
	temperature := fmax2(2.0 * f64(score_arr.min().item_f32()), 1.0)
	logp := score_arr.multiply(mlx.f32_scalar(f32(-1.0 / temperature))).add(weight_arr.log())
	posterior := logp.subtract(logp.logsumexp()).exp()
	mut calibrated := [][]f64{len: params.len}
	for i, p in params {
		ps := f64(sr_s_proxy(int(p[0]), stats).item_f32()) + s_resid
		calibrated[i] = [p[0], p[1], p[2], ps, p[4], p[5], p[6], p[7]]
	}
	best_i := posterior.argmax().item_i32()
	return calibrated[best_i], calibrated, score_arr, posterior, temperature
}

// sr_marginal_appearance marginalises the joint appearance posterior over nuisance factors.
fn sr_marginal_appearance(posterior mlx.Array, factor string) mlx.Array {
	p := posterior.reshape([n_hue, light_colors_len, light_dirs_len])
	if factor == 'hue' {
		return p.sum_axes([1, 2], false)
	}
	if factor == 'lcol' {
		return p.sum_axes([0, 2], false)
	}
	if factor == 'ldir' {
		return p.sum_axes([0, 1], false)
	}
	panic('unknown appearance factor: ${factor} (expected hue/lcol/ldir)')
}

// sr_rig returns the public training rig.
fn sr_rig() (cga.Renderer, cga.PerspectiveCamera, cga.PerspectiveCamera) {
	return make_renderer(stereo_base)
}

// sr_cat_sizes returns the single-family category sizes (optionally textured).
fn sr_cat_sizes(n_textures int) []int {
	if n_textures > 0 {
		return [n_kind, n_hue, light_colors_len, light_dirs_len, n_textures]
	}
	return [n_kind, n_hue, light_colors_len, light_dirs_len]
}

// sr_physical_targets maps residual targets back to physical [u,v,s,z].
fn sr_physical_targets(t_pred mlx.Array, stats mlx.Array, kind mlx.Array) mlx.Array {
	c0 := t_pred.take_axis(mlx.arange(0.0, 2.0, 1.0, .int32), 1)
	c2 := t_pred.take_axis(sel1(2), 1).squeeze_axis(1).add(sr_s_proxy_arr(kind, stats)).expand_dims(1)
	c3 := t_pred.take_axis(sel1(3), 1).squeeze_axis(1).add(stats.take_axis(sel1(0),
		1).squeeze_axis(1)).expand_dims(1)
	return mlx.concatenate([c0, c2, c3], 1)
}

// sr_targets_from_params maps full scene params → physical [u,v,s,z] targets.
fn sr_targets_from_params(params [][]f64) mlx.Array {
	mut flat := []f32{}
	for p in params {
		flat << f32(p[1])
		flat << f32(p[2])
		flat << f32(p[3])
		flat << f32(p[4])
	}
	return mlx.array_f32(flat, [params.len, 4])
}

// sr_residual_targets maps physical targets → kind/size-proxy residuals.
fn sr_residual_targets(t_tr mlx.Array, c_tr mlx.Array, s_tr mlx.Array) mlx.Array {
	kind := c_tr.take_axis(sel1(0), 1).squeeze_axis(1)
	c0 := t_tr.take_axis(mlx.arange(0.0, 2.0, 1.0, .int32), 1)
	c2 := t_tr.take_axis(sel1(2), 1).squeeze_axis(1).subtract(sr_s_proxy_arr(kind, s_tr)).expand_dims(1)
	c3 := t_tr.take_axis(sel1(3), 1).squeeze_axis(1).subtract(s_tr.take_axis(sel1(0),
		1).squeeze_axis(1)).expand_dims(1)
	return mlx.concatenate([c0, c2, c3], 1)
}

// sr_em_refine runs the (default-off) geometry↔lighting ECM refinement.
fn sr_em_refine(app InverseApp, prm []f64, fl mlx.Array, fr mlx.Array) ([]f64, []f64) {
	if !app.cfg.em_refine {
		return prm, []f64{}
	}
	kind := int(prm[0])
	fz := app.cfg.em_freeze_sz
	cb := app.codebook as Codebook
	mut refiner := new_scene_em_refiner(cb, kind, fl, fr, app.cfg.em_appearance_topk,
		[false, false, fz, fz])
	mut loop := EMLoop[SceneEMRefiner, FramePair, mlx.Array]{
		model: refiner
		max_iters: app.cfg.em_max_iters
	}
	res := loop.run(FramePair{
		fl: fl
		fr: fr
	}, prm[1..5])
	return [f64(kind), res.params[0], res.params[1], res.params[2], res.params[3],
		prm[5], prm[6], prm[7]], res.trajectory
}

// sr_frame_features returns (model feature (1,V), stereo stats, Riesz workspace).
fn sr_frame_features(app InverseApp, fl mlx.Array, fr mlx.Array, rw_opt ?RieszWavelet) (mlx.Array, mlx.Array, RieszWavelet) {
	vec, rw := app.extractor.of_frame(fl, rw_opt)
	fam := app.codebook.geometry_family()
	if fam == 'lateral' {
		stat := lgc_estimate(fl, fr)
		scaled := sl_scaled(arr32(stat, [1, 8])).take_axis(sel1(0), 0).squeeze_axis(0)
		vec2 := mlx.concatenate([vec, scaled], 0)
		return vec2.expand_dims(0), arr32(stat, [1, 8]), rw
	}
	if fam == 'composite' {
		stat := cg_estimate(fl, fr)
		scaled := sl_scaled(arr32(stat, [1, 8])).take_axis(sel1(0), 0).squeeze_axis(0)
		vec2 := mlx.concatenate([vec, scaled], 0)
		return vec2.expand_dims(0), arr32(stat, [1, 8]), rw
	}
	if fam == 'layered' {
		stat := sl_estimate(fl, fr)
		scaled := sl_scaled(arr32(stat, [1, 8])).take_axis(sel1(0), 0).squeeze_axis(0)
		vec2 := mlx.concatenate([vec, scaled], 0)
		return vec2.expand_dims(0), arr32(stat, [1, 8]), rw
	}
	z_hat, d, area := StereoDepth{}.estimate(fl, fr)
	vec2 := mlx.concatenate([vec, arr32([z_hat, area / 1000.0], [2])], 0)
	return vec2.expand_dims(0), arr32([z_hat, d, area], [1, 3]), rw
}

// sr_from_frames decodes a single-family observation into a StructuredHypothesis.
fn sr_from_frames(app InverseApp, net MixtureSPN, fl mlx.Array, fr mlx.Array, rw_opt ?RieszWavelet, refine bool, kind_topk int) StructuredHypothesis {
	f, stats, _ := sr_frame_features(app, fl, fr, rw_opt)
	t, cat_p, r := net.predict(f)
	prm := sr_params(t, cat_p, stats)[0]
	cat_p0 := cat_p.take_axis(sel1(0), 0).squeeze_axis(0)
	lin := app.codebook.template_lineage()
	_, ent0, nov0 := sr_novelty_metrics(cat_p0, r, none)
	if !refine {
		return StructuredHypothesis{
			scene: app.codebook.to_scene(prm)
			params: prm
			spn_posterior: cat_p0
			geometry_family: app.codebook.geometry_family()
			template_delta: lin.delta
			candidate_params: [prm]
			hypotheses: [HypothesisCandidate{
				params: prm
				probability: 1.0
			}]
			factor_sizes: [n_kind, n_hue, light_colors_len, light_dirs_len]
			factor_indices: [0, 5, 6, 7]
			responsibility_max: f64(r.max().item_f32())
			posterior_entropy: ent0
			complexity: lin.complexity
			novelty_score: nov0
		}
	}
	cb := app.codebook as Codebook
	mut renderer, cam_l, cam_r := sr_rig()
	kind_p := cat_p0.take_axis(mlx.arange(0.0, f64(n_kind), 1.0, .int32), 0)
	prm2, candidates, scores, posterior, temperature := sr_refine_scene(cb, prm, kind_p,
		stats, fl, fr, kind_topk, mut renderer, cam_l, cam_r)
	final_prm, em_traj := sr_em_refine(app, prm2, fl, fr)
	top := if candidates.len < 5 { candidates.len } else { 5 }
	order := posterior.negative().argsort().take(mlx.arange(0.0, f64(top), 1.0, .int32)).data_i32()
	mut hypotheses := []HypothesisCandidate{}
	for i in order {
		hypotheses << HypothesisCandidate{
			params: candidates[i]
			probability: f64(posterior.take(sel1(i)).item_f32())
			residual: f64(scores.take(sel1(i)).item_f32())
		}
	}
	best_residual := f64(scores.min().item_f32())
	_, ent, novelty := sr_novelty_metrics(cat_p0, r, best_residual)
	return StructuredHypothesis{
		scene: cb.to_scene(final_prm)
		params: final_prm
		spn_posterior: cat_p0
		geometry_family: app.codebook.geometry_family()
		template_delta: lin.delta
		candidate_params: candidates
		candidate_scores: scores
		candidate_posterior: posterior
		candidate_temperature: temperature
		hypotheses: hypotheses
		factor_sizes: [n_kind, n_hue, light_colors_len, light_dirs_len]
		factor_indices: [0, 5, 6, 7]
		responsibility_max: f64(r.max().item_f32())
		posterior_entropy: ent
		residual: best_residual
		complexity: lin.complexity
		novelty_score: novelty
		em_trajectory: em_traj
	}
}
