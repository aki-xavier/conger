module conger

// composite_reconstructor.v — part-aware attached-composite parameter decoding
// (V port of src/composite_reconstructor.py).
import mlx

const cr_residual_scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

// cr_topk returns the top-k indices of a 1-D posterior.
fn cr_topk(p mlx.Array, k int) []int {
	kk := if k < 1 {
		1
	} else if k > p.dim(0) {
		p.dim(0)
	} else {
		k
	}
	return p.negative().argsort().take(mlx.arange(0.0, f64(kk), 1.0, .int32)).data_i32()
}

// cr_refine_scene re-renders top-k discrete candidates for the render posterior.
fn cr_refine_scene(codebook CompositeCodebook, base_params []f64, cat_p mlx.Array, fl mlx.Array, fr mlx.Array, kind_topk int, hue_topk int, light_topk int) ([]f64, [][]f64, mlx.Array, mlx.Array, f64) {
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	heads := lrc_split_cat(cat_p.expand_dims(0))
	mut hv := []mlx.Array{}
	for h in heads {
		hv << h.take_axis(sel1(0), 0).squeeze_axis(0)
	}
	tops := [
		cr_topk(hv[0], kind_topk),
		cr_topk(hv[1], kind_topk),
		cr_topk(hv[2], hue_topk),
		cr_topk(hv[3], hue_topk),
		cr_topk(hv[4], light_topk),
		cr_topk(hv[5], light_topk),
	]
	mut candidates := [][]f64{}
	mut scores := []f64{}
	mut weights := []f64{}
	wl := foreground_weights(fl)
	wr := foreground_weights(fr)
	for k0 in tops[0] {
		for k1 in tops[1] {
			for h0 in tops[2] {
				for h1 in tops[3] {
					for lc in tops[4] {
						for ld in tops[5] {
							prm := [f64(k0), base_params[1], base_params[2], base_params[3], base_params[4],
								f64(h0), f64(k1), base_params[7], base_params[8], base_params[9],
								base_params[10], f64(h1), f64(lc), f64(ld)]
							scene := codebook.to_scene(prm)
							cl := renderer.render(scene, cam_l)
							cr := renderer.render(scene, cam_r)
							score := 0.5 * (sr_masked_mse(fl, cl, wl) + sr_masked_mse(fr, cr, wr))
							candidates << prm
							scores << score
							weights << f64(hv[0].take(sel1(k0)).item_f32()) * f64(hv[1].take(sel1(k1)).item_f32()) * f64(hv[2].take(sel1(h0)).item_f32()) * f64(hv[3].take(sel1(h1)).item_f32()) * f64(hv[4].take(sel1(lc)).item_f32()) * f64(hv[5].take(sel1(ld)).item_f32())
						}
					}
				}
			}
		}
	}
	score_arr := arr32(scores, [scores.len])
	weight_arr := arr32(weights, [weights.len]).maximum(mlx.f32_scalar(1e-12))
	temperature := fmax2(2.0 * f64(score_arr.min().item_f32()), 1.0)
	logp := score_arr.negative().divide(mlx.f32_scalar(f32(temperature))).add(weight_arr.log())
	posterior := logp.subtract(logp.logsumexp()).exp()
	best_i := posterior.argmax().item_i32()
	return candidates[best_i], candidates, score_arr, posterior, temperature
}

// cr_from_frames decodes an attached-composite observation into a hypothesis.
fn cr_from_frames(app InverseApp, net MixtureSPN, fl mlx.Array, fr mlx.Array, rw_opt ?RieszWavelet, refine bool) StructuredHypothesis {
	f, stats, _ := sr_frame_features(app, fl, fr, rw_opt)
	t, cat_p, r := net.predict(f)
	mut prm := lrc_params(t, cat_p, stats, cr_residual_scale)[0]
	cat_p0 := cat_p.take_axis(sel1(0), 0).squeeze_axis(0)
	lin := app.codebook.template_lineage()
	mut candidates := [prm]
	mut scores := ?mlx.Array(none)
	mut posterior := ?mlx.Array(none)
	mut temperature := f64(0)
	mut hypotheses := [HypothesisCandidate{
		params:      prm
		probability: 1.0
	}]
	mut residual := ?f64(none)
	if refine {
		cb := app.codebook as CompositeCodebook
		prm2, cands, sc, post, temp := cr_refine_scene(cb, prm, cat_p0, fl, fr, 2, 1, 1)
		prm = prm2.clone()
		candidates = cands.clone()
		scores = sc
		posterior = post
		temperature = temp
		top := if candidates.len < 5 { candidates.len } else { 5 }
		order := post.negative().argsort().take(mlx.arange(0.0, f64(top), 1.0, .int32)).data_i32()
		hypotheses = []
		for i in order {
			hypotheses << HypothesisCandidate{
				params:      candidates[i]
				probability: f64(post.take(sel1(i)).item_f32())
				residual:    f64(sc.take(sel1(i)).item_f32())
			}
		}
		residual = f64(sc.min().item_f32())
	}
	_, ent, novelty := sr_novelty_metrics_sized(cat_p0, r, residual, lcb_cat_sizes)
	return StructuredHypothesis{
		scene:                 app.codebook.to_scene(prm)
		params:                prm
		spn_posterior:         cat_p0
		geometry_family:       app.codebook.geometry_family()
		template_delta:        lin.delta
		candidate_params:      candidates
		candidate_scores:      scores
		candidate_posterior:   posterior
		candidate_temperature: temperature
		hypotheses:            hypotheses
		factor_sizes:          lcb_cat_sizes
		factor_indices:        [0, 6, 5, 11, 12, 13]
		responsibility_max:    f64(r.max().item_f32())
		posterior_entropy:     ent
		residual:              residual or { 0.0 }
		complexity:            lin.complexity
		novelty_score:         novelty
	}
}
