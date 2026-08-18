module conger

// layered_reconstructor.v — two-object occlusion scene parameter decoding
// (V port of src/layered_reconstructor.py).
import mlx

const lrc_residual_scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

// lrc_split_cat splits the concatenated posterior (N,24) into six heads.
fn lrc_split_cat(cat_p mlx.Array) []mlx.Array {
	mut out := []mlx.Array{}
	mut lo := 0
	for sz in lcb_cat_sizes {
		out << cat_p.take_axis(mlx.arange(f64(lo), f64(lo + sz), 1.0, .int32), 1)
		lo += sz
	}
	return out
}

// lrc_proxy returns the kind-conditioned size proxy for one layer.
fn lrc_proxy(kind int, stats []f64, off int) f64 {
	st := arr32([stats[off + 2], 0.0, stats[off + 3]], [1, 3])
	return f64(sr_s_proxy(kind, st).item_f32())
}

// lrc_residual_targets returns physical targets − per-layer anchors, scaled.
fn lrc_residual_targets(t mlx.Array, classes mlx.Array, stats mlx.Array, scale []f64) mlx.Array {
	tv := t.data_f32()
	cv := classes.data_i32()
	sv := stats.data_f32()
	n := t.dim(0)
	mut out := []f32{}
	for i in 0 .. n {
		k0 := cv[i * 6 + 0]
		k1 := cv[i * 6 + 1]
		mut st := []f64{}
		for j in 0 .. 8 {
			st << f64(sv[i * 8 + j])
		}
		p0 := lrc_proxy(k0, st, 0)
		p1 := lrc_proxy(k1, st, 4)
		mut vals := []f64{}
		vals << f64(tv[i * 8 + 0]) - st[0]
		vals << f64(tv[i * 8 + 1]) - st[1]
		vals << f64(tv[i * 8 + 2]) - p0
		vals << f64(tv[i * 8 + 3]) - st[2]
		vals << f64(tv[i * 8 + 4]) - st[4]
		vals << f64(tv[i * 8 + 5]) - st[5]
		vals << f64(tv[i * 8 + 6]) - p1
		vals << f64(tv[i * 8 + 7]) - st[6]
		for j in 0 .. 8 {
			out << f32(vals[j] * scale[j])
		}
	}
	return mlx.array_f32(out, [n, 8])
}

// lrc_params decodes residual/read-through targets + discrete heads → 14-d params.
fn lrc_params(t_pred mlx.Array, cat_p mlx.Array, stats mlx.Array, scale []f64) [][]f64 {
	probs := lrc_split_cat(cat_p)
	mut pvals := [][]int{}
	for p in probs {
		pvals << p.argmax_axis(1, false).astype(.int32).data_i32()
	}
	tv := t_pred.data_f32()
	sv := stats.data_f32()
	n := t_pred.dim(0)
	mut rows := [][]f64{len: n}
	for i in 0 .. n {
		mut st := []f64{}
		for j in 0 .. 8 {
			st << f64(sv[i * 8 + j])
		}
		mut r := []f64{len: 8}
		for j in 0 .. 8 {
			r[j] = f64(tv[i * 8 + j]) * scale[j]
		}
		k0 := pvals[0][i]
		k1 := pvals[1][i]
		h0 := pvals[2][i]
		h1 := pvals[3][i]
		lcol := pvals[4][i]
		ldir := pvals[5][i]
		s0 := fmax2(r[2] + lrc_proxy(k0, st, 0), s_floor)
		s1 := fmax2(r[6] + lrc_proxy(k1, st, 4), s_floor)
		mut z0 := r[3] + st[2]
		mut z1 := r[7] + st[6]
		if z0 < z_min {
			z0 = z_min
		}
		if z0 > z_max {
			z0 = z_max
		}
		if z1 < z_min {
			z1 = z_min
		}
		if z1 > z_max {
			z1 = z_max
		}
		rows[i] = [f64(k0), r[0] + st[0], r[1] + st[1], s0, z0, f64(h0), f64(k1), r[4] + st[4],
			r[5] + st[5], s1, z1, f64(h1), f64(lcol), f64(ldir)]
	}
	return rows
}

// lrc_params_raw decodes read-through targets (no stereo anchors).
fn lrc_params_raw(t_pred mlx.Array, cat_p mlx.Array) [][]f64 {
	probs := lrc_split_cat(cat_p)
	mut pvals := [][]int{}
	for p in probs {
		pvals << p.argmax_axis(1, false).astype(.int32).data_i32()
	}
	tv := t_pred.data_f32()
	n := t_pred.dim(0)
	mut rows := [][]f64{len: n}
	for i in 0 .. n {
		k0 := pvals[0][i]
		k1 := pvals[1][i]
		h0 := pvals[2][i]
		h1 := pvals[3][i]
		lcol := pvals[4][i]
		ldir := pvals[5][i]
		rows[i] = [f64(k0), f64(tv[i * 8 + 0]), f64(tv[i * 8 + 1]), f64(tv[i * 8 + 2]),
			f64(tv[i * 8 + 3]), f64(h0), f64(k1), f64(tv[i * 8 + 4]), f64(tv[i * 8 + 5]),
			f64(tv[
				i * 8 + 6]), f64(tv[i * 8 + 7]), f64(h1), f64(lcol), f64(ldir)]
	}
	return rows
}

// lrc_targets_from_params maps 14-d params → 8-d continuous targets.
fn lrc_targets_from_params(params [][]f64) mlx.Array {
	mut flat := []f32{}
	for p in params {
		for j in lcb_target_idx {
			flat << f32(p[j])
		}
	}
	return mlx.array_f32(flat, [params.len, 8])
}

// lrc_from_frames decodes a layered observation into a StructuredHypothesis
// (SPN posterior, no render refinement; the constrained-child variant uses the
// all-ones residual scale).
fn lrc_from_frames(app InverseApp, net MixtureSPN, fl mlx.Array, fr mlx.Array, rw_opt ?RieszWavelet, scale []f64) StructuredHypothesis {
	f, stats, _ := sr_frame_features(app, fl, fr, rw_opt)
	t, cat_p, r := net.predict(f)
	prm := lrc_params(t, cat_p, stats, scale)[0]
	cat_p0 := cat_p.take_axis(sel1(0), 0).squeeze_axis(0)
	lin := app.codebook.template_lineage()
	_, ent, novelty := sr_novelty_metrics_sized(cat_p0, r, none, lcb_cat_sizes)
	return StructuredHypothesis{
		scene:              app.codebook.to_scene(prm)
		params:             prm
		spn_posterior:      cat_p0
		geometry_family:    app.codebook.geometry_family()
		template_delta:     lin.delta
		candidate_params:   [prm]
		hypotheses:         [
			HypothesisCandidate{
				params:      prm
				probability: 1.0
			},
		]
		factor_sizes:       lcb_cat_sizes
		factor_indices:     [0, 6, 5, 11, 12, 13]
		responsibility_max: f64(r.max().item_f32())
		posterior_entropy:  ent
		complexity:         lin.complexity
		novelty_score:      novelty
	}
}

// lrc_split_cat splits a concatenated posterior (N, Σsizes) into per-head slices.
