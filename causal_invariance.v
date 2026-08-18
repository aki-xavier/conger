module conger

// causal_invariance.v — held-out lighting probe + marginal estimation +
// invariance scoring (V port of src/causal_invariance.py).
import cga
import mlx

struct LightingHoldout {
	train_colors   []int
	train_dirs     []int
	holdout_colors []int
	holdout_dirs   []int
}

fn lighting_holdout_split(n_colors int, n_dirs int, holdout_color int, holdout_dir int) LightingHoldout {
	mut tc := []int{}
	mut td := []int{}
	for c in 0 .. n_colors {
		if c != holdout_color {
			tc << c
		}
	}
	for d in 0 .. n_dirs {
		if d != holdout_dir {
			td << d
		}
	}
	return LightingHoldout{
		train_colors:   tc
		train_dirs:     td
		holdout_colors: [holdout_color]
		holdout_dirs:   [holdout_dir]
	}
}

fn (h LightingHoldout) in_support(lcol int, ldir int) bool {
	return lcol in h.train_colors && ldir in h.train_dirs
}

fn (h LightingHoldout) holdout(lcol int, ldir int) bool {
	return lcol in h.holdout_colors || ldir in h.holdout_dirs
}

// invariance_score returns the worst-group accuracy (bottleneck).
fn invariance_score(group_accuracies []f64) f64 {
	if group_accuracies.len == 0 {
		return 0.0
	}
	mut m := 1e18
	for v in group_accuracies {
		if v < m {
			m = v
		}
	}
	return m
}

struct InvarianceReport {
	factor              string
	in_support_accuracy f64
	holdout_accuracy    f64
	invariance_score    f64
	gap                 f64
	per_group_accuracy  map[string]f64
	n_groups            int
}

// summarize_invariance aggregates (true, pred) pairs grouped by (lcol, ldir).
fn summarize_invariance(groups map[string][][2]int, factor string, holdout LightingHoldout) InvarianceReport {
	mut per_group := map[string]f64{}
	mut in_support := []f64{}
	mut held_out := []f64{}
	for key, pairs in groups {
		parts := key.split(',')
		lcol := parts[0].int()
		ldir := parts[1].int()
		mut correct := 0
		for p in pairs {
			if p[0] == p[1] {
				correct++
			}
		}
		acc := if pairs.len > 0 { f64(correct) / f64(pairs.len) } else { 0.0 }
		per_group[key] = acc
		if holdout.in_support(lcol, ldir) {
			in_support << acc
		} else {
			held_out << acc
		}
	}
	in_acc := mean_f64s(in_support)
	out_acc := mean_f64s(held_out)
	mut min_acc := 1.0
	for _, v in per_group {
		if v < min_acc {
			min_acc = v
		}
	}
	return InvarianceReport{
		factor:              factor
		in_support_accuracy: in_acc
		holdout_accuracy:    out_acc
		invariance_score:    min_acc
		gap:                 in_acc - out_acc
		per_group_accuracy:  per_group
		n_groups:            per_group.len
	}
}

fn mean_f64s(a []f64) f64 {
	if a.len == 0 {
		return 0.0
	}
	mut s := 0.0
	for v in a {
		s += v
	}
	return s / f64(a.len)
}

// invariance_probe_run is the end-to-end render + analysis-by-synthesis +
// marginalisation invariance probe.
fn invariance_probe_run(codebook Codebook, scenes [][]f64, holdout LightingHoldout, factor string, mut renderer cga.Renderer, cam_l cga.PerspectiveCamera, cam_r cga.PerspectiveCamera) InvarianceReport {
	mut groups := map[string][][2]int{}
	for row in scenes {
		scene := codebook.to_scene(row)
		fl := renderer.render(scene, cam_l)
		fr := renderer.render(scene, cam_r)
		base := [row[0], row[1], row[2], row[3], row[4]]
		_, _, score_arr := sr_refine_appearance(codebook, base, fl, fr, mut renderer, cam_l, cam_r)
		temperature := fmax2(2.0 * f64(score_arr.min().item_f32()), 1.0)
		logp := score_arr.negative().divide(mlx.f32_scalar(f32(temperature)))
		posterior := logp.subtract(logp.logsumexp()).exp()
		lcol := int(row[6])
		ldir := int(row[7])
		key := '${lcol},${ldir}'
		mut true_val := int(row[5])
		if factor == 'lcol' {
			true_val = int(row[6])
		}
		if factor == 'ldir' {
			true_val = int(row[7])
		}
		marginal := sr_marginal_appearance(posterior, factor)
		pred := marginal.argmax().item_i32()
		if key !in groups {
			groups[key] = [][2]int{}
		}
		groups[key] << [true_val, pred]!
	}
	return summarize_invariance(groups, factor, holdout)
}
