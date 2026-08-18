module conger

// scene_em_refiner.v — single-object geometry↔lighting ECM refinement
// (V port of src/scene_em_refiner.py, a GenericEM instance).
import cga
import mlx

// FramePair is the observation (left/right rendered frames).
struct FramePair {
	fl mlx.Array
	fr mlx.Array
}

struct SceneEMRefiner {
	codebook        Codebook
	kind            int
	fl              mlx.Array
	fr              mlx.Array
	appearance_topk int
	deltas          []f64
	freeze          []bool
	wl              mlx.Array
	wr              mlx.Array
mut:
	appearances   [][3]int
	target        []f64
	use_quadratic bool
	renderer      cga.Renderer
	cam_l         cga.PerspectiveCamera
	cam_r         cga.PerspectiveCamera
}

// new_scene_em_refiner builds the rendering-backed refiner.
fn new_scene_em_refiner(codebook Codebook, kind int, fl mlx.Array, fr mlx.Array, appearance_topk int, freeze []bool) SceneEMRefiner {
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	mut apps := [][3]int{}
	for hue in 0 .. n_hue {
		for lcol in 0 .. light_colors_len {
			for ldir in 0 .. light_dirs_len {
				apps << [hue, lcol, ldir]!
			}
		}
	}
	return SceneEMRefiner{
		codebook:        codebook
		kind:            kind
		fl:              fl
		fr:              fr
		appearance_topk: appearance_topk
		deltas:          [2.0, 2.0, 0.05, 0.1]
		freeze:          freeze
		wl:              foreground_weights(fl)
		wr:              foreground_weights(fr)
		appearances:     apps
		renderer:        renderer
		cam_l:           cam_l
		cam_r:           cam_r
	}
}

// residual returns the render residual (or synthetic quadratic when overridden).
fn (mut r SceneEMRefiner) residual(geometry []f64, appearance [3]int) f64 {
	if r.use_quadratic {
		mut s := 0.0
		for i, g in geometry {
			d := g - r.target[i]
			s += d * d
		}
		return s
	}
	u := geometry[0]
	v := geometry[1]
	s_ := geometry[2]
	z := geometry[3]
	prm := [f64(r.kind), u, v, s_, z, f64(appearance[0]), f64(appearance[1]), f64(appearance[2])]
	sc := r.codebook.to_scene(prm)
	cl := r.renderer.render(sc, r.cam_l)
	cr := r.renderer.render(sc, r.cam_r)
	return 0.5 * (sr_masked_mse(r.fl, cl, r.wl) + sr_masked_mse(r.fr, cr, r.wr))
}

// responsibilities runs the E step: q(A|G) over appearance candidates.
fn (mut r SceneEMRefiner) responsibilities(geometry []f64, observation FramePair, temperature f64) mlx.Array {
	mut scores := []f32{}
	for a in r.appearances {
		scores << f32(r.residual(geometry, a))
	}
	score_arr := mlx.array_f32(scores, [scores.len])
	t := fmax2(2.0 * f64(score_arr.min().item_f32()), 1.0) * temperature
	logp := score_arr.multiply(mlx.f32_scalar(f32(-1.0 / t)))
	return logp.subtract(logp.logsumexp()).exp()
}

// maximize runs the M step: coordinate search over unfrozen geometry dims.
fn (mut r SceneEMRefiner) maximize(q mlx.Array, observation FramePair, geometry []f64, damping f64) []f64 {
	ov := q.argsort().data_i32()
	mut order := []int{}
	for i := q.dim(0) - 1; i >= 0 && order.len < r.appearance_topk; i-- {
		order << ov[i]
	}
	qf := q.data_f32()
	mut qsum := 0.0
	for j in order {
		qsum += f64(qf[j])
	}
	mut top_q := []f64{}
	for j in order {
		top_q << f64(qf[j]) / qsum
	}
	mut cur := geometry.clone()
	for i, delta in r.deltas {
		if r.freeze[i] {
			continue
		}
		mut best := expected_residual(mut r, order, top_q, cur)
		for sign in [-1.0, 1.0] {
			mut cand := cur.clone()
			cand[i] += sign * delta
			e := expected_residual(mut r, order, top_q, cand)
			if e < best {
				best = e
				cur = cand.clone()
			}
		}
	}
	if damping > 0.0 {
		for i in 0 .. cur.len {
			cur[i] = (1.0 - damping) * cur[i] + damping * geometry[i]
		}
	}
	return cur
}

// log_likelihood returns −min appearance residual.
fn (mut r SceneEMRefiner) log_likelihood(geometry []f64, observation FramePair) f64 {
	mut best := 1e18
	for a in r.appearances {
		res := r.residual(geometry, a)
		if res < best {
			best = res
		}
	}
	return -best
}

// expected_residual computes the top-k appearance-weighted expected residual.
fn expected_residual(mut r SceneEMRefiner, order []int, top_q []f64, g []f64) f64 {
	mut s := 0.0
	for i, j in order {
		s += top_q[i] * r.residual(g, r.appearances[j])
	}
	return s
}
