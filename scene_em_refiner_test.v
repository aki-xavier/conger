module conger

// scene_em_refiner_test.v — geometry↔lighting ECM refinement black-box tests.

import math

import cga

import mlx

fn em_err(a []f64, b []f64) f64 {
	mut s := 0.0
	for i in 0 .. a.len {
		d := a[i] - b[i]
		s += d * d
	}
	return math.sqrt(s / f64(a.len))
}

fn em_render_true(cb Codebook, mut renderer cga.Renderer, cam_l cga.PerspectiveCamera, cam_r cga.PerspectiveCamera, true_geom []f64) (mlx.Array, mlx.Array) {
	prm := [0.0, true_geom[0], true_geom[1], true_geom[2], true_geom[3], 2.0,
		0.0, 1.0]
	sc := cb.to_scene(prm)
	return renderer.render(sc, cam_l), renderer.render(sc, cam_r)
}

fn test_scene_em_refines_geometry_toward_truth() {
	cb := new_codebook(InverseConfig{})
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	true_geom := [72.0, 72.0, 0.45, 3.2]
	fl, fr := em_render_true(cb, mut renderer, cam_l, cam_r, true_geom)
	init := [68.0, 76.0, 0.52, 3.0]
	mut refiner := new_scene_em_refiner(cb, 0, fl, fr, 3, [false, false, true, true])
	mut loop := EMLoop[SceneEMRefiner, FramePair, mlx.Array]{
		model: refiner
		max_iters: 4
		tol: 0.0
	}
	result := loop.run(FramePair{fl: fl, fr: fr}, init)
	got := result.params
	assert em_err(got, true_geom) < em_err(init, true_geom)
}

fn test_scene_em_freezes_sz_by_default() {
	cb := new_codebook(InverseConfig{})
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	true_geom := [72.0, 72.0, 0.45, 3.2]
	fl, fr := em_render_true(cb, mut renderer, cam_l, cam_r, true_geom)
	init := [68.0, 76.0, 0.52, 3.0]
	mut refiner := new_scene_em_refiner(cb, 0, fl, fr, 3, [false, false, true, true])
	mut loop := EMLoop[SceneEMRefiner, FramePair, mlx.Array]{
		model: refiner
		max_iters: 4
		tol: 0.0
	}
	result := loop.run(FramePair{fl: fl, fr: fr}, init)
	got := result.params
	assert got[2] == init[2] && got[3] == init[3]
	uv_err := math.sqrt((got[0] - true_geom[0]) * (got[0] - true_geom[0]) + (got[1] -
		true_geom[1]) * (got[1] - true_geom[1]))
	init_uv_err := math.sqrt((init[0] - true_geom[0]) * (init[0] - true_geom[0]) +
		(init[1] - true_geom[1]) * (init[1] - true_geom[1]))
	assert uv_err < init_uv_err
}

fn new_quadratic_refiner(freeze []bool) SceneEMRefiner {
	mut r := new_scene_em_refiner(new_codebook(InverseConfig{}), 0,
		mlx.zeros([8, 8, 3], .float32), mlx.zeros([8, 8, 3], .float32), 1, freeze)
	r.appearances = [[0, 0, 0]!]
	r.target = [10.0, 20.0, 0.5, 1.0]
	r.use_quadratic = true
	return r
}

fn test_scene_em_maximize_skips_frozen_dims() {
	init := [0.0, 0.0, 0.3, 0.7]
	mut refiner := new_quadratic_refiner([false, false, true, true])
	q := refiner.responsibilities(init, FramePair{fl: mlx.zeros([8, 8, 3], .float32), fr: mlx.zeros([8, 8, 3], .float32)}, 1.0)
	got := refiner.maximize(q, FramePair{fl: mlx.zeros([8, 8, 3], .float32), fr: mlx.zeros([8, 8, 3], .float32)}, init, 0.0)
	assert got[2] == 0.3 && got[3] == 0.7
	assert got[0] > 0.0 && got[1] > 0.0

	mut ref_full := new_quadratic_refiner([false, false, false, false])
	qf := ref_full.responsibilities(init, FramePair{fl: mlx.zeros([8, 8, 3], .float32), fr: mlx.zeros([8, 8, 3], .float32)}, 1.0)
	gotf := ref_full.maximize(qf, FramePair{fl: mlx.zeros([8, 8, 3], .float32), fr: mlx.zeros([8, 8, 3], .float32)}, init, 0.0)
	assert gotf[2] != 0.3
}
