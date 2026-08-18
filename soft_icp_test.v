module conger

// soft_icp_test.v — SoftICPModel (EM-ICP) black-box tests.
import math

fn test_soft_icp_recovers_rigid_transform() {
	mut rng := new_rng(0)
	mut source := [][]f64{len: 20, init: []f64{len: 2}}
	for i in 0 .. 20 {
		source[i][0] = rng.normal(0.0, 0.6)
		source[i][1] = rng.normal(0.0, 0.6)
	}
	source[0][0] = 0.0
	source[0][1] = 0.0
	gt := [0.4, 0.5, -0.3]
	model := new_soft_icp_model(source, 0.03)
	obs := model.sample(gt, 0)

	mut loop := EMLoop[SoftICPModel, [][]f64, [][]f64]{
		model:     model
		max_iters: 40
		tol:       1e-10
	}
	result := loop.run(obs, [0.0, 0.0, 0.0])
	for i in 0 .. 3 {
		assert math.abs(result.params[i] - gt[i]) < 0.05
	}
	for i in 0 .. result.trajectory.len - 1 {
		assert result.trajectory[i + 1] >= result.trajectory[i] - 1e-6
	}
}

fn test_soft_icp_improves_from_perturbed_init() {
	mut rng := new_rng(1)
	mut source := [][]f64{len: 16, init: []f64{len: 2}}
	for i in 0 .. 16 {
		source[i][0] = rng.normal(0.0, 0.5)
		source[i][1] = rng.normal(0.0, 0.5)
	}
	gt := [-0.25, -0.4, 0.6]
	model := new_soft_icp_model(source, 0.02)
	obs := model.sample(gt, 1)

	init := [0.1, 0.1, -0.1]
	mut loop := EMLoop[SoftICPModel, [][]f64, [][]f64]{
		model:     model
		max_iters: 30
		tol:       1e-10
	}
	result := loop.run(obs, init)
	got_norm := norm3(sub3(result.params, gt))
	init_norm := norm3(sub3(init, gt))
	assert got_norm < init_norm
}

fn sub3(a []f64, b []f64) []f64 {
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(a []f64) f64 {
	return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
}
