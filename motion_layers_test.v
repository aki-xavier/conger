module conger

// motion_layers_test.v — MotionLayersModel (motion segmentation↔optical flow) test.

import math

fn test_motion_layers_recovers_velocities() {
	model := new_motion_layers_model(2, 64, 0.03, 4)
	gt := [0.2, 0.8]
	obs := model.sample(gt, 0)

	mut loop := EMLoop[MotionLayersModel, []f64, [][]f64]{
		model: model
		max_iters: 30
		tol: 1e-10
	}
	result := loop.run(obs, [0.3, 0.7])

	mut min_abs := 1e9
	mut max_abs := 0.0
	for i in 0 .. 2 {
		a := math.abs(result.params[i] - gt[i])
		if a < min_abs {
			min_abs = a
		}
		if a > max_abs {
			max_abs = a
		}
	}
	assert min_abs < 0.05
	assert max_abs < 0.05
}
