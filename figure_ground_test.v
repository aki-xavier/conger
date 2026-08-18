module conger

// figure_ground_test.v — FigureGroundModel (segmentation↔pose) black-box test.
import math

fn test_figure_ground_recovers_pose_and_intensities() {
	model := new_figure_ground_model(100, 0.03, 0.02, 0.02, 0.05)
	gt := [0.5, 0.2, 2.0, 0.5]
	obs := model.sample(gt, 0)

	init := [0.35, 0.3, 1.0, 1.0]
	mut loop := EMLoop[FigureGroundModel, []f64, []f64]{
		model:     model
		max_iters: 30
		tol:       1e-10
	}
	result := loop.run(obs, init)
	for i in 0 .. 4 {
		assert math.abs(result.params[i] - gt[i]) < 0.05
	}
}
