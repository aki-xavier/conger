module conger

// occlusion_layers_test.v — OcclusionLayerModel (occlusion + depth order) tests.
import math

fn test_occlusion_em_recovers_layers_and_depth_order() {
	model := new_occlusion_layer_model(96, 0.4, 0.6, 0.02)
	gt := [0.5, 1.0, -0.3, -1.2]
	obs := model.render(gt, 0, 0)

	mut loop := EMLoop[OcclusionLayerModel, []f64, []f64]{
		model:     model
		max_iters: 40
		tol:       1e-10
	}
	result := loop.run(obs, [0.0, 0.0, 0.0, 0.0])
	for i in 0 .. 4 {
		assert math.abs(result.params[i] - gt[i]) < 0.15
	}
	assert result.responsibilities[0] > 0.5
}

fn test_occlusion_em_reverses_depth_order() {
	model := new_occlusion_layer_model(96, 0.4, 0.6, 0.02)
	gt := [0.4, 0.8, -0.6, 1.0]
	obs := model.render(gt, 1, 1)

	mut loop := EMLoop[OcclusionLayerModel, []f64, []f64]{
		model:     model
		max_iters: 40
		tol:       1e-10
	}
	result := loop.run(obs, [0.0, 0.0, 0.0, 0.0])
	for i in 0 .. 4 {
		assert math.abs(result.params[i] - gt[i]) < 0.15
	}
	assert result.responsibilities[0] < 0.5
}
