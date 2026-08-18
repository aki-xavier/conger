module conger

// generic_em_test.v — GenericEM + transparent-layer superposition EM probe.
import math

fn test_transparent_layer_em_recovers_layers() {
	n := 256
	alpha := linspace(0.08, 0.92, n)
	model := new_transparent_layer_model(alpha, 0.05)
	obs := model.sample([0.4, -0.6], 0)
	mut loop := EMLoop[TransparentLayerModel, []f64, []f64]{
		model:     model
		max_iters: 200
		tol:       1e-10
	}
	result := loop.run(obs, [0.0, 0.0])
	assert math.abs(result.params[0] - 0.4) < 0.03
	assert math.abs(result.params[1] - -0.6) < 0.03
	for i in 0 .. result.trajectory.len - 1 {
		assert result.trajectory[i + 1] >= result.trajectory[i] - 1e-6
	}
}

fn test_em_loop_temperature_damping_and_convergence() {
	n := 128
	alpha := linspace(0.1, 0.9, n)
	model := new_transparent_layer_model(alpha, 0.05)
	obs := model.sample([0.7, -0.3], 1)

	mut loop := EMLoop[TransparentLayerModel, []f64, []f64]{
		model:     model
		max_iters: 200
		tol:       1e-10
	}
	result := loop.run(obs, [0.0, 0.0])
	assert result.iterations < 200
	assert !math.is_nan(result.log_likelihood) && !math.is_inf(result.log_likelihood, 0)

	mut sharp := EMLoop[TransparentLayerModel, []f64, []f64]{
		model:       model
		max_iters:   200
		tol:         1e-10
		temperature: 0.7
	}
	sharp_res := sharp.run(obs, [0.0, 0.0])
	assert math.abs(sharp_res.params[0] - 0.7) < 0.05
	assert math.abs(sharp_res.params[1] - -0.3) < 0.05

	mut damped := EMLoop[TransparentLayerModel, []f64, []f64]{
		model:     model
		max_iters: 200
		tol:       1e-10
		damping:   0.3
	}
	damped_res := damped.run(obs, [0.0, 0.0])
	assert math.abs(damped_res.params[0] - 0.7) < 0.05
	assert math.abs(damped_res.params[1] - -0.3) < 0.05
}
