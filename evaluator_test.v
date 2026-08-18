module conger

// evaluator_test.v — full Evaluator coverage (single + layered-composite).

import math

import mlx

fn test_evaluator_layered_report() {
	p_gt := arr32([
		0.0, 10.0, 20.0, 0.4, 3.0, 1.0, 1.0, 30.0, 40.0, 0.3, 2.5, 2.0, 0.0,
		1.0,
		0.0, 50.0, 60.0, 0.5, 3.5, 3.0, 2.0, 70.0, 80.0, 0.35, 2.8, 4.0, 1.0,
		2.0,
	], [2, 14])
	t_pred := arr32([
		10.0, 20.0, 0.4, 3.0, 30.0, 40.0, 0.3, 2.5,
		50.0, 60.0, 0.5, 3.5, 70.0, 80.0, 0.35, 2.8,
	], [2, 8])
	scene_pred := [
		[0.0, 10.0, 20.0, 0.4, 3.0, 1.0, 1.0, 30.0, 40.0, 0.3, 2.5, 2.0, 0.0,
			1.0],
		[0.0, 50.0, 60.0, 0.5, 3.5, 3.0, 2.0, 70.0, 80.0, 0.35, 2.8, 4.0, 1.0,
			2.0],
	]
	out := Evaluator{}.report('layered', p_gt, t_pred, scene_pred, p_gt)
	assert out['kind0'] == 1.0
	assert out['kind1'] == 1.0
	assert out['hue0'] == 1.0
	assert out['hue1'] == 1.0
	assert out['lcol'] == 1.0
	assert out['ldir'] == 1.0
	assert math.abs(out['u0_rmse']) < 1e-5
	assert math.abs(out['z1_rmse']) < 1e-5
	assert math.abs(out['u0_r2'] - 1.0) < 1e-5
	assert math.abs(out['z1_r2'] - 1.0) < 1e-5
}

fn test_evaluator_single_report() {
	p_gt := arr32([
		0.0, 10.0, 20.0, 0.4, 3.0, 1.0, 0.0, 1.0,
		1.0, 50.0, 60.0, 0.5, 3.5, 3.0, 1.0, 2.0,
	], [2, 8])
	t_pred := arr32([
		10.0, 20.0, 0.4, 3.0,
		50.0, 60.0, 0.5, 3.5,
	], [2, 4])
	scene_pred := [
		[0.0, 10.0, 20.0, 0.4, 3.0, 1.0, 0.0, 1.0],
		[1.0, 50.0, 60.0, 0.5, 3.5, 3.0, 1.0, 2.0],
	]
	out := Evaluator{}.report('single', p_gt, t_pred, scene_pred, p_gt)
	assert out['kind'] == 1.0
	assert out['hue'] == 1.0
	assert out['lcol'] == 1.0
	assert out['ldir'] == 1.0
	assert math.abs(out['u_rmse']) < 1e-5
	assert math.abs(out['z_rmse']) < 1e-5
	assert math.abs(out['u_r2'] - 1.0) < 1e-5
}
