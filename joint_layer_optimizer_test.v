module conger

// joint_layer_optimizer_test.v — joint template/occlusion/disparity test.

import math

import mlx

fn test_joint_layer_optimization() {
	h := 144
	w := 144
	yy, xx := meshgrid_ij(h, w)
	front := xx.subtract(mlx.f32_scalar(87.0)).square().add(yy.subtract(mlx.f32_scalar(90.0)).square()).less_equal(mlx.f32_scalar(24.0 * 24.0))
	back_full := xx.subtract(mlx.f32_scalar(60.0)).abs().less_equal(mlx.f32_scalar(27.0)).logical_and(yy.subtract(mlx.f32_scalar(63.0)).abs().less_equal(mlx.f32_scalar(27.0)))
	back := back_full.logical_and(front.logical_not())
	fg := front.logical_or(back)
	mut disp := mlx.zeros([h, w], .float32)
	disp = mlx.where(front, mlx.f32_scalar(10.0), disp)
	disp = mlx.where(back, mlx.f32_scalar(6.5), disp)
	out := jlo_optimize(fg, disp, fg, front, back, 10.0, 6.5)
	assert out.len == 8
	u0 := out[0]
	v0 := out[1]
	z0 := out[2]
	a0 := out[3]
	u1 := out[4]
	v1 := out[5]
	z1 := out[6]
	a1 := out[7]
	assert math.abs(u0 - 87.0) < 8.0 && math.abs(v0 - 90.0) < 8.0
	assert math.abs(u1 - 60.0) < 8.0 && math.abs(v1 - 63.0) < 8.0
	assert math.abs(z0 - 3.7) < 0.2 && math.abs(z1 - 2.73) < 0.3
	assert math.abs(a0 - 3.1416 * 24.0 * 24.0) / (3.1416 * 24.0 * 24.0) < 0.2
	assert math.abs(a1 - 54.0 * 54.0) / (54.0 * 54.0) < 0.2
}
