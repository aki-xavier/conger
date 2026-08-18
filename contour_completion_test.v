module conger

// contour_completion_test.v — contour completion black-box tests.
import math
import mlx

fn test_square_contour_completion() {
	h := 144
	w := 144
	yy, xx := meshgrid_ij(h, w)
	back :=
		xx.greater_equal(mlx.f32_scalar(50.0)).logical_and(xx.less(mlx.f32_scalar(90.0))).logical_and(yy.greater_equal(mlx.f32_scalar(50.0))).logical_and(yy.less(mlx.f32_scalar(90.0)))
	front :=
		xx.subtract(mlx.f32_scalar(88.0)).square().add(yy.subtract(mlx.f32_scalar(82.0)).square()).less_equal(mlx.f32_scalar(22.0 * 22.0))
	visible := back.logical_and(front.logical_not())
	u, v, area, kind, score := cc_complete(front, visible)
	assert math.abs(u - 69.5) < 5.0 && math.abs(v - 69.5) < 5.0
	assert math.abs(area - 1600.0) / 1600.0 < 0.25
	assert kind == 2
	assert score < 0.35
}

fn test_circle_contour_completion() {
	h := 144
	w := 144
	yy, xx := meshgrid_ij(h, w)
	back :=
		xx.subtract(mlx.f32_scalar(60.0)).square().add(yy.subtract(mlx.f32_scalar(60.0)).square()).less_equal(mlx.f32_scalar(20.0 * 20.0))
	front :=
		xx.subtract(mlx.f32_scalar(77.0)).square().add(yy.subtract(mlx.f32_scalar(66.0)).square()).less_equal(mlx.f32_scalar(18.0 * 18.0))
	visible := back.logical_and(front.logical_not())
	u, v, area, kind, score := cc_complete(front, visible)
	assert math.abs(u - 60.0) < 5.0 && math.abs(v - 60.0) < 5.0
	assert math.abs(area - 1256.6) / 1256.6 < 0.25
	assert kind == 0
	assert score < 0.35
}
