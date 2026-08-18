module conger

// color_test.v — optical-prior black-box tests (white balance / log chromaticity).
import math
import mlx

fn base_array() mlx.Array {
	mut data := []f32{len: 32 * 96 * 3}
	for y in 0 .. 32 {
		for x in 0 .. 96 {
			idx := (y * 96 + x) * 3
			if x < 32 {
				data[idx] = 0.7
				data[idx + 1] = 0.2
				data[idx + 2] = 0.2
			} else if x < 64 {
				data[idx] = 0.2
				data[idx + 1] = 0.6
				data[idx + 2] = 0.3
			} else {
				data[idx] = 0.5
				data[idx + 1] = 0.5
				data[idx + 2] = 0.55
			}
		}
	}
	return mlx.array_f32(data, [32, 96, 3])
}

fn test_gray_world_wb() {
	base := base_array()
	cast := base.multiply(mlx.array_f32([f32(1.3), 1.0, 0.75], [3]))
	c := Color{}
	wb := c.gray_world_wb(cast)
	means := wb.mean_axes([0, 1], false)
	spread := means.max().item_f32() - means.min().item_f32()
	assert spread < 0.05

	wb4 := c.gray_world_wb(mlx.stack([cast, cast], 0))
	assert wb4.shape() == [2, 32, 96, 3]
	row0 := wb4.take_axis(sel1(0), 0).squeeze_axis(0)
	diff := row0.subtract(wb).abs().max().item_f32()
	assert diff < 1e-6
}

fn test_log_chromaticity() {
	base := base_array()
	c := Color{}
	c_bright := c.log_chromaticity(base, 1e-3)
	c_dark := c.log_chromaticity(base.multiply(mlx.f32_scalar(0.3)), 1e-3)
	diff := c_bright.subtract(c_dark).abs().max().item_f32()
	assert diff < 1e-3

	cb := c_bright.data_f32()
	idx1 := (16 * 96 + 16) * 2
	idx2 := (16 * 96 + 48) * 2
	gap := math.abs(cb[idx1] - cb[idx2]) + math.abs(cb[idx1 + 1] - cb[idx2 + 1])
	assert gap > 0.5
}
