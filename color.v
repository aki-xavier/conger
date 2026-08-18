module conger

// color.v — colour space conversions (V port of src/color.py, MLX backend).
//
// The Python reference bridges PIL→numpy for image loading and uses numpy for a
// few constant matrices; those constants are inlined as f32 arrays here. The
// complex-valued `hsl_to_complex`/`split_dual_path` and the PIL-based
// `image_to_mlx` are not ported (no complex-array construction in mlx-v, and no
// PIL); they are unused by the test suite.
import mlx

struct Color {}

// lab_to_rgb converts CIELAB → sRGB, clipped to [0,1].
fn (c Color) lab_to_rgb(lab_image mlx.Array) mlx.Array {
	l := lab_image.take_axis(sel1(0), -1)
	a := lab_image.take_axis(sel1(1), -1)
	b := lab_image.take_axis(sel1(2), -1)

	f_y := l.add(mlx.f32_scalar(16.0)).divide(mlx.f32_scalar(116.0))
	f_x := a.divide(mlx.f32_scalar(500.0)).add(f_y)
	f_z := f_y.subtract(b.divide(mlx.f32_scalar(200.0)))

	f_xyz := mlx.stack([f_x, f_y, f_z], -1)
	f_xyz_cubed := f_xyz.multiply(f_xyz).multiply(f_xyz)

	epsilon := 0.008856
	xyz_normalized := mlx.where(f_xyz_cubed.greater(mlx.f32_scalar(f32(epsilon))), f_xyz_cubed,
		f_xyz.subtract(mlx.f32_scalar(16.0 / 116.0)).divide(mlx.f32_scalar(7.787)))

	white_point := mlx.array_f32([f32(0.95047), 1.00000, 1.08883], [3])
	xyz := xyz_normalized.multiply(white_point)

	xyz_to_rgb := mlx.array_f32([
		f32(3.2404542),
		-1.5371385,
		-0.4985314,
		-0.9692660,
		1.8760108,
		0.0415560,
		0.0556434,
		-0.2040259,
		1.0572252,
	], [3, 3])
	linear_rgb := xyz.matmul(xyz_to_rgb.transpose())

	safe := linear_rgb.maximum(mlx.f32_scalar(1e-6))
	mask := linear_rgb.greater(mlx.f32_scalar(f32(0.0031308)))
	srgb := mlx.where(mask,
		safe.power(mlx.f32_scalar(1.0 / 2.4)).multiply(mlx.f32_scalar(1.055)).subtract(mlx.f32_scalar(0.055)),
		linear_rgb.multiply(mlx.f32_scalar(12.92)))
	return srgb.clip(mlx.f32_scalar(0.0), mlx.f32_scalar(1.0))
}

// rgb_to_lab converts sRGB → CIELAB.
fn (c Color) rgb_to_lab(rgb_image mlx.Array) mlx.Array {
	rgb := rgb_image.maximum(mlx.f32_scalar(0.0))
	mask_linear := rgb.greater(mlx.f32_scalar(f32(0.04045)))
	linear_rgb := mlx.where(mask_linear,
		rgb.add(mlx.f32_scalar(0.055)).divide(mlx.f32_scalar(1.055)).power(mlx.f32_scalar(2.4)),
		rgb.divide(mlx.f32_scalar(12.92)))

	xyz_matrix := mlx.array_f32([
		f32(0.4124564),
		0.3575761,
		0.1804375,
		0.2126729,
		0.7151522,
		0.0721750,
		0.0193339,
		0.1191920,
		0.9503041,
	], [3, 3])
	xyz := linear_rgb.matmul(xyz_matrix.transpose())

	white_point := mlx.array_f32([f32(0.95047), 1.00000, 1.08883], [3])
	xyz_normalized := xyz.divide(white_point)

	epsilon := 0.008856
	safe_xyz := xyz_normalized.maximum(mlx.f32_scalar(1e-6))
	mask_lab := xyz_normalized.greater(mlx.f32_scalar(f32(epsilon)))
	f_xyz := mlx.where(mask_lab, safe_xyz.power(mlx.f32_scalar(1.0 / 3.0)),
		xyz_normalized.multiply(mlx.f32_scalar(7.787)).add(mlx.f32_scalar(16.0 / 116.0)))

	f_x := f_xyz.take_axis(sel1(0), -1)
	f_y := f_xyz.take_axis(sel1(1), -1)
	f_z := f_xyz.take_axis(sel1(2), -1)

	l :=
		f_y.multiply(mlx.f32_scalar(116.0)).subtract(mlx.f32_scalar(16.0)).maximum(mlx.f32_scalar(0.0))
	a := f_x.subtract(f_y).multiply(mlx.f32_scalar(500.0))
	b := f_y.subtract(f_z).multiply(mlx.f32_scalar(200.0))

	return mlx.stack([l, a, b], -1)
}

// gray_world_wb applies grey-world white balance.
fn (c Color) gray_world_wb(rgb_image mlx.Array) mlx.Array {
	means := rgb_image.mean_axes([-3, -2], false)
	gray := means.mean_axis(-1, true)
	gain := gray.divide(means.maximum(mlx.f32_scalar(1e-6)))
	gain_b := gain.expand_dims(-2).expand_dims(-2)
	return rgb_image.multiply(gain_b).clip(mlx.f32_scalar(0.0), mlx.f32_scalar(1.0))
}

// log_chromaticity returns c1=log(R/G), c2=log(B/G) (lighting-invariant).
fn (c Color) log_chromaticity(rgb_image mlx.Array, eps f64) mlx.Array {
	safe := rgb_image.maximum(mlx.f32_scalar(f32(eps)))
	c0 := safe.take_axis(sel1(0), -1)
	c1 := safe.take_axis(sel1(1), -1)
	c2 := safe.take_axis(sel1(2), -1)
	l0 := c0.divide(c1).log()
	l2 := c2.divide(c1).log()
	return mlx.concatenate([l0, l2], -1)
}
