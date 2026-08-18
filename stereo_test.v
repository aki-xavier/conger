module conger

// stereo_test.v — StereoDepth black-box test.
import math
import mlx

const stereo_shift = 8

fn make_stereo_frames() (mlx.Array, mlx.Array) {
	h := 144
	w := 144
	mut fl := []u8{len: h * w * 4}
	mut fr := []u8{len: h * w * 4}
	for y in 50 .. 90 {
		for x in 60 .. 100 {
			idx := (y * w + x) * 4
			fl[idx] = 200
			fl[idx + 3] = 255
			xr := x - stereo_shift
			idxr := (y * w + xr) * 4
			fr[idxr] = 200
			fr[idxr + 3] = 255
		}
	}
	return mlx.array_with(fl, [h, w, 4], .uint8), mlx.array_with(fr, [h, w, 4], .uint8)
}

fn test_disparity_pipeline() {
	fl, fr := make_stereo_frames()
	z, d, area := StereoDepth{
		b: 0.2
	}.estimate(fl, fr)
	assert math.abs(d - f64(stereo_shift)) < 0.1
	assert math.abs(z - 3.25) < 0.05
	assert math.abs(area - 1600.0) < 1.0
	_, d2, _ := StereoDepth{
		b: 0.2
	}.estimate(fr, fl)
	assert math.abs(d2 + f64(stereo_shift)) < 0.1
}
