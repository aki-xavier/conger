module conger

// stereo.v — binocular disparity → depth (parallel camera rig), V port of
// src/stereo.py.

import mlx

struct StereoDepth {
	b f64 = stereo_base
}

// foreground_weights returns m = S² + (lum−bg)² per pixel.
fn foreground_weights(frame mlx.Array) mlx.Array {
	re, im := frame_chroma(frame)
	lum := frame_lum(frame)
	ld := lum.data_f32()
	h := lum.dim(0)
	w := lum.dim(1)
	mut corners := []f32{}
	for y in 0 .. 8 {
		for x in 0 .. 8 {
			corners << ld[y * w + x]
		}
		for x in w - 8 .. w {
			corners << ld[y * w + x]
		}
	}
	for y in h - 8 .. h {
		for x in 0 .. 8 {
			corners << ld[y * w + x]
		}
		for x in w - 8 .. w {
			corners << ld[y * w + x]
		}
	}
	corners.sort()
	bg := f32(corners[corners.len / 2])
	dl := lum.subtract(mlx.f32_scalar(bg))
	return re.multiply(re).add(im.multiply(im)).add(dl.multiply(dl))
}

// centroid returns (weighted x centroid, mask pixel count).
fn centroid(frame mlx.Array) (f64, f64) {
	m := foreground_weights(frame)
	xs := mlx.arange(0.0, f64(m.dim(1)), 1.0, .float32).expand_dims(0)
	tot := m.sum().item_f32()
	cx := m.multiply(xs).sum().item_f32() / fmax2(f64(tot), 1e-8)
	area := m.greater(mlx.f32_scalar(0.01)).astype(.float32).sum().item_f32()
	return f64(cx), f64(area)
}

// estimate returns (ẑ world units, disparity px, left-mask area px²).
fn (sd StereoDepth) estimate(frame_l mlx.Array, frame_r mlx.Array) (f64, f64, f64) {
	cx_l, area := centroid(frame_l)
	cx_r, _ := centroid(frame_r)
	d := cx_l - cx_r
	mut z := cam_z - fx * sd.b / fmax2(d, 1e-6)
	if z < 0.5 {
		z = 0.5
	}
	if z > cam_z + 1.0 {
		z = cam_z + 1.0
	}
	return z, d, area
}
