module conger

// textures.v — procedural albedo texture library (V port of src/textures.py).

import cga

import mlx

// texture_checker returns a two-colour checkerboard (tile = texels per cell).
fn texture_checker(size int, c1 []f64, c2 []f64, tile int) cga.Texture {
	mut px := [][][]f64{}
	for i in 0 .. size {
		mut row := [][]f64{}
		for j in 0 .. size {
			c := if ((i / tile) + (j / tile)) % 2 == 0 { c1 } else { c2 }
			row << [c[0], c[1], c[2], 1.0]
		}
		px << row
	}
	return cga.texture_from_rgba(px)
}

// texture_stripes returns vertical stripes (period = texels per band).
fn texture_stripes(size int, c1 []f64, c2 []f64, period int) cga.Texture {
	mut px := [][][]f64{}
	for _ in 0 .. size {
		mut row := [][]f64{}
		for j in 0 .. size {
			c := if (j / period) % 2 == 0 { c1 } else { c2 }
			row << [c[0], c[1], c[2], 1.0]
		}
		px << row
	}
	return cga.texture_from_rgba(px)
}

// texture_gray_noise returns broadband gray noise in [lo, hi].
fn texture_gray_noise(size int, seed u64, lo f64, hi f64) cga.Texture {
	arr := mlx.random_normal([size, size], .float32, 0.0, 1.0, mlx.random_key(seed))
	mn := arr.min().item_f32()
	mx_ := arr.max().item_f32()
	vals := arr.subtract(mlx.f32_scalar(mn)).divide(mlx.f32_scalar(mx_ - mn + 1e-12)).multiply(mlx.f32_scalar(f32(hi - lo))).add(mlx.f32_scalar(f32(lo))).data_f32()
	mut px := [][][]f64{}
	for i in 0 .. size {
		mut row := [][]f64{}
		for j in 0 .. size {
			v := f64(vals[i * size + j])
			row << [v, v, v, 1.0]
		}
		px << row
	}
	return cga.texture_from_rgba(px)
}

// default_library returns the default texture library (n truncated to 1..3).
fn default_library(n int) []cga.Texture {
	mut out := []cga.Texture{}
	out << texture_checker(16, [0.9, 0.9, 0.9], [0.5, 0.5, 0.5], 4)
	out << texture_stripes(16, [0.9, 0.9, 0.9], [0.5, 0.5, 0.5], 3)
	out << texture_gray_noise(16, 0, 0.3, 0.7)
	k := if n < 1 { 1 } else if n > 3 { 3 } else { n }
	return out[..k]
}
