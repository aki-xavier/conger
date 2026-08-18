module conger

// codebook_const.v — shared scene-domain constants from the Codebook class.

import math

const n_kind = 3
const n_hue = 6
const light_colors = [0xFFFFFF, 0xFF4040, 0x4040FF]
const light_colors_len = 3
const light_dirs_len = 3
const cam_z = 5.5
const fx = 90.0
const fy = 90.0
const stereo_base = 0.2
const img_h = 144
const img_w = 144
const s_range_lo = 0.35
const s_range_hi = 0.6
const z_range_lo = 2.5
const z_range_hi = 4.0
const extent = 1.8

// codebook_fov returns the vertical field of view in degrees.
fn codebook_fov() f64 {
	return 2.0 * math.degrees(math.atan((f64(img_h) / 2.0) / fy))
}

// codebook_light_dirs returns the three directional-light directions.
fn codebook_light_dirs() [][3]f64 {
	return [
		[0.3, -0.7, 0.4]!,
		[-0.6, -0.4, 0.7]!,
		[0.6, -0.4, 0.7]!,
	]
}

const n_combo_layered = 3 * 3 * 6 * 6 * 3 * 3 // 2916
