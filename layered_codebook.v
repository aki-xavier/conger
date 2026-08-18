module conger

// layered_codebook.v — two-object occlusion front/back scene family
// (V port of src/layered_codebook.py).
import cga
import math
import mlx

const lcb_target_idx = [1, 2, 3, 4, 7, 8, 9, 10]
const lcb_class_idx = [0, 6, 5, 11, 12, 13]
const lcb_cat_sizes = [3, 3, 6, 6, 3, 3]

struct LayeredCodebook {
	cfg InverseConfig
}

fn new_layered_codebook(cfg InverseConfig) LayeredCodebook {
	return LayeredCodebook{
		cfg: cfg
	}
}

// lcb_margin returns the pixel margin for a primitive at scale s, depth z.
fn lcb_margin(s f64, z f64) f64 {
	zc := cam_z - z
	m := extent * s * fx / zc + 2.0
	return m + stereo_base / 2.0 * fx / zc
}

fn lcb_inside(u f64, v f64, s f64, z f64) bool {
	m := lcb_margin(s, z)
	return m <= u && u <= f64(img_w) - m && m <= v && v <= f64(img_h) - m
}

// lcb_sample_free samples one free object (u,v,s,z) with rejection.
fn lcb_sample_free(mut rng Rng, extrap bool, z_lo f64, z_hi f64) (f64, f64, f64, f64) {
	mut s := 0.0
	mut z := 0.0
	mut m := 0.0
	for _ in 0 .. 8 {
		s = rng.uniform(s_range_lo, s_range_hi)
		z = rng.uniform(z_lo, z_hi)
		m = lcb_margin(s, z)
		if 2.0 * m <= f64(img_w) - 4.0 {
			break
		}
	}
	u := rng.uniform(m, f64(img_w) - m)
	v := rng.uniform(m, f64(img_h) - m)
	return u, v, s, z
}

// lcb_sample_pair samples front/back continuous params.
fn lcb_sample_pair(mut rng Rng, extrap bool) []f64 {
	for _ in 0 .. 8 {
		u0, v0, s0, z0 := lcb_sample_free(mut rng, extrap, 3.1, 4.2)
		z1 := fmax2(z0 - rng.uniform(0.7, 1.4), 2.3)
		mut u1, mut v1, mut s1, _ := lcb_sample_free(mut rng, extrap, 2.3, 3.5)
		mut z1c := math_min_f64(z1, z0 - 0.05)
		if rng.f64() < 0.7 {
			a0 := extent * s0 * fx / (cam_z - z0)
			a1 := extent * s1 * fx / (cam_z - z1c)
			reach := 0.75 * (a0 + a1)
			u1 = u0 + rng.uniform(-reach, reach)
			v1 = v0 + rng.uniform(-reach, reach)
			m1 := lcb_margin(s1, z1c)
			if !(m1 <= u1 && u1 <= f64(img_w) - m1 && m1 <= v1 && v1 <= f64(img_h) - m1) {
				u1, v1, s1, _ = lcb_sample_free(mut rng, extrap, 2.3, 3.5)
				z1c = fmax2(z0 - rng.uniform(0.7, 1.4), 2.3)
			}
		}
		return [u0, v0, s0, z0, u1, v1, s1, z1c]
	}
	panic('LayeredCodebook 子模板取景拒绝重采失败')
}

// lcb_block builds one replicate block → (2916, 14).
fn lcb_block(seed u64, extrap bool) mlx.Array {
	mut rng := new_rng(seed)
	mut rows := []f32{}
	for k0 in 0 .. n_kind {
		for k1 in 0 .. n_kind {
			for h0 in 0 .. n_hue {
				for h1 in 0 .. n_hue {
					for lc in 0 .. light_colors_len {
						for ld in 0 .. light_dirs_len {
							g := lcb_sample_pair(mut rng, extrap)
							rows << f32(k0)
							rows << f32(g[0])
							rows << f32(g[1])
							rows << f32(g[2])
							rows << f32(g[3])
							rows << f32(h0)
							rows << f32(k1)
							rows << f32(g[4])
							rows << f32(g[5])
							rows << f32(g[6])
							rows << f32(g[7])
							rows << f32(h1)
							rows << f32(lc)
							rows << f32(ld)
						}
					}
				}
			}
		}
	}
	return mlx.array_f32(rows, [n_combo_layered, 14])
}

// sample returns (2916×R, 14).
fn lcb_sample(replicates int, seed u64, extrap bool) mlx.Array {
	mut blocks := []mlx.Array{}
	for r in 0 .. replicates {
		blocks << lcb_block(seed * 1000 + u64(r), extrap)
	}
	return mlx.concatenate(blocks, 0)
}

// SceneFamily interface methods for the layered family.
fn (cb LayeredCodebook) sample(replicates int, seed u64, extrap bool) mlx.Array {
	return lcb_sample(replicates, seed, extrap)
}

fn (cb LayeredCodebook) n_combo() int {
	return n_combo_layered
}

fn (cb LayeredCodebook) template_variant() string {
	return ''
}

fn (cb LayeredCodebook) geometry_family() string {
	return 'layered'
}

fn (cb LayeredCodebook) template_lineage() TemplateLineage {
	return layered_lineage()
}

// to_scene builds a two-object cga Scene.
fn (cb LayeredCodebook) to_scene(params []f64) cga.Scene {
	assert params.len == 14
	mut sc := cga.scene(cga.color_hex(cb.cfg.bg_color))
	sc.add_light(cga.ambient_light(cga.color_hex(0xFFFFFF), 0.5))
	lcol := int(params[12])
	ldir := int(params[13])
	dirs := codebook_light_dirs()
	sc.add_light(cga.directional_light(cga.color_hex(light_colors[lcol]), 0.7, dirs[ldir]))
	for off in [0, 6] {
		kind := int(params[off])
		u := params[off + 1]
		v := params[off + 2]
		s := params[off + 3]
		z := params[off + 4]
		hue := int(params[off + 5])
		x, y := unproject(u, v, z)
		mat := cga.standard_material(cga.color_hex(obj_color(hue)), 0.55, 0.0, cga.color_hex(0),
			1.0, 1.5, 0.0)
		sc.add_mesh(cga.mesh(codebook_geometry(kind, s), mat, [x, y, z]!, [0.0, 0.0, 0.0]!, 0.0,
			none))
	}
	return sc
}

// lcb_targets returns the continuous targets [u0,v0,s0,z0,u1,v1,s1,z1].
fn lcb_targets(p mlx.Array) mlx.Array {
	return p.take_axis(mlx.arange(0.0, 8.0, 1.0, .int32), 0).take_axis(mlx.array_i32([
		i32(1),
		i32(2),
		i32(3),
		i32(4),
		i32(7),
		i32(8),
		i32(9),
		i32(10),
	], [8]), 1)
}

// lcb_scene_classes returns the discrete factors [k0,k1,h0,h1,lcol,ldir].
fn lcb_scene_classes(p mlx.Array) mlx.Array {
	return p.take_axis(mlx.array_i32([i32(0), i32(6), i32(5), i32(11), i32(12), i32(13)], [
		6,
	]), 1).astype(.int32)
}

fn math_min_f64(a f64, b f64) f64 {
	return if a < b { a } else { b }
}

// layered_make_stats builds the 8-d [u,v,z,area]×2 anchor stats for a params row.
fn layered_make_stats(p []f64, c []int) []f64 {
	mut stats := []f64{}
	for layer in 0 .. 2 {
		off := if layer == 0 { 0 } else { 6 }
		u := p[off + 1]
		v := p[off + 2]
		s := p[off + 3]
		z := p[off + 4]
		kind := c[layer]
		ratio := if kind == 2 { 0.5 } else { 1.0 / math.sqrt(math.pi) }
		area := (s * fx / (ratio * (cam_z - z))) * (s * fx / (ratio * (cam_z - z)))
		stats << u
		stats << v
		stats << z
		stats << area
	}
	return stats
}
