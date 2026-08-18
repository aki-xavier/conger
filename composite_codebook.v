module conger

// composite_codebook.v — explicit attached composite template (base + attached
// part), V port of src/composite_codebook.py.
import cga
import mlx

const ccb_scale_ratio_lo = 0.35
const ccb_scale_ratio_hi = 0.75
const ccb_lateral_ratio = 0.25
const ccb_overlap_lo = 0.03
const ccb_overlap_hi = 0.10
const ccb_depth_jitter_lo = -0.06
const ccb_depth_jitter_hi = 0.06

struct CompositeCodebook {
	cfg InverseConfig
}

fn new_composite_codebook(cfg InverseConfig) CompositeCodebook {
	return CompositeCodebook{
		cfg: cfg
	}
}

// ccb_sample_composite samples an attached composite → (u0,v0,s0,z0,u1,v1,s1,z1).
fn ccb_sample_composite(mut rng Rng, _ bool) []f64 {
	for _ in 0 .. 64 {
		s0 := rng.uniform(s_range_lo, s_range_hi)
		z0 := rng.uniform(z_range_lo, z_range_hi)
		s1 := s0 * rng.uniform(ccb_scale_ratio_lo, ccb_scale_ratio_hi)
		mut z1 := z0 + rng.uniform(ccb_depth_jitter_lo, ccb_depth_jitter_hi)
		if z1 < 2.2 {
			z1 = 2.2
		}
		if z1 > 4.3 {
			z1 = 4.3
		}
		m0 := lcb_margin(s0, z0)
		if 2.0 * m0 > f64(img_w) - 4.0 {
			continue
		}
		u0 := rng.uniform(m0, f64(img_w) - m0)
		v0 := rng.uniform(m0, f64(img_h) - m0)
		x0, y0 := unproject(u0, v0, z0)
		dx := rng.uniform(-ccb_lateral_ratio, ccb_lateral_ratio) * (s0 + s1)
		overlap := rng.uniform(ccb_overlap_lo, ccb_overlap_hi) * math_min_f64(s0, s1)
		x1 := x0 + dx
		y1 := y0 + s0 + s1 - overlap
		zc1 := cam_z - z1
		u1 := f64(img_w - 1) / 2.0 + x1 * fx / zc1
		v1 := f64(img_h - 1) / 2.0 - y1 * fy / zc1
		if lcb_inside(u0, v0, s0, z0) && lcb_inside(u1, v1, s1, z1) {
			return [u0, v0, s0, z0, u1, v1, s1, z1]
		}
	}
	panic('CompositeCodebook 取景拒绝重采失败')
}

// ccb_block builds one replicate block → (2916, 14).
fn ccb_block(seed u64, extrap bool) mlx.Array {
	mut rng := new_rng(seed)
	mut rows := []f32{}
	for k0 in 0 .. n_kind {
		for k1 in 0 .. n_kind {
			for h0 in 0 .. n_hue {
				for h1 in 0 .. n_hue {
					for lc in 0 .. light_colors_len {
						for ld in 0 .. light_dirs_len {
							g := ccb_sample_composite(mut rng, extrap)
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

// ccb_sample returns (2916×R, 14).
fn ccb_sample(replicates int, seed u64, extrap bool) mlx.Array {
	mut blocks := []mlx.Array{}
	for r in 0 .. replicates {
		blocks << ccb_block(seed * 1000 + u64(r), extrap)
	}
	return mlx.concatenate(blocks, 0)
}

// SceneFamily interface methods for the composite family.
fn (cb CompositeCodebook) sample(replicates int, seed u64, extrap bool) mlx.Array {
	return ccb_sample(replicates, seed, extrap)
}

fn (cb CompositeCodebook) n_combo() int {
	return n_combo_layered
}

fn (cb CompositeCodebook) template_variant() string {
	return ''
}

fn (cb CompositeCodebook) geometry_family() string {
	return 'composite'
}

fn (cb CompositeCodebook) template_lineage() TemplateLineage {
	return composite_lineage()
}

// to_scene builds a two-object attached-composite cga Scene.
fn (cb CompositeCodebook) to_scene(params []f64) cga.Scene {
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
