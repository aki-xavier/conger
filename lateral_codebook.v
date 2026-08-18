module conger

// lateral_codebook.v — mirror/repeat lateral same-kind composite template
// (V port of src/lateral_codebook.py; param order stays 14-compatible).

import cga

import mlx

const lc_part_period_lo = 0.15
const lc_part_period_hi = 0.25
const lc_spacing_mirror = 5.0
const lc_spacing_repeat = 7.5

struct LateralCompositeCodebook {
	cfg InverseConfig
}

fn new_lateral_codebook(cfg InverseConfig) LateralCompositeCodebook {
	return LateralCompositeCodebook{
		cfg: cfg
	}
}

// lc_spacing_factor returns the period scaling constant for mirror/repeat.
fn lc_spacing_factor(operation string) f64 {
	if operation == 'mirror' {
		return lc_spacing_mirror
	}
	return lc_spacing_repeat
}

// lc_sample_composite samples base (left) / part (right) at the same depth →
// (u0,v0,s0,z0,u1,v1,s1,z1). The part keeps the same y as the base.
fn lc_sample_composite(mut rng Rng, extrap bool) []f64 {
	for _ in 0 .. 64 {
		s0 := rng.uniform(s_range_lo, s_range_hi)
		z0 := rng.uniform(z_range_lo, z_range_hi)
		s1 := s0 * rng.uniform(ccb_scale_ratio_lo, ccb_scale_ratio_hi)
		z1 := z0
		m0 := lcb_margin(s0, z0)
		if 2.0 * m0 > f64(img_w) - 4.0 {
			continue
		}
		u0 := rng.uniform(m0, f64(img_w) - m0)
		v0 := rng.uniform(m0, f64(img_h) - m0)
		x0, y0 := unproject(u0, v0, z0)
		period := rng.uniform(lc_part_period_lo, lc_part_period_hi)
		x1 := x0 + period * lc_spacing_mirror * (s0 + s1)
		zc1 := cam_z - z1
		u1 := f64(img_w-1)/2.0 + x1 * fx / zc1
		v1 := f64(img_h-1)/2.0 - y0 * fy / zc1
		if lcb_inside(u0, v0, s0, z0) && lcb_inside(u1, v1, s1, z1) {
			return [u0, v0, s0, z0, u1, v1, s1, z1]
		}
	}
	panic('LateralCompositeCodebook 取景拒绝重采失败')
}

// lc_block builds one replicate block → (162, 14); part shares kind/hue with base.
fn lc_block(seed u64, extrap bool) mlx.Array {
	mut rng := new_rng(seed)
	mut rows := []f32{}
	for k0 in 0 .. n_kind {
		for h0 in 0 .. n_hue {
			for lc in 0 .. light_colors_len {
				for ld in 0 .. light_dirs_len {
					g := lc_sample_composite(mut rng, extrap)
					rows << f32(k0)
					rows << f32(g[0])
					rows << f32(g[1])
					rows << f32(g[2])
					rows << f32(g[3])
					rows << f32(h0)
					rows << f32(k0)
					rows << f32(g[4])
					rows << f32(g[5])
					rows << f32(g[6])
					rows << f32(g[7])
					rows << f32(h0)
					rows << f32(lc)
					rows << f32(ld)
				}
			}
		}
	}
	return mlx.array_f32(rows, [n_kind * n_hue * light_colors_len * light_dirs_len, 14])
}

// lc_sample returns (162×R, 14).
fn lc_sample(replicates int, seed u64, extrap bool) mlx.Array {
	mut blocks := []mlx.Array{}
	for r in 0 .. replicates {
		blocks << lc_block(seed * 1000 + u64(r), extrap)
	}
	return mlx.concatenate(blocks, 0)
}

// to_scene reuses the attached-composite two-object scene builder.
fn (cb LateralCompositeCodebook) to_scene(params []f64) cga.Scene {
	return new_composite_codebook(cb.cfg).to_scene(params)
}

// SceneFamily interface methods for the lateral family.
fn (cb LateralCompositeCodebook) sample(replicates int, seed u64, extrap bool) mlx.Array {
	return lc_sample(replicates, seed, extrap)
}

fn (cb LateralCompositeCodebook) n_combo() int {
	return n_kind * n_hue * light_colors_len * light_dirs_len
}

fn (cb LateralCompositeCodebook) template_variant() string {
	return ''
}

fn (cb LateralCompositeCodebook) geometry_family() string {
	return 'lateral'
}

fn (cb LateralCompositeCodebook) template_lineage() TemplateLineage {
	return lateral_lineage()
}
