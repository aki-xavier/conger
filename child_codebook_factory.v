module conger

// child_codebook_factory.v — materialise a ChildTemplateSpec into a
// constraint-bearing child scene family (V port of src/child_codebook_factory.py;
// dynamic Python classes become a constraint struct + parametrised samplers).
import cga
import math
import mlx

// ChildCodebook is a constraint-bearing scene family derived from a spec.
struct ChildCodebook {
	cfg          InverseConfig
	operation    string
	scale_lo     f64
	scale_hi     f64
	lateral_lo   f64
	lateral_hi   f64
	depth_gap_lo f64
	depth_gap_hi f64
	part_kinds   []int
	part_hues    []int
	base_kinds   []int
	base_hues    []int
	spacing      f64
	period_lo    f64
	period_hi    f64
	n_combo_n    int
	lineage      TemplateLineage
	variant      string
}

// ccf_list_or returns a []f64 constraint or a default pair.
fn ccf_list_or(delta map[string]MetaValue, key string, def_lo f64, def_hi f64) (f64, f64) {
	lst := meta_list(delta, key)
	if lst.len >= 2 {
		return lst[0], lst[1]
	}
	return def_lo, def_hi
}

// ccf_int_list returns a sorted int list constraint (or the default 0..n-1).
fn ccf_int_list(delta map[string]MetaValue, key string, default_n int) []int {
	lst := meta_list(delta, key)
	mut out := []int{}
	for v in lst {
		out << int(v)
	}
	if out.len > 0 {
		out.sort()
		return out
	}
	for i in 0 .. default_n {
		out << i
	}
	return out
}

// ccf_build derives the constraint fields for a spec.
fn ccf_build(spec ChildTemplateSpec) ChildCodebook {
	op := spec.operation
	scale_lo, scale_hi := ccf_list_or(spec.constraints, 'scale_ratio', ccb_scale_ratio_lo,
		ccb_scale_ratio_hi)
	part_kinds := ccf_int_list(spec.constraints, 'part_kinds', n_kind)
	part_hues := ccf_int_list(spec.constraints, 'part_hues', n_hue)
	mut base_kinds := [0, 1, 2]
	mut base_hues := [0, 1, 2, 3, 4, 5]
	mut lateral_lo := 0.0
	mut lateral_hi := 0.0
	mut depth_gap_lo := 0.0
	mut depth_gap_hi := 0.0
	mut spacing := 0.0
	mut period_lo := 0.0
	mut period_hi := 0.0
	mut n_combo_n := 0
	mut lineage := spec.lineage()
	if op == 'attach' {
		lateral_lo, lateral_hi = ccf_list_or(spec.constraints, 'lateral_ratio', -ccb_lateral_ratio,
			ccb_lateral_ratio)
		depth_gap_lo, depth_gap_hi = ccf_list_or(spec.constraints, 'depth_jitter',
			ccb_depth_jitter_lo, ccb_depth_jitter_hi)
		n_combo_n = 3 * part_kinds.len * 6 * part_hues.len * light_colors_len * light_dirs_len
	} else if op == 'layer' {
		depth_gap_lo, depth_gap_hi = ccf_list_or(spec.constraints, 'depth_gap', 0.7, 1.4)
		mut l_lo, mut l_hi := ccf_list_or(spec.constraints, 'lateral_ratio', -0.75, 0.75)
		if fmax2(math.abs(l_lo), math.abs(l_hi)) < 0.35 {
			l_lo, l_hi = 0.35, 0.7
		}
		lateral_lo, lateral_hi = l_lo, l_hi
		n_combo_n = 3 * part_kinds.len * 6 * part_hues.len * light_colors_len * light_dirs_len
		// widened lateral is written back into the lineage delta
		def_lo, def_hi := ccf_list_or(spec.constraints, 'lateral_ratio', -0.75, 0.75)
		if l_lo != def_lo || l_hi != def_hi {
			mut delta := spec.constraints.clone()
			delta['lateral_ratio'] = [l_lo, l_hi]
			lineage = TemplateLineage{
				family:        spec.name
				parent_family: spec.parent_family
				operation:     spec.operation
				complexity:    spec.complexity
				generation:    spec.generation
				delta:         delta
			}
		}
	} else {
		// mirror / repeat
		spacing = lc_spacing_factor(op)
		period_lo, period_hi = ccf_list_or(spec.constraints, 'period_ratio', lc_part_period_lo,
			lc_part_period_hi)
		base_kinds = part_kinds.clone()
		base_hues = part_hues.clone()
		n_combo_n = part_kinds.len * part_hues.len * light_colors_len * light_dirs_len
	}
	return ChildCodebook{
		operation:    op
		scale_lo:     scale_lo
		scale_hi:     scale_hi
		lateral_lo:   lateral_lo
		lateral_hi:   lateral_hi
		depth_gap_lo: depth_gap_lo
		depth_gap_hi: depth_gap_hi
		part_kinds:   part_kinds
		part_hues:    part_hues
		base_kinds:   base_kinds
		base_hues:    base_hues
		spacing:      spacing
		period_lo:    period_lo
		period_hi:    period_hi
		n_combo_n:    n_combo_n
		lineage:      lineage
		variant:      spec.name
	}
}

// ccf_sample_composite samples an attach child → (u0,v0,s0,z0,u1,v1,s1,z1).
fn ccf_sample_composite(mut rng Rng, cb ChildCodebook) []f64 {
	for _ in 0 .. 64 {
		s0 := rng.uniform(s_range_lo, s_range_hi)
		z0 := rng.uniform(z_range_lo, z_range_hi)
		s1 := s0 * rng.uniform(cb.scale_lo, cb.scale_hi)
		mut z1 := z0 + rng.uniform(cb.depth_gap_lo, cb.depth_gap_hi)
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
		dx := rng.uniform(cb.lateral_lo, cb.lateral_hi) * (s0 + s1)
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

// ccf_sample_layer samples a constrained layer child → (u0,v0,s0,z0,u1,v1,s1,z1).
fn ccf_sample_layer(mut rng Rng, cb ChildCodebook) []f64 {
	for _ in 0 .. 8 {
		u0, v0, s0, z0 := lcb_sample_free(mut rng, false, 3.1, 4.2)
		s1 := s0 * rng.uniform(cb.scale_lo, cb.scale_hi)
		z1 := fmax2(z0 - rng.uniform(cb.depth_gap_lo, cb.depth_gap_hi), 2.3)
		a0 := extent * s0 * fx / (cam_z - z0)
		a1 := extent * s1 * fx / (cam_z - z1)
		u1 := u0 + rng.uniform(cb.lateral_lo, cb.lateral_hi) * (a0 + a1)
		v1 := v0
		if lcb_inside(u0, v0, s0, z0) && lcb_inside(u1, v1, s1, z1) {
			return [u0, v0, s0, z0, u1, v1, s1, z1]
		}
	}
	panic('LayeredCodebook 子模板取景拒绝重采失败')
}

// ccf_sample_lateral samples a mirror/repeat child → (u0,v0,s0,z0,u1,v1,s1,z1).
fn ccf_sample_lateral(mut rng Rng, cb ChildCodebook) []f64 {
	for _ in 0 .. 64 {
		s0 := rng.uniform(s_range_lo, s_range_hi)
		z0 := rng.uniform(z_range_lo, z_range_hi)
		s1 := s0 * rng.uniform(cb.scale_lo, cb.scale_hi)
		z1 := z0
		m0 := lcb_margin(s0, z0)
		if 2.0 * m0 > f64(img_w) - 4.0 {
			continue
		}
		u0 := rng.uniform(m0, f64(img_w) - m0)
		v0 := rng.uniform(m0, f64(img_h) - m0)
		x0, y0 := unproject(u0, v0, z0)
		period := rng.uniform(cb.period_lo, cb.period_hi)
		x1 := x0 + period * cb.spacing * (s0 + s1)
		zc1 := cam_z - z1
		u1 := f64(img_w - 1) / 2.0 + x1 * fx / zc1
		v1 := f64(img_h - 1) / 2.0 - y0 * fy / zc1
		if lcb_inside(u0, v0, s0, z0) && lcb_inside(u1, v1, s1, z1) {
			return [u0, v0, s0, z0, u1, v1, s1, z1]
		}
	}
	panic('LateralCompositeCodebook 取景拒绝重采失败')
}

// ccf_block builds one child replicate block.
fn ccf_block(cb ChildCodebook, seed u64, extrap bool) mlx.Array {
	mut rng := new_rng(seed)
	mut rows := []f32{}
	for k0 in cb.base_kinds {
		for k1 in cb.part_kinds {
			for h0 in cb.base_hues {
				for h1 in cb.part_hues {
					for lc in 0 .. light_colors_len {
						for ld in 0 .. light_dirs_len {
							g := if cb.operation == 'attach' {
								ccf_sample_composite(mut rng, cb)
							} else if cb.operation == 'layer' {
								ccf_sample_layer(mut rng, cb)
							} else {
								ccf_sample_lateral(mut rng, cb)
							}
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
	return mlx.array_f32(rows, [cb.n_combo_n, 14])
}

// ccf_sample returns (n_combo × R, 14).
fn ccf_sample(cb ChildCodebook, replicates int, seed u64, extrap bool) mlx.Array {
	mut blocks := []mlx.Array{}
	for r in 0 .. replicates {
		blocks << ccf_block(cb, seed * 1000 + u64(r), extrap)
	}
	return mlx.concatenate(blocks, 0)
}

// to_scene builds the two-object cga Scene for a child family.
fn (cb ChildCodebook) to_scene(params []f64) cga.Scene {
	return new_composite_codebook(cb.cfg).to_scene(params)
}

// SceneFamily interface methods for a child codebook.
fn (cb ChildCodebook) sample(replicates int, seed u64, extrap bool) mlx.Array {
	return ccf_sample(cb, replicates, seed, extrap)
}

fn (cb ChildCodebook) n_combo() int {
	return cb.n_combo_n
}

fn (cb ChildCodebook) template_variant() string {
	return cb.variant
}

fn (cb ChildCodebook) geometry_family() string {
	if cb.operation == 'attach' {
		return 'composite'
	}
	if cb.operation == 'layer' {
		return 'layered'
	}
	return 'lateral'
}

fn (cb ChildCodebook) template_lineage() TemplateLineage {
	return cb.lineage
}
