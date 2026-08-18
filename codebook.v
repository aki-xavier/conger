module conger

// codebook.v — continuous scene params ⇄ cga Scene + domain constants
// (V port of src/codebook.py).
import cga
import mlx

// Codebook holds the scene-domain config (rendering integration).
struct Codebook {
	cfg InverseConfig
}

fn new_codebook(cfg InverseConfig) Codebook {
	return Codebook{
		cfg: cfg
	}
}

// obj_color maps a hue index to an RGB hex (HSV(H, 0.8, 0.85)).
fn obj_color(hue_idx int) int {
	h := f64(hue_idx) / f64(n_hue)
	r, g, b := hsv_to_rgb(h, 0.8, 0.85)
	return (int(r * 255.0) << 16) | (int(g * 255.0) << 8) | int(b * 255.0)
}

// hsv_to_rgb converts HSV (h∈[0,1], s,v∈[0,1]) to sRGB (r,g,b ∈[0,1]).
fn hsv_to_rgb(h f64, s f64, v f64) (f64, f64, f64) {
	if s == 0.0 {
		return v, v, v
	}
	i := int(h * 6.0)
	f := h * 6.0 - f64(i)
	p := v * (1.0 - s)
	q := v * (1.0 - s * f)
	t := v * (1.0 - s * (1.0 - f))
	im := i % 6
	match im {
		0 { return v, t, p }
		1 { return q, v, p }
		2 { return p, v, t }
		3 { return p, q, v }
		4 { return t, p, v }
		else { return v, p, q }
	}
}

// unproject maps pixel (u,v) + depth to world coordinates.
fn unproject(u f64, v f64, z0 f64) (f64, f64) {
	zc := cam_z - z0
	x := (u - f64(img_w - 1) / 2.0) * zc / fx
	y := (f64(img_h - 1) / 2.0 - v) * zc / fy
	return x, y
}

// geometry returns the primitive geometry for (kind, scale).
fn codebook_geometry(kind int, s f64) cga.Geometry {
	if kind == 0 {
		return cga.sphere_geometry(s)
	}
	if kind == 1 {
		return cga.cylinder_geometry(s, 2.2 * s)
	}
	return cga.box_geometry(2.0 * s, 2.0 * s, 2.0 * s)
}

// to_scene builds a cga Scene from scene params (kind,u,v,s,z,hue,lcol,ldir).
fn (cb Codebook) to_scene(params []f64) cga.Scene {
	kind := int(params[0])
	u := params[1]
	v := params[2]
	s := params[3]
	z := params[4]
	hue := int(params[5])
	lcol := int(params[6])
	ldir := int(params[7])
	x, y := unproject(u, v, z)
	geom := codebook_geometry(kind, s)
	mut sc := cga.scene(cga.color_hex(cb.cfg.bg_color))
	sc.add_light(cga.ambient_light(cga.color_hex(0xFFFFFF), 0.5))
	dirs := codebook_light_dirs()
	sc.add_light(cga.directional_light(cga.color_hex(light_colors[lcol]), 0.7, dirs[ldir]))
	mat := cga.standard_material(cga.color_hex(obj_color(hue)), 0.55, 0.0, cga.color_hex(0), 1.0,
		1.5, 0.0)
	sc.add_mesh(cga.mesh(geom, mat, [x, y, z]!, [0.0, 0.0, 0.0]!, 0.0, none))
	return sc
}

// make_renderer builds the parallel rig (left/right cameras).
fn make_renderer(baseline f64) (cga.Renderer, cga.PerspectiveCamera, cga.PerspectiveCamera) {
	mut r := cga.renderer(img_w, img_h, 1, 3)
	fv := codebook_fov()
	mut cam_l := cga.perspective_camera(fv, 1.0, 0.1, 50.0, [-baseline / 2.0, 0.0, cam_z]!, [
		-baseline / 2.0,
		0.0,
		0.0,
	]!, [0.0, 1.0, 0.0]!)
	cam_l.look_at([-baseline / 2.0, 0.0, 0.0]!, none)
	mut cam_r := cga.perspective_camera(fv, 1.0, 0.1, 50.0, [baseline / 2.0, 0.0, cam_z]!, [
		baseline / 2.0,
		0.0,
		0.0,
	]!, [0.0, 1.0, 0.0]!)
	cam_r.look_at([baseline / 2.0, 0.0, 0.0]!, none)
	return r, cam_l, cam_r
}

// render_pair renders left/right frames for a scene.
fn render_pair(renderer cga.Renderer, sc cga.Scene, cam_l cga.PerspectiveCamera, cam_r cga.PerspectiveCamera) (mlx.Array, mlx.Array) {
	mut r := renderer
	fl := r.render(sc, cam_l)
	fr := r.render(sc, cam_r)
	return fl, fr
}

// cb_block builds one single-family replicate block → (162, 8) with the
// [kind, u, v, s, z, hue, lcol, ldir] layout (full discrete cartesian product).
fn cb_block(seed u64, extrap bool) mlx.Array {
	mut rng := new_rng(seed)
	mut rows := []f32{}
	for k in 0 .. n_kind {
		for h in 0 .. n_hue {
			for lc in 0 .. light_colors_len {
				for ld in 0 .. light_dirs_len {
					u, v, s, z := lcb_sample_free(mut rng, extrap, z_range_lo, z_range_hi)
					rows << f32(k)
					rows << f32(u)
					rows << f32(v)
					rows << f32(s)
					rows << f32(z)
					rows << f32(h)
					rows << f32(lc)
					rows << f32(ld)
				}
			}
		}
	}
	return mlx.array_f32(rows, [n_kind * n_hue * light_colors_len * light_dirs_len, 8])
}

// cb_sample returns (162×R, 8).
fn cb_sample(replicates int, seed u64, extrap bool) mlx.Array {
	mut blocks := []mlx.Array{}
	for r in 0 .. replicates {
		blocks << cb_block(seed * 1000 + u64(r), extrap)
	}
	return mlx.concatenate(blocks, 0)
}

// SceneFamily interface methods for the single family.
fn (cb Codebook) sample(replicates int, seed u64, extrap bool) mlx.Array {
	return cb_sample(replicates, seed, extrap)
}

fn (cb Codebook) n_combo() int {
	return n_kind * n_hue * light_colors_len * light_dirs_len
}

fn (cb Codebook) template_variant() string {
	return ''
}

fn (cb Codebook) geometry_family() string {
	return 'single'
}

fn (cb Codebook) template_lineage() TemplateLineage {
	return single_lineage()
}
