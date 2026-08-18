module conger

// scm_proxy.v — appearance-mechanism proxy (multiplicative decomposition of the
// renderer appearance subgraph), V port of src/scm_proxy.py.
import mlx

struct AppearanceMechanism {
	n_hue    int = n_hue
	n_lcol   int = light_colors_len
	n_ldir   int = light_dirs_len
	max_iter int = 200
	tol      f64 = 1e-8
mut:
	albedo   ?mlx.Array // (n_hue, 3)
	lighting ?mlx.Array // (n_lcol, n_ldir, 3)
}

// fit estimates albedo/lighting via alternating least squares.
fn (mut m AppearanceMechanism) fit(rgb mlx.Array) AppearanceMechanism {
	mut a := rgb.mean_axes([1, 2], false)
	mut g := mlx.ones([m.n_lcol, m.n_ldir, 3], .float32)
	for _ in 0 .. m.max_iter {
		a_exp := a.expand_dims(1).expand_dims(1)
		den_g := a_exp.square().sum_axis(0, false).maximum(mlx.f32_scalar(1e-12))
		g_new := rgb.multiply(a_exp).sum_axis(0, false).divide(den_g)
		g_exp := g_new.expand_dims(0)
		den_a := g_exp.square().sum_axes([1, 2], false).maximum(mlx.f32_scalar(1e-12))
		a_new := rgb.multiply(g_exp).sum_axes([1, 2], false).divide(den_a)
		change := a_new.subtract(a).abs().max().item_f32() +
			g_new.subtract(g).abs().max().item_f32()
		a = a_new
		g = g_new
		if f64(change) < m.tol {
			break
		}
	}
	g_mean := g.mean_axes([0, 1], false).maximum(mlx.f32_scalar(1e-12))
	m.lighting = g.divide(g_mean.expand_dims(0).expand_dims(0))
	m.albedo = a.multiply(g_mean)
	return *m
}

// predict returns the proxy foreground colour (3,) for one (hue, lcol, ldir).
fn (m AppearanceMechanism) predict(hue int, lcol int, ldir int) mlx.Array {
	albedo := m.albedo or { panic('fit() before query') }
	lighting := m.lighting or { panic('fit() before query') }
	a_row := albedo.take_axis(sel1(hue), 0).squeeze_axis(0)
	l_row :=
		lighting.take_axis(sel1(lcol), 0).squeeze_axis(0).take_axis(sel1(ldir), 0).squeeze_axis(0)
	return a_row.multiply(l_row)
}

// do_lighting is the counterfactual do(lighting=…) query.
fn (m AppearanceMechanism) do_lighting(hue int, lcol int, ldir int, lcol_new int, ldir_new int) mlx.Array {
	return m.predict(hue, lcol_new, ldir_new)
}

// reconstruct returns the full proxy reconstruction.
fn (m AppearanceMechanism) reconstruct() mlx.Array {
	albedo := m.albedo or { panic('fit() before query') }
	lighting := m.lighting or { panic('fit() before query') }
	return albedo.expand_dims(1).expand_dims(1).multiply(lighting.expand_dims(0))
}

// reconstruction_error returns ||rgb − a⊗g|| / ||rgb||.
fn (m AppearanceMechanism) reconstruction_error(rgb mlx.Array) f64 {
	rec := m.reconstruct()
	num := rgb.subtract(rec).square().mean().sqrt().item_f32()
	den := rgb.square().mean().sqrt().item_f32()
	return f64(num) / fmax2(f64(den), 1e-12)
}

// albedo_invariance returns 1 − reconstruction error.
fn (m AppearanceMechanism) albedo_invariance(rgb mlx.Array) f64 {
	return 1.0 - m.reconstruction_error(rgb)
}

// foreground_mean_rgb returns the foreground-weighted average RGB (3,).
fn foreground_mean_rgb(frame mlx.Array, weights mlx.Array) mlx.Array {
	mut w := weights
	if weights.ndim() == 2 {
		w = weights.expand_dims(-1)
	}
	frame_rgb := frame.take_axis(mlx.arange(0.0, 3.0, 1.0, .int32), -1).astype(.float32)
	num := w.multiply(frame_rgb).sum_axes([0, 1], false)
	den := w.sum().maximum(mlx.f32_scalar(1e-8))
	return num.divide(den)
}

fn fmax2(a f64, b f64) f64 {
	return if a > b { a } else { b }
}
