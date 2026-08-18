module conger

// mixture_spn.v — MixtureSPN: full-resolution shallow mixture SPN (sum of
// instance-level diagonal-Gaussian blocks), V port of src/mixture_spn.py.
import math
import mlx

pub const nc = 64 // E-step sample block rows
pub const kc = 8 // E-step component block columns

pub struct MixtureSPN {
pub:
	rel_floor f64
	f_mean    ?mlx.Array // (V,)
	basis     ?mlx.Array // (V,D)
pub mut:
	log_w     mlx.Array // (K,)
	f_mu      mlx.Array // (K,D)
	f_var     mlx.Array // (K,D)
	t_mu      mlx.Array // (K,T)
	cat_logp  mlx.Array // (K, ΣC)
	cat_sizes []int
	n_stratum int
	norm      mlx.Array // (K,) feature-side normalising constant
}

pub const n_stratum_default = 3

pub fn (mut m MixtureSPN) init_norm() {
	log2pi := math.log(2.0 * math.pi)
	m.norm =
		m.log_w.subtract(m.f_var.log().add(mlx.f32_scalar(f32(log2pi))).sum_axis(1, false).multiply(mlx.f32_scalar(0.5)))
}

// z projects raw features into whitened coordinates (N,D).
pub fn (m MixtureSPN) z(f mlx.Array) mlx.Array {
	f_mean := m.f_mean or { panic('model missing whitening basis: cannot predict') }
	basis := m.basis or { panic('model missing whitening basis: cannot predict') }
	return f.subtract(f_mean.expand_dims(0)).matmul(basis)
}

// logq_feat computes the feature-side unnormalised log joint (N,K), blockwise.
pub fn (m MixtureSPN) logq_feat(z mlx.Array) mlx.Array {
	zn := z.dim(0)
	kn := m.f_mu.dim(0)
	mut out := []mlx.Array{}
	for i := 0; i < zn; i += nc {
		xb := slice_rows(z, i, min_i(i + nc, zn))
		mut parts := []mlx.Array{}
		for j := 0; j < kn; j += kc {
			je := min_i(j + kc, kn)
			fmu := slice_rows(m.f_mu, j, je)
			fvar := slice_rows(m.f_var, j, je)
			norm := slice_rows(m.norm, j, je)
			d := xb.expand_dims(1).subtract(fmu.expand_dims(0))
			q := norm.expand_dims(0).subtract(d.multiply(d).divide(fvar.expand_dims(0)).sum_axis(2,
				false).multiply(mlx.f32_scalar(0.5)))
			q.eval()
			parts << q
		}
		out << mlx.concatenate(parts, 1)
	}
	return mlx.concatenate(out, 0)
}

// tied_vars computes per-stratum tied diagonal variance.
pub fn tied_vars(z mlx.Array, stratum mlx.Array, rel_floor f64, n_stratum int) mlx.Array {
	mut out := []mlx.Array{}
	for j in 0 .. n_stratum {
		sel := nonzero_indices(stratum.equal(mlx.int_scalar(j)))
		if sel.dim(0) == 0 {
			out << mlx.ones([1, z.dim(1)], .float32)
			continue
		}
		zj := z.take_axis(sel, 0)
		vr := axis_var(zj, 0)
		fl :=
			axis_std(zj, 0).multiply(mlx.f32_scalar(f32(rel_floor))).square().add(mlx.f32_scalar(1e-8))
		out << vr.maximum(fl)
	}
	return mlx.concatenate(out, 0)
}

// cat_logp converts per-factor class columns into concatenated one-hot log probs.
pub fn cat_logp(classes mlx.Array, sizes []int) mlx.Array {
	mut cols := []mlx.Array{}
	for j, sz in sizes {
		cl := classes.take_axis(sel1(j), 1) // (N,1)
		ar := mlx.arange(0.0, f64(sz), 1.0, .int32).expand_dims(0) // (1,sz)
		eye := cl.equal(ar).astype(.float32) // (N,sz)
		cols << eye.log()
	}
	return mlx.concatenate(cols, 1)
}

// infer_cat_sizes maps the concatenated head width to per-factor class counts.
pub fn infer_cat_sizes(cat_width int) []int {
	if cat_width == n_stratum_default {
		return [n_stratum_default]
	}
	if cat_width == 24 {
		return [3, 3, 6, 6, 3, 3]
	}
	return [3, 6, 3, 3]
}

// fit assembles the instance-level mixture deterministically.
pub fn fit_mixture_spn(f mlx.Array, t mlx.Array, stratum mlx.Array, rel_floor f64, scene_classes mlx.Array, cat_sizes []int, basis_dim int) MixtureSPN {
	mut f_mean, mut basis, mut zz := whiten(f)
	if basis_dim >= 1 {
		d := zz.dim(1)
		bd := min_i(basis_dim, d)
		basis = basis.take_axis(mlx.arange(f64(d - bd), f64(d), 1.0, .int32), 1)
		zz = zz.take_axis(mlx.arange(f64(d - bd), f64(d), 1.0, .int32), 1)
	}
	n_stratum := cat_sizes[0]
	gvar := tied_vars(zz, stratum, rel_floor, n_stratum)
	mut mus := []mlx.Array{}
	mut vars_ := []mlx.Array{}
	mut tmus := []mlx.Array{}
	mut clps := []mlx.Array{}
	mut total_components := 0
	for j in 0 .. n_stratum {
		sel := nonzero_indices(stratum.equal(mlx.int_scalar(j)))
		nj := sel.dim(0)
		if nj == 0 {
			continue
		}
		total_components += nj
		zj := zz.take_axis(sel, 0)
		tj := t.take_axis(sel, 0)
		scj := scene_classes.take_axis(sel, 0)
		gj := gvar.take_axis(sel1(j), 0)
		mus << zj
		vars_ << gj.tile([nj, 1])
		tmus << tj
		clps << cat_logp(scj, cat_sizes)
	}
	mut m := MixtureSPN{
		log_w:     mlx.full_value([total_components], f32(-math.log(f64(total_components))),
			.float32)
		f_mu:      mlx.concatenate(mus, 0)
		f_var:     mlx.concatenate(vars_, 0)
		t_mu:      mlx.concatenate(tmus, 0)
		cat_logp:  mlx.concatenate(clps, 0)
		rel_floor: rel_floor
		f_mean:    f_mean
		basis:     basis
		cat_sizes: cat_sizes
		n_stratum: n_stratum
	}
	m.init_norm()
	return m
}

// fit_simple fits with a single-kind stratum (cat_sizes=[3]), matching the
// Python MixtureSPN.fit(f, t, stratum) default path.
pub fn fit_simple(f mlx.Array, t mlx.Array, stratum mlx.Array, basis_dim int) MixtureSPN {
	scene := stratum.expand_dims(1).astype(.int32)
	return fit_mixture_spn(f, t, stratum, 1e-2, scene, [n_stratum_default], basis_dim)
}

// expand_categories pads the category contract for new classes (logp=-inf).
pub fn (mut m MixtureSPN) expand_categories(new_sizes []int) {
	old := m.cat_sizes
	assert old.len == new_sizes.len
	mut cols := []mlx.Array{}
	mut lo := 0
	for i, o in old {
		nn := new_sizes[i]
		mut part := m.cat_logp.take_axis(mlx.arange(f64(lo), f64(lo + o), 1.0, .int32), 1)
		if nn > o {
			pad := mlx.full_value([m.cat_logp.dim(0), nn - o], f32(math.inf(-1)), .float32)
			part = mlx.concatenate([part, pad], 1)
		}
		cols << part
		lo += o
	}
	m.cat_logp = mlx.concatenate(cols, 1)
	m.cat_sizes = new_sizes
	m.n_stratum = new_sizes[0]
}

// add appends new sample components and re-estimates tied variance/weights.
pub fn (mut m MixtureSPN) add(f mlx.Array, t mlx.Array, stratum mlx.Array, scene_classes mlx.Array) {
	z_new := m.z(f)
	m.f_mu = mlx.concatenate([m.f_mu, z_new], 0)
	m.t_mu = mlx.concatenate([m.t_mu, t], 0)
	csizes := m.cat_sizes
	m.cat_logp = mlx.concatenate([m.cat_logp, cat_logp(scene_classes, csizes)], 0)
	n := m.f_mu.dim(0)
	n_new := scene_classes.dim(0)
	old_rows := slice_rows(m.cat_logp, 0, n - n_new)
	old_stratum := old_rows.take_axis(mlx.arange(0.0, f64(m.n_stratum), 1.0, .int32), 1).argmax_axis(1,
		false)
	s_all := mlx.concatenate([old_stratum, stratum.astype(.int32)], 0)
	gvar := tied_vars(m.f_mu, s_all, m.rel_floor, m.n_stratum)
	m.f_var = gvar.take_axis(s_all, 0)
	m.log_w = mlx.full_value([n], f32(-math.log(f64(n))), .float32)
	m.init_norm()
}

// whiten returns (mean, basis, whitened coords) via PCA (Gram eigendecomposition).
pub fn whiten(f mlx.Array) (mlx.Array, mlx.Array, mlx.Array) {
	f_mean := f.mean_axis(0, false)
	xc := f.subtract(f_mean.expand_dims(0))
	g := xc.matmul(xc.transpose())
	mut lam, mut u := eigh_cpu(g)
	// threshold computed in f64 then narrowed (matches Python's float(max(lam))*1e-6)
	maxlam := f32(f64(lam.max().item_f32()) * 1e-6)
	keep := nonzero_indices(lam.greater(mlx.f32_scalar(maxlam)))
	lam = lam.take(keep)
	u = u.take_axis(keep, 1)
	sq := lam.sqrt().expand_dims(0)
	basis := xc.transpose().matmul(u.divide(sq)).astype(.float32)
	z := xc.matmul(basis)
	return f_mean, basis, z
}

// predict returns (E[t|x], P(scene factors|x), responsibilities).
pub fn (m MixtureSPN) predict(f mlx.Array) (mlx.Array, mlx.Array, mlx.Array) {
	logq := m.logq_feat(m.z(f))
	r := logq.subtract(axis_logsumexp(logq, 1)).exp()
	t_mean := r.matmul(m.t_mu)
	cat_p := r.matmul(m.cat_logp.exp())
	return t_mean, cat_p, r
}

// save writes the model to a safetensors file.
pub fn (m MixtureSPN) save(path string) {
	mut tens := mlx.new_map_string_to_array()
	tens.insert('log_w', m.log_w)
	tens.insert('f_mu', m.f_mu)
	tens.insert('f_var', m.f_var)
	tens.insert('t_mu', m.t_mu)
	tens.insert('cat_logp', m.cat_logp)
	tens.insert('f_mean', m.f_mean or { mlx.array_f32([f32(0)], [1]) })
	tens.insert('basis', m.basis or { mlx.array_f32([f32(0)], [1, 1]) })
	mut meta := mlx.new_map_string_to_string()
	meta.insert('rel_floor', m.rel_floor.str())
	meta.insert('cat_sizes', encode_ints(m.cat_sizes))
	meta.insert('n_stratum', m.n_stratum.str())
	mlx.save_safetensors(path, tens, meta)
}

// load_mixture_spn reads a safetensors model back.
pub fn load_mixture_spn(path string) MixtureSPN {
	// safetensors tensors are mmap-lazy; force the Load ops onto the CPU stream
	// (a Load op cannot be evaluated on the GPU stream).
	mlx.use_cpu()
	tens, meta := mlx.load_safetensors(path)
	log_w := tens.get('log_w')
	f_mu := tens.get('f_mu')
	f_var := tens.get('f_var')
	t_mu := tens.get('t_mu')
	cat_logp := tens.get('cat_logp')
	f_mean := tens.get('f_mean')
	basis := tens.get('basis')
	for a in [log_w, f_mu, f_var, t_mu, cat_logp, f_mean, basis] {
		a.eval()
	}
	mlx.use_gpu()
	rel_floor := meta.get('rel_floor').f64()
	cat_sizes := decode_ints(meta.get('cat_sizes'))
	n_stratum := meta.get('n_stratum').int()
	mut m := MixtureSPN{
		log_w:     log_w
		f_mu:      f_mu
		f_var:     f_var
		t_mu:      t_mu
		cat_logp:  cat_logp
		rel_floor: rel_floor
		f_mean:    f_mean
		basis:     basis
		cat_sizes: cat_sizes
		n_stratum: n_stratum
	}
	if m.cat_sizes.len == 0 {
		m.cat_sizes = infer_cat_sizes(m.cat_logp.dim(1))
	}
	if m.n_stratum == 0 {
		m.n_stratum = m.cat_sizes[0]
	}
	m.init_norm()
	return m
}

pub fn min_i(a int, b int) int {
	return if a < b { a } else { b }
}

pub fn encode_ints(vals []int) string {
	mut parts := []string{cap: vals.len}
	for v in vals {
		parts << v.str()
	}
	return parts.join(',')
}

pub fn decode_ints(s string) []int {
	if s == '' {
		return []int{}
	}
	mut out := []int{}
	for p in s.split(',') {
		out << p.int()
	}
	return out
}
