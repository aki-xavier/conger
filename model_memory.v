module conger

// model_memory.v — model memory / on-demand load + dynamic forgetting
// (V port of src/model_memory.py).
import math
import mlx

// split_save writes the whitening transform and component table to separate files.
fn split_save(m MixtureSPN, path string) (string, string) {
	t_path := path + '.transform.safetensors'
	c_path := path + '.components.safetensors'
	mut tens := mlx.new_map_string_to_array()
	tens.insert('f_mean', m.f_mean or { mlx.zeros([1], .float32) })
	tens.insert('basis', m.basis or { mlx.zeros([1, 1], .float32) })
	mlx.save_safetensors(t_path, tens, mlx.new_map_string_to_string())
	mut ctens := mlx.new_map_string_to_array()
	ctens.insert('log_w', m.log_w)
	ctens.insert('f_mu', m.f_mu)
	ctens.insert('f_var', m.f_var)
	ctens.insert('t_mu', m.t_mu)
	ctens.insert('cat_logp', m.cat_logp)
	mut meta := mlx.new_map_string_to_string()
	meta.insert('rel_floor', m.rel_floor.str())
	meta.insert('cat_sizes', encode_ints(m.cat_sizes))
	meta.insert('n_stratum', m.n_stratum.str())
	mlx.save_safetensors(c_path, ctens, meta)
	return t_path, c_path
}

// load_transform loads only the whitening transform (f_mean, basis).
fn load_transform(path string) (mlx.Array, mlx.Array) {
	mlx.use_cpu()
	tens, _ := mlx.load_safetensors(path + '.transform.safetensors')
	f_mean := tens.get('f_mean')
	basis := tens.get('basis')
	f_mean.eval()
	basis.eval()
	mlx.use_gpu()
	return f_mean, basis
}

// ModelMeta holds the component-file metadata.
struct ModelMeta {
	rel_floor f64
	cat_sizes []int
	n_stratum int
}

// load_components loads only the component table + metadata (no basis).
fn load_components(path string) (map[string]mlx.Array, ModelMeta) {
	mlx.use_cpu()
	tens, meta := mlx.load_safetensors(path + '.components.safetensors')
	mut comp := map[string]mlx.Array{}
	for key in ['log_w', 'f_mu', 'f_var', 't_mu', 'cat_logp'] {
		a := tens.get(key)
		a.eval()
		comp[key] = a
	}
	mlx.use_gpu()
	out_meta := ModelMeta{
		rel_floor: meta.get('rel_floor').f64()
		cat_sizes: decode_ints(meta.get('cat_sizes'))
		n_stratum: meta.get('n_stratum').int()
	}
	return comp, out_meta
}

// assemble builds a full MixtureSPN from split files.
fn assemble_model(path string) MixtureSPN {
	f_mean, basis := load_transform(path)
	comp, meta := load_components(path)
	mut csizes := meta.cat_sizes.clone()
	if csizes.len == 0 {
		csizes = [3]
	}
	mut m := MixtureSPN{
		log_w:     comp['log_w']
		f_mu:      comp['f_mu']
		f_var:     comp['f_var']
		t_mu:      comp['t_mu']
		cat_logp:  comp['cat_logp']
		rel_floor: meta.rel_floor
		f_mean:    f_mean
		basis:     basis
		cat_sizes: csizes
		n_stratum: meta.n_stratum
	}
	m.init_norm()
	return m
}

// truncate_basis keeps the highest-variance d_max columns.
fn truncate_basis(m MixtureSPN, d_max int) MixtureSPN {
	d := m.f_mu.dim(1)
	dm := min_i(max_i(1, d_max), d)
	if dm == d {
		return m
	}
	basis := m.basis or { panic('missing basis') }
	sel := mlx.arange(f64(d - dm), f64(d), 1.0, .int32)
	mut out := MixtureSPN{
		log_w:     m.log_w
		f_mu:      m.f_mu.take_axis(sel, 1)
		f_var:     m.f_var.take_axis(sel, 1)
		t_mu:      m.t_mu
		cat_logp:  m.cat_logp
		rel_floor: m.rel_floor
		f_mean:    m.f_mean
		basis:     basis.take_axis(sel, 1)
		cat_sizes: m.cat_sizes
		n_stratum: m.n_stratum
	}
	out.init_norm()
	return out
}

// coreset returns k farthest-point row indices of Z.
fn coreset(z mlx.Array, k int, rng mlx.Array) mlx.Array {
	n := z.dim(0)
	if k >= n {
		return mlx.arange(0.0, f64(n), 1.0, .int32)
	}
	start := mlx.random_randint(mlx.int_scalar(0), mlx.int_scalar(n), [1], .int32, rng).data_i32()[0]
	mut sel := [start]
	zstart := z.take_axis(sel1(start), 0) // (1,D)
	mut d2 := z.subtract(zstart).square().sum_axis(1, false)
	for sel.len < k {
		nxt := d2.argmax().item_i32()
		sel << nxt
		znxt := z.take_axis(sel1(nxt), 0) // (1,D)
		d2 = d2.minimum(z.subtract(znxt).square().sum_axis(1, false))
	}
	mut vals := []i32{len: sel.len}
	for i, v in sel {
		vals[i] = i32(v)
	}
	return mlx.array_i32(vals, [vals.len])
}

// forget_components bounds K to k_max (coreset/random), per-stratum proportional.
fn forget_components(m MixtureSPN, k_max int, policy string, seed u64) MixtureSPN {
	k := m.f_mu.dim(0)
	km := max_i(m.n_stratum, min_i(k_max, k))
	if km >= k {
		return m
	}
	stratum := m.cat_logp.take_axis(mlx.arange(0.0, f64(m.n_stratum), 1.0, .int32), 1).argmax_axis(1,
		false)
	rng := mlx.random_key(seed)
	mut kept := []int{}
	for j in 0 .. m.n_stratum {
		sel := nonzero_indices(stratum.equal(mlx.int_scalar(j)))
		nj := sel.dim(0)
		mut kj := max_i(1, py_round(f64(km) * f64(nj) / f64(k)))
		kj = min_i(kj, nj)
		mut pick := mlx.Array{}
		if policy == 'random' {
			pick = mlx.random_permutation(sel, 0, rng).take(mlx.arange(0.0, f64(kj), 1.0, .int32))
		} else {
			pick = sel.take(coreset(m.f_mu.take_axis(sel, 0), kj, rng))
		}
		for v in pick.data_i32() {
			kept << v
		}
	}
	kept.sort()
	mut kvals := []i32{len: kept.len}
	for i, v in kept {
		kvals[i] = i32(v)
	}
	idx := mlx.array_i32(kvals, [kvals.len])
	s_kept := stratum.take_axis(idx, 0)
	gvar := tied_vars(m.f_mu.take_axis(idx, 0), s_kept, m.rel_floor, m.n_stratum)
	f_var := gvar.take_axis(s_kept, 0)
	log_w := mlx.full_value([kept.len], f32(-math.log(f64(kept.len))), .float32)
	mut out := MixtureSPN{
		log_w:     log_w
		f_mu:      m.f_mu.take_axis(idx, 0)
		f_var:     f_var
		t_mu:      m.t_mu.take_axis(idx, 0)
		cat_logp:  m.cat_logp.take_axis(idx, 0)
		rel_floor: m.rel_floor
		f_mean:    m.f_mean
		basis:     m.basis
		cat_sizes: m.cat_sizes
		n_stratum: m.n_stratum
	}
	out.init_norm()
	return out
}

// model_size_mb returns the total tensor bytes in MB.
fn model_size_mb(m MixtureSPN) f64 {
	mut tot := usize(0)
	tot += m.log_w.size() * m.log_w.itemsize()
	tot += m.f_mu.size() * m.f_mu.itemsize()
	tot += m.f_var.size() * m.f_var.itemsize()
	tot += m.t_mu.size() * m.t_mu.itemsize()
	tot += m.cat_logp.size() * m.cat_logp.itemsize()
	if fm := m.f_mean {
		tot += fm.size() * fm.itemsize()
	}
	if bs := m.basis {
		tot += bs.size() * bs.itemsize()
	}
	return f64(tot) / 1e6
}

fn max_i(a int, b int) int {
	return if a > b { a } else { b }
}
