module conger

// model_memory.v — model memory / on-demand load + dynamic forgetting.
import math
import mlx

// split_save writes the whitening transform and component table to separate files.
pub fn split_save(m MixtureSPN, path string) (string, string) {
	t_path := path + '.transform.safetensors'
	c_path := path + '.components.safetensors'
	mut tens := mlx.new_map_string_to_array()
	tens.insert('f_mean', m.f_mean or { mlx.zeros([1], .float32) })
	tens.insert('basis', m.basis or { mlx.zeros([1, 1], .float32) })
	mlx.save_safetensors(t_path, tens, mlx.new_map_string_to_string())
	mlx.save_safetensors(c_path, m.spn_component_tensors(), m.spn_meta())
	return t_path, c_path
}

// load_transform loads only the whitening transform (f_mean, basis).
pub fn load_transform(path string) (mlx.Array, mlx.Array) {
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
pub struct ModelMeta {
pub:
	rel_floor f64
	cat_sizes []int
	n_stratum int
}

// load_components loads only the component table + metadata (no basis).
pub fn load_components(path string) (map[string]mlx.Array, ModelMeta) {
	mlx.use_cpu()
	tens, meta := mlx.load_safetensors(path + '.components.safetensors')
	mut comp := map[string]mlx.Array{}
	for key in ['log_w', 'f_mu', 'f_var', 't_mu', 'cat_logp'] {
		a := tens.get(key)
		a.eval()
		comp[key] = a
	}
	mlx.use_gpu()
	return comp, decode_spn_meta(meta)
}

// assemble builds a full MixtureSPN from split files.
pub fn assemble_model(path string) MixtureSPN {
	f_mean, basis := load_transform(path)
	comp, meta := load_components(path)
	mut m := MixtureSPN{
		log_w:     comp['log_w'] or { mlx.empty() }
		f_mu:      comp['f_mu'] or { mlx.empty() }
		f_var:     comp['f_var'] or { mlx.empty() }
		t_mu:      comp['t_mu'] or { mlx.empty() }
		cat_logp:  comp['cat_logp'] or { mlx.empty() }
		rel_floor: meta.rel_floor
		f_mean:    f_mean
		basis:     basis
		cat_sizes: meta.cat_sizes
		n_stratum: meta.n_stratum
	}
	repair_spn_defaults(mut m)
	m.init_norm()
	return m
}

// truncate_basis keeps the highest-variance d_max columns.
pub fn truncate_basis(m MixtureSPN, d_max int) MixtureSPN {
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
pub fn coreset(z mlx.Array, k int, rng mlx.Array) mlx.Array {
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
pub fn forget_components(m MixtureSPN, k_max int, policy string, seed u64) MixtureSPN {
	k := m.f_mu.dim(0)
	km := max_i(m.n_stratum, min_i(k_max, k))
	if km >= k {
		return m
	}
	stratum := m.cat_logp.take_axis(mlx.arange(0.0, f64(m.n_stratum), 1.0, .int32), 1).argmax_axis(1,
		false)
	rng := mlx.random_key(seed)
	// Exact proportional allocation (largest-remainder method): the kept total
	// must be exactly `km`. Per-stratum independent rounding could overshoot or
	// undershoot the target.
	mut sels := []mlx.Array{}
	mut njs := []int{}
	mut quotas := []f64{}
	for j in 0 .. m.n_stratum {
		sel := nonzero_indices(stratum.equal(mlx.int_scalar(j)))
		nj := sel.dim(0)
		if nj == 0 {
			// an empty stratum contributes no components (the proportional
			// split is over the *observed* strata only)
			continue
		}
		sels << sel
		njs << nj
		quotas << f64(km) * f64(nj) / f64(k)
	}
	ns := sels.len
	mut alloc := []int{len: ns}
	mut frac := []f64{len: ns}
	mut total := 0
	for i in 0 .. ns {
		mut base := int(quotas[i]) // floor quota
		if base < 1 {
			base = 1 // keep every observed stratum represented
		}
		if base > njs[i] {
			base = njs[i]
		}
		alloc[i] = base
		total += base
		frac[i] = quotas[i] - f64(int(quotas[i]))
	}
	// grow to exactly km by largest fractional remainder
	for total < km {
		mut best := -1
		mut best_frac := 0.0
		mut first := true
		for i in 0 .. ns {
			if alloc[i] < njs[i] && (first || frac[i] > best_frac) {
				best_frac = frac[i]
				best = i
				first = false
			}
		}
		if best < 0 {
			break
		}
		alloc[best]++
		total++
	}
	// trim back to exactly km (defensive: only if floor→at-least-1 overshot)
	for total > km {
		mut best := -1
		mut best_frac := 0.0
		mut first := true
		for i in 0 .. ns {
			if alloc[i] > 1 && (first || frac[i] < best_frac) {
				best_frac = frac[i]
				best = i
				first = false
			}
		}
		if best < 0 {
			break
		}
		alloc[best]--
		total--
	}
	mut kept := []int{}
	for i in 0 .. ns {
		sel := sels[i]
		kj := alloc[i]
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
pub fn model_size_mb(m MixtureSPN) f64 {
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
