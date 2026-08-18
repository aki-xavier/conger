module conger

// mixture_spn_test.v — MixtureSPN black-box tests (axioms / instance regression /
// whitening pathology / serialisation).
import math
import os
import mlx

fn manual_model() (MixtureSPN, mlx.Array) {
	f_mu := arr32([0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0], [3, 4])
	f_var := mlx.full_value([3, 4], 0.01, .float32)
	t_mu := arr32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
	cat_logp := arr32([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [3, 3]).log()
	log_w := mlx.full_value([3], f32(-math.log(3.0)), .float32)
	mut m := MixtureSPN{
		log_w:     log_w
		f_mu:      f_mu
		f_var:     f_var
		t_mu:      t_mu
		cat_logp:  cat_logp
		rel_floor: 0.1
		f_mean:    mlx.zeros([4], .float32)
		basis:     mlx.eye(4, 4, 0, .float32)
		cat_sizes: [3]
		n_stratum: 3
	}
	m.init_norm()
	return m, t_mu
}

fn separable_data() (mlx.Array, mlx.Array, mlx.Array) {
	n_per := 200
	d_f := 6
	d_t := 2
	k_a, k_b := mlx.random_split(mlx.random_key(0))
	k0, k1 := mlx.random_split(k_a)
	k2, _ := mlx.random_split(k_b)
	true_fmu := mlx.random_normal([3, d_f], .float32, 0.0, 1.0, k0).multiply(mlx.f32_scalar(3.0))
	true_tmu := arr32([0.0, 0.0, 8.0, 8.0, -8.0, 8.0], [3, 2])
	mut fs := []mlx.Array{}
	mut ts := []mlx.Array{}
	mut ks := []mlx.Array{}
	for c in 0 .. 3 {
		center := true_fmu.take_axis(sel1(c), 0)
		fs << center.add(mlx.random_normal([n_per, d_f], .float32, 0.0, 0.3, k1))
		tc := true_tmu.take_axis(sel1(c), 0)
		ts << tc.add(mlx.random_normal([n_per, d_t], .float32, 0.0, 0.1, k2))
		ks << mlx.full_value([n_per], f32(c % 3), .float32)
	}
	return mlx.concatenate(fs, 0), mlx.concatenate(ts, 0), mlx.concatenate(ks, 0)
}

fn allclose(a mlx.Array, b mlx.Array, atol f32) bool {
	return a.subtract(b).abs().max().item_f32() < atol
}

fn all_equal(a mlx.Array, b mlx.Array) bool {
	return a.equal(b).all().item_bool()
}

fn test_axioms() {
	m, t_mu := manual_model()
	xs := arr32([0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0], [2, 4])
	tm, kp, r := m.predict(xs)
	assert allclose(r.sum_axis(1, false), mlx.ones([2], .float32), 1e-4)
	assert allclose(kp.sum_axis(1, false), mlx.ones([2], .float32), 1e-4)
	tm0 := tm.take_axis(sel1(0), 0).squeeze_axis(0)
	t0 := t_mu.take_axis(sel1(0), 0).squeeze_axis(0)
	assert allclose(tm0, t0, 1e-3)
	assert kp.data_f32()[1 * 3 + 0] > 0.999

	mut m1 := MixtureSPN{
		log_w:     mlx.zeros([1], .float32)
		f_mu:      slice_rows(m.f_mu, 0, 1)
		f_var:     slice_rows(m.f_var, 0, 1)
		t_mu:      slice_rows(t_mu, 0, 1)
		cat_logp:  slice_rows(m.cat_logp, 0, 1)
		rel_floor: 0.1
		f_mean:    mlx.zeros([4], .float32)
		basis:     mlx.eye(4, 4, 0, .float32)
		cat_sizes: [3]
		n_stratum: 3
	}
	m1.init_norm()
	tm1, _, _ := m1.predict(arr32([99.0, 99.0, 99.0, 99.0], [1, 4]))
	tm1_0 := tm1.take_axis(sel1(0), 0).squeeze_axis(0)
	assert allclose(tm1_0, t0, 1e-4)
}

fn test_instance_regression() {
	f, t, k := separable_data()
	m := fit_simple(f, t, k, 0)
	tm, kp, _ := m.predict(f)
	rmse := tm.subtract(t).square().mean().sqrt().item_f32()
	assert rmse < 0.5
	acc := kp.argmax_axis(1, false).astype(.float32).equal(k).astype(.float32).mean().item_f32()
	assert acc > 0.99
}

fn test_full_scene_heads() {
	f := arr32([0.0, 0.0, 10.0, 0.0, 0.0, 10.0], [3, 2])
	t := mlx.zeros([3, 1], .float32)
	scene := mlx.array_i32([i32(0), 1, 0, 2, 1, 5, 1, 0, 2, 3, 2, 1], [3, 4])
	stratum := col(scene, 0)
	m := fit_mixture_spn(f, t, stratum, 1e-5, scene, [3, 6, 3, 3], 0)
	_, cp, _ := m.predict(f)
	mut lo := 0
	for sz in [3, 6, 3, 3] {
		p := cp.take_axis(mlx.arange(f64(lo), f64(lo + sz), 1.0, .int32), 1)
		assert allclose(p.sum_axis(1, false), mlx.ones([3], .float32), 1e-3)
		lo += sz
	}
	mut cols := []mlx.Array{}
	mut lo2 := 0
	for sz in [3, 6, 3, 3] {
		cols << cp.take_axis(mlx.arange(f64(lo2), f64(lo2 + sz), 1.0, .int32), 1).argmax_axis(1,
			false).expand_dims(1)
		lo2 += sz
	}
	got := mlx.concatenate(cols, 1)
	assert all_equal(got.astype(.int32), scene)
}

fn test_incremental_add() {
	f, t, k := separable_data()
	even := mlx.arange(0.0, 600.0, 2.0, .int32)
	odd := mlx.arange(1.0, 600.0, 2.0, .int32)
	mut m := fit_simple(f.take_axis(even, 0), t.take_axis(even, 0), k.take_axis(even, 0), 0)
	f_odd := f.take_axis(odd, 0)
	t_odd := t.take_axis(odd, 0)
	k_odd := k.take_axis(odd, 0)
	m.add(f_odd, t_odd, k_odd, k_odd.expand_dims(1))
	assert m.f_mu.dim(0) == 600
	tm, kp, _ := m.predict(f)
	rmse := tm.subtract(t).square().mean().sqrt().item_f32()
	assert rmse < 0.5
	acc := kp.argmax_axis(1, false).astype(.float32).equal(k).astype(.float32).mean().item_f32()
	assert acc > 0.99
}

fn test_fit_basis_dim_truncates_high_variance() {
	f, t, k := separable_data()
	full := fit_simple(f, t, k, 0)
	m := fit_simple(f, t, k, 3)
	assert m.f_mu.dim(1) == 3
	full_basis := full.basis or { panic('') }
	mbasis := m.basis or { panic('') }
	assert mbasis.dim(1) == 3
	d := full.f_mu.dim(1)
	sel := mlx.arange(f64(d - 3), f64(d), 1.0, .int32)
	assert all_equal(mbasis, full_basis.take_axis(sel, 1))
	assert all_equal(m.f_mu, full.f_mu.take_axis(sel, 1))
	assert allclose(m.f_var, full.f_var.take_axis(sel, 1), 1e-6)
}

fn test_fit_basis_dim_nonpositive_means_full() {
	f, t, k := separable_data()
	full := fit_simple(f, t, k, 0)
	for bd in [0, -1] {
		m := fit_simple(f, t, k, bd)
		assert m.f_mu.dim(1) == full.f_mu.dim(1)
	}
}

fn test_correlation_pathology() {
	n := 120
	k_a, k_b := mlx.random_split(mlx.random_key(5))
	direction := arr32([1.0, 1.0, 1.0, 1.0], [4]).divide(mlx.f32_scalar(2.0))
	perp := arr32([1.0, -1.0, 1.0, -1.0], [4]).divide(mlx.f32_scalar(2.0))
	lo := mlx.random_normal([n, 1], .float32, 0.0, 1.0, k_a).multiply(mlx.f32_scalar(3.0))
	pe := mlx.random_normal([n, 1], .float32, 0.0, 1.0, k_b).multiply(mlx.f32_scalar(0.1))
	ar := mlx.arange(0.0, f64(n), 1.0, .float32)
	off :=
		mlx.where(ar.less(mlx.f32_scalar(f32(n / 2))), mlx.f32_scalar(0.0), mlx.f32_scalar(0.6)).expand_dims(1)
	f :=
		lo.multiply(direction.expand_dims(0)).add(pe.multiply(perp.expand_dims(0))).add(off.multiply(perp.expand_dims(0)))
	k := ar.greater_equal(mlx.f32_scalar(f32(n / 2))).astype(.int32)
	t := mlx.zeros([n, 1], .float32)
	m := fit_simple(f, t, k, 0)
	_, kp, _ := m.predict(f)
	acc :=
		kp.argmax_axis(1, false).astype(.float32).equal(k.astype(.float32)).astype(.float32).mean().item_f32()
	assert acc > 0.95
}

fn test_serialization_roundtrip() {
	f, t, k := separable_data()
	m := fit_simple(f, t, k, 0)
	tm, kp, _ := m.predict(f)
	path := os.temp_dir() + '/m_spn_roundtrip.safetensors'
	m.save(path)
	loaded := load_mixture_spn(path)
	tm2, kp2, _ := loaded.predict(f)
	assert all_equal(tm, tm2)
	assert all_equal(kp, kp2)
}

fn test_category_contract_expansion() {
	f := arr32([0.0, 0.0, 8.0, 0.0, 0.0, 8.0], [3, 2])
	t := mlx.zeros([3, 1], .float32)
	cls := mlx.array_i32([i32(0), 0, 1, 0, 2, 1], [3, 2])
	stratum := col(cls, 0)
	m := fit_mixture_spn(f, t, stratum, 1e-2, cls, [3, 2], 0)
	path := os.temp_dir() + '/m_spn_cat.safetensors'
	m.save(path)
	mut loaded := load_mixture_spn(path)
	assert loaded.cat_sizes == [3, 2]
	loaded.expand_categories([4, 3])
	assert loaded.cat_sizes == [4, 3]
	assert loaded.cat_logp.dim(1) == 7
	_, cp, _ := loaded.predict(f)
	assert cp.data_f32()[0 * 7 + 3] == 0.0
	cols45 := cp.take_axis(mlx.arange(4.0, 7.0, 1.0, .int32), 1)
	assert allclose(cols45.sum_axis(1, false), mlx.ones([3], .float32), 1e-4)

	f_new := arr32([30.0, 30.0], [1, 2])
	c_new := mlx.array_i32([i32(3), 2], [1, 2])
	loaded.add(f_new, mlx.zeros([1, 1], .float32), col(c_new, 0), c_new)
	_, cp_new, _ := loaded.predict(f_new)
	assert cp_new.take_axis(mlx.arange(0.0, 4.0, 1.0, .int32), 1).argmax_axis(1, false).item_i32() == 3
	assert cp_new.take_axis(mlx.arange(4.0, 7.0, 1.0, .int32), 1).argmax_axis(1, false).item_i32() == 2
}
