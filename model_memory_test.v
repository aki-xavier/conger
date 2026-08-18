module conger

// model_memory_test.v — model memory/forgetting mechanism black-box tests.

import math

import os

import mlx

fn tiny_model(n int, v int, seed u64) MixtureSPN {
	rng := mlx.random_key(seed)
	f := mlx.random_normal([n, v], .float32, 0.0, 1.0, rng)
	t := mlx.random_normal([n, 3], .float32, 0.0, 1.0, rng)
	stratum := mlx.random_randint(mlx.int_scalar(0), mlx.int_scalar(3), [n], .int32,
		rng)
	scene := stratum.expand_dims(1).astype(.int32)
	return fit_mixture_spn(f, t, stratum, 1e-2, scene, [3], 0)
}

fn test_split_assemble_roundtrip() {
	m := tiny_model(60, 16, 0)
	path := os.temp_dir() + '/mm_roundtrip'
	split_save(m, path)
	m2 := assemble_model(path)
	mbasis := m.basis or { panic('') }
	m2basis := m2.basis or { panic('') }
	assert m2.f_mu.dim(0) == m.f_mu.dim(0)
	assert m2.f_mu.dim(1) == m.f_mu.dim(1)
	assert m2.f_mu.subtract(m.f_mu).abs().max().item_f32() < 1e-6
	assert m2basis.subtract(mbasis).abs().max().item_f32() < 1e-6
	assert m2.cat_sizes == m.cat_sizes
	assert m2.rel_floor == m.rel_floor
}

fn test_load_components_excludes_basis() {
	m := tiny_model(60, 16, 0)
	path := os.temp_dir() + '/mm_components'
	split_save(m, path)
	comp, meta := load_components(path)
	assert 'f_mu' in comp
	assert 'basis' !in comp
	assert meta.cat_sizes == [3]
	assert meta.rel_floor == 0.01
}

fn test_truncate_basis_keeps_high_variance_columns() {
	m := tiny_model(60, 16, 0)
	m2 := truncate_basis(m, 8)
	mbasis := m.basis or { panic('') }
	m2basis := m2.basis or { panic('') }
	assert m2basis.dim(1) == 8
	assert m2.f_mu.dim(1) == 8
	sel := mlx.arange(8.0, 16.0, 1.0, .int32)
	assert m2basis.subtract(mbasis.take_axis(sel, 1)).abs().max().item_f32() < 1e-6
	assert model_size_mb(m2) < model_size_mb(m)
}

fn test_truncate_basis_noop_when_d_max_too_large() {
	m := tiny_model(60, 16, 0)
	m2 := truncate_basis(m, 999)
	assert m2.f_mu.dim(1) == m.f_mu.dim(1)
}

fn test_forget_components_bounds_k_and_uniform_weights() {
	m := tiny_model(60, 16, 0)
	m2 := forget_components(m, 30, 'coreset', 0)
	assert m2.f_mu.dim(0) == 30
	assert math.abs(m2.log_w.exp().sum().item_f32() - 1.0) < 1e-4
	strat := m2.cat_logp.take_axis(mlx.arange(0.0, 3.0, 1.0, .int32), 1).argmax_axis(1,
		false)
	mut seen := map[int]bool{}
	for v in strat.data_i32() {
		seen[v] = true
	}
	assert seen.len == 3
}

fn test_forget_components_noop_when_k_max_large() {
	m := tiny_model(60, 16, 0)
	m2 := forget_components(m, 999, 'coreset', 0)
	assert m2.f_mu.dim(0) == m.f_mu.dim(0)
}

fn test_coreset_returns_distinct_indices() {
	z := mlx.random_normal([100, 5], .float32, 0.0, 1.0, mlx.random_key(0))
	idx := coreset(z, 10, mlx.random_key(1))
	vals := idx.data_i32()
	mut seen := map[int]bool{}
	for v in vals {
		seen[v] = true
		assert v >= 0 && v < 100
	}
	assert seen.len == 10
}
