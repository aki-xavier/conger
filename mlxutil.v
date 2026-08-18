module conger

// mlxutil.v — small MLX helpers used across the mlx-based modules.
//
// The Python reference uses `mlx.core` (`mx`) arrays throughout; the V port
// uses the `mlx-v` bindings (module `mlx`). This file centralises the few
// idioms that mlx-v does not expose ergonomically: boolean-mask indexing,
// axis-wise variance/std, axis-wise logsumexp and CPU-only eigendecomposition.
import mlx

// nonzero_indices returns the flat int32 indices where a boolean array is true.
// Port of Utils.nonzero (the argsort trick: MLX has no boolean indexing).
pub fn nonzero_indices(sel mlx.Array) mlx.Array {
	n := int(sel.size())
	flat := sel.reshape([n])
	k := int(flat.astype(.float32).sum().item_f32())
	key := mlx.where(flat, mlx.arange(0.0, f64(n), 1.0, .float32), mlx.full_value([n], f32(n),
		.float32))
	return key.argsort().take(mlx.arange(0.0, f64(k), 1.0, .int32))
}

// axis_var returns the population variance along `axis` (ddof=0), keepdims.
pub fn axis_var(x mlx.Array, axis int) mlx.Array {
	return x.var_axis(axis, true, 0)
}

// axis_std returns the population standard deviation along `axis`, keepdims.
pub fn axis_std(x mlx.Array, axis int) mlx.Array {
	return x.std_axis(axis, true, 0)
}

// axis_logsumexp returns log(Σ exp(x)) along `axis`, keepdims, computed stably.
pub fn axis_logsumexp(x mlx.Array, axis int) mlx.Array {
	m := x.max_axis(axis, true)
	return m.add(x.subtract(m).exp().sum_axis(axis, true).log())
}

// split_keys splits a PRNG key into `num` independent keys (matching the Python
// reference's mx.random.split(key, num)).
pub fn split_keys(seed u64, num int) []mlx.Array {
	keys := mlx.random_split_n(mlx.random_key(seed), num)
	mut out := []mlx.Array{len: num}
	for i in 0 .. num {
		out[i] = keys.take_axis(sel1(i), 0).squeeze_axis(0)
	}
	return out
}

// arr32 builds a float32 array from f64 literals (avoiding f32() casts everywhere).
pub fn arr32(vals []f64, shape []int) mlx.Array {
	mut f := []f32{len: vals.len}
	for i, v in vals {
		f[i] = f32(v)
	}
	return mlx.array_f32(f, shape)
}

// sel1 returns a length-1 int32 index array selecting element `n` along an axis.
pub fn sel1(n int) mlx.Array {
	return mlx.array_i32([i32(n)], [1])
}

// col returns a true 1-D column j of an array (take_axis keeps the index dim,
// so squeeze it away to match Python's a[:, j]).
pub fn col(a mlx.Array, j int) mlx.Array {
	return a.take_axis(sel1(j), 1).squeeze_axis(1)
}

// slice_rows returns rows [start, end) of a 2-D array.
pub fn slice_rows(x mlx.Array, start int, end int) mlx.Array {
	return x.take_axis(mlx.arange(f64(start), f64(end), 1.0, .int32), 0)
}

// eigh_cpu returns (eigenvalues ascending, eigenvectors) of a symmetric matrix,
// evaluated on the CPU stream (MLX has no GPU eigendecomposition).
pub fn eigh_cpu(g mlx.Array) (mlx.Array, mlx.Array) {
	mlx.use_cpu()
	lam, u := g.eigh('L')
	mlx.use_gpu()
	return lam, u
}

// roll_axis rolls a 2-D array along `axis` by `shift` (wrapping), matching
// numpy's mx.roll(lum, shift, axis). shift = -1 pulls the next row/column in.
pub fn roll_axis(x mlx.Array, shift int, axis int) mlx.Array {
	n := x.dim(axis)
	mut idx := []i32{len: n}
	for i in 0 .. n {
		mut j := i - shift
		j %= n
		if j < 0 {
			j += n
		}
		idx[i] = i32(j)
	}
	return x.take_axis(mlx.array_i32(idx, [n]), axis)
}

// overwrite_region returns a copy of `base` with the region at rows
// [top, top+patch.dim(0)) and cols [left, left+patch.dim(1)) replaced by
// `patch` (both must share all trailing dims beyond axis 1). This is the
// immutable-mlx replacement for numpy's `base[top:top+ph, left:left+pw] = patch`.
pub fn overwrite_region(base mlx.Array, patch mlx.Array, top int, left int) mlx.Array {
	h := base.dim(0)
	w := base.dim(1)
	ih := patch.dim(0)
	iw := patch.dim(1)
	top_rows := base.take_axis(mlx.arange(0.0, f64(top), 1.0, .int32), 0)
	bottom_rows := base.take_axis(mlx.arange(f64(top + ih), f64(h), 1.0, .int32), 0)
	mid_rows := base.take_axis(mlx.arange(f64(top), f64(top + ih), 1.0, .int32), 0)
	left_cols := mid_rows.take_axis(mlx.arange(0.0, f64(left), 1.0, .int32), 1)
	right_cols := mid_rows.take_axis(mlx.arange(f64(left + iw), f64(w), 1.0, .int32), 1)
	mid := mlx.concatenate([left_cols, patch, right_cols], 1)
	return mlx.concatenate([top_rows, mid, bottom_rows], 0)
}

// complex_from builds a complex64 array re + i·im from two real arrays.
pub fn complex_from(re mlx.Array, im mlx.Array) mlx.Array {
	re_c := re.astype(.complex64)
	im_c := im.astype(.complex64)
	i := mlx.complex_scalar(0.0, 1.0)
	return re_c.add(i.multiply(im_c))
}

// fft2 returns the 2-D complex FFT (backward norm, matching mx.fft.fft2).
pub fn fft2(x mlx.Array) mlx.Array {
	return x.fftn([x.dim(0), x.dim(1)], [0, 1], .backward)
}

// ifft2 returns the 2-D complex inverse FFT (backward norm, matching mx.fft.ifft2).
pub fn ifft2(x mlx.Array) mlx.Array {
	return x.ifftn([x.dim(0), x.dim(1)], [0, 1], .backward)
}

// pad_edge pads a 2-D array by `p` on both axes with edge mode.
pub fn pad_edge(x mlx.Array, p int) mlx.Array {
	return x.pad([0, 1], [p, p], [p, p], mlx.f32_scalar(0.0), 'edge')
}
