module conger

// mlxutil.v — MLX helpers that stay conger-local.
//
// The generic helpers that used to live here (boolean-mask indexing,
// axis-wise logsumexp, key splitting, small array constructors, …) moved into
// the `mlx-v` bindings (module `mlx`) and are called as `mlx.<name>`. What
// remains is the CPU/GPU stream-switch policy for eigendecomposition, which
// is a call-site decision and therefore stays with the caller.
import mlx

// eigh_cpu returns (eigenvalues ascending, eigenvectors) of a symmetric matrix,
// evaluated on the CPU stream (MLX has no GPU eigendecomposition).
pub fn eigh_cpu(g mlx.Array) (mlx.Array, mlx.Array) {
	mlx.use_cpu()
	lam, u := g.eigh('L')
	mlx.use_gpu()
	return lam, u
}
