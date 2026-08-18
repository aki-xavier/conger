module conger

// vecmath.v — pure-f64 array/numerical helpers.
//
// The GenericEM framework and its instances need a small set of vectorised
// numerical primitives (linspace / arange / diff / roll / 2x2 & n×n solves /
// least squares / 2D Kabsch), plus a deterministic xorshift64* PRNG used by
// the forward models' `sample`.
//
// Exact RNG stream parity across implementations is intentionally NOT required:
// the test suite checks statistical convergence within tolerances, not
// bit-identical noise realisations.
import math

// Rng is a deterministic xorshift64* generator (statistical stand-in for
// numpy.random.default_rng).
pub struct Rng {
pub mut:
	state u64
}

// new_rng creates a deterministic generator from a 64-bit seed (0 is remapped
// so that seed 0 is still usable).
pub fn new_rng(seed u64) Rng {
	mut s := seed
	if s == 0 {
		s = 0x9e3779b97f4a7c15
	}
	return Rng{
		state: s
	}
}

@[inline]
pub fn (mut r Rng) next_u64() u64 {
	mut x := r.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.state = x
	return x * 2685821657736338717
}

// f64 returns a uniform float in [0, 1).
@[inline]
pub fn (mut r Rng) f64() f64 {
	return f64(r.next_u64() >> 11) * (1.0 / 9007199254740992.0)
}

// uniform returns a uniform float in [lo, hi).
pub fn (mut r Rng) uniform(lo f64, hi f64) f64 {
	return lo + (hi - lo) * r.f64()
}

// normal returns a normally distributed value via Box-Muller.
pub fn (mut r Rng) normal(mu f64, sigma f64) f64 {
	u1 := math.max(r.f64(), 1e-12)
	u2 := r.f64()
	z := math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
	return mu + sigma * z
}

// linspace returns n evenly spaced values in [start, stop].
pub fn linspace(start f64, stop f64, n int) []f64 {
	mut out := []f64{len: n}
	if n == 1 {
		out[0] = start
		return out
	}
	step := (stop - start) / f64(n - 1)
	for i in 0 .. n {
		out[i] = start + step * f64(i)
	}
	return out
}

// arange_f64 returns [0, 1, ..., n-1] as f64.
pub fn arange_f64(n int) []f64 {
	mut out := []f64{len: n}
	for i in 0 .. n {
		out[i] = f64(i)
	}
	return out
}

// diff returns forward differences a[i+1]-a[i].
pub fn diff(a []f64) []f64 {
	mut out := []f64{len: a.len - 1}
	for i in 0 .. a.len - 1 {
		out[i] = a[i + 1] - a[i]
	}
	return out
}

// roll shifts a 1D array by `shift` positions (wraps around), like np.roll(a, shift, axis=0).
pub fn roll(a []f64, shift int) []f64 {
	n := a.len
	mut out := []f64{len: n}
	for i in 0 .. n {
		j := ((i - shift) % n + n) % n
		out[i] = a[j]
	}
	return out
}

// solve_2x2 solves A x = b for a 2x2 matrix given row-major.
pub fn solve_2x2(a [][]f64, b []f64) []f64 {
	det := a[0][0] * a[1][1] - a[0][1] * a[1][0]
	if math.abs(det) < 1e-15 {
		panic('solve_2x2: singular matrix (det ≈ 0)')
	}
	return [
		(b[0] * a[1][1] - b[1] * a[0][1]) / det,
		(a[0][0] * b[1] - a[1][0] * b[0]) / det,
	]
}

// solve_n solves a dense n×n linear system via Gaussian elimination with
// partial pivoting.
pub fn solve_n(a [][]f64, b []f64) []f64 {
	n := b.len
	mut aug := [][]f64{len: n, init: []f64{len: n + 1}}
	for i in 0 .. n {
		for j in 0 .. n {
			aug[i][j] = a[i][j]
		}
		aug[i][n] = b[i]
	}
	for col in 0 .. n {
		mut piv := col
		mut best := math.abs(aug[col][col])
		for r in col + 1 .. n {
			if math.abs(aug[r][col]) > best {
				best = math.abs(aug[r][col])
				piv = r
			}
		}
		if best < 1e-12 {
			panic('solve_n: singular matrix (zero pivot at column ${col})')
		}
		if piv != col {
			aug[col], aug[piv] = aug[piv], aug[col]
		}
		for r in col + 1 .. n {
			f := aug[r][col] / aug[col][col]
			for c in col .. n + 1 {
				aug[r][c] -= f * aug[col][c]
			}
		}
	}
	mut x := []f64{len: n}
	for i := n - 1; i >= 0; i-- {
		mut s := aug[i][n]
		for j := i + 1; j < n; j++ {
			s -= aug[i][j] * x[j]
		}
		x[i] = s / aug[i][i]
	}
	return x
}

// lstsq_2 solves a two-column least-squares problem via normal equations
// (XᵀX) c = Xᵀy. Used by the 2-parameter linear fits (Retinex, occlusion).
pub fn lstsq_2(x [][]f64, y []f64) []f64 {
	n := x.len
	mut xtx := [][]f64{len: 2, init: []f64{len: 2}}
	mut xty := []f64{len: 2}
	for i in 0 .. n {
		x0 := x[i][0]
		x1 := x[i][1]
		xtx[0][0] += x0 * x0
		xtx[0][1] += x0 * x1
		xtx[1][0] += x0 * x1
		xtx[1][1] += x1 * x1
		xty[0] += x0 * y[i]
		xty[1] += x1 * y[i]
	}
	return solve_2x2(xtx, xty)
}

// kabsch_2d returns the least-squares rigid transform (R, t) mapping
// corresponding points p → q (q ≈ R·p + t). For 2D the optimal rotation angle
// is atan2(H01-H10, H00+H11), which is exactly the SVD-based Kabsch result.
pub fn kabsch_2d(p [][]f64, q [][]f64) ([][]f64, []f64) {
	n := p.len
	mut pm := []f64{len: 2}
	mut qm := []f64{len: 2}
	for i in 0 .. n {
		pm[0] += p[i][0]
		pm[1] += p[i][1]
		qm[0] += q[i][0]
		qm[1] += q[i][1]
	}
	pm[0] /= f64(n)
	pm[1] /= f64(n)
	qm[0] /= f64(n)
	qm[1] /= f64(n)
	mut h00 := 0.0
	mut h01 := 0.0
	mut h10 := 0.0
	mut h11 := 0.0
	for i in 0 .. n {
		pc0 := p[i][0] - pm[0]
		pc1 := p[i][1] - pm[1]
		qc0 := q[i][0] - qm[0]
		qc1 := q[i][1] - qm[1]
		h00 += pc0 * qc0
		h01 += pc0 * qc1
		h10 += pc1 * qc0
		h11 += pc1 * qc1
	}
	theta := math.atan2(h01 - h10, h00 + h11)
	mut r := [][]f64{len: 2, init: []f64{len: 2}}
	r[0][0] = math.cos(theta)
	r[0][1] = -math.sin(theta)
	r[1][0] = math.sin(theta)
	r[1][1] = math.cos(theta)
	t0 := qm[0] - (r[0][0] * pm[0] + r[0][1] * pm[1])
	t1 := qm[1] - (r[1][0] * pm[0] + r[1][1] * pm[1])
	return r, [t0, t1]
}

// f32s converts []f64 to []f32 (test/data helper).
pub fn f32s(a []f64) []f32 {
	mut out := []f32{len: a.len}
	for i, v in a {
		out[i] = f32(v)
	}
	return out
}

// fmin2 returns the smaller of two f64 values.
pub fn fmin2(a f64, b f64) f64 {
	return if a < b { a } else { b }
}

// fmax2 returns the larger of two f64 values.
pub fn fmax2(a f64, b f64) f64 {
	return if a > b { a } else { b }
}

// min_i returns the smaller of two ints.
pub fn min_i(a int, b int) int {
	return if a < b { a } else { b }
}

// max_i returns the larger of two ints.
pub fn max_i(a int, b int) int {
	return if a > b { a } else { b }
}

// clamp01 clamps x into [0, 1].
pub fn clamp01(x f64) f64 {
	return fmax2(0.0, fmin2(x, 1.0))
}

// logsumexp returns log(Σ exp(x)), computed stably. Empty input → -inf.
pub fn logsumexp(x []f64) f64 {
	if x.len == 0 {
		return math.inf(-1)
	}
	mut m := x[0]
	for v in x {
		m = fmax2(m, v)
	}
	mut s := 0.0
	for v in x {
		s += math.exp(v - m)
	}
	return m + math.log(s)
}
