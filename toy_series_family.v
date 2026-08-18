module conger

// toy_series_family.v — toy time-series mechanism family (linear / sine).
import math
import mlx

// toy_x returns the shared fixed grid X ∈ [-1, 1] (32 points).
pub fn toy_x() mlx.Array {
	return mlx.linspace(-1.0, 1.0, 32, .float32)
}

pub struct ToySeriesFamily {
	mechanism string
}

pub fn new_toy_series_family(mechanism string) ToySeriesFamily {
	if mechanism != 'linear' && mechanism != 'sine' {
		panic('unknown toy mechanism ${mechanism}')
	}
	return ToySeriesFamily{
		mechanism: mechanism
	}
}

fn (f ToySeriesFamily) n_params() int {
	return if f.mechanism == 'linear' { 2 } else { 3 }
}

// sample returns (n, n_params) uniform parameters.
pub fn (f ToySeriesFamily) sample(n int, seed u64) mlx.Array {
	ks := split_keys(seed, 3)
	k1 := ks[0]
	k2 := ks[1]
	k3 := ks[2]
	lo := mlx.f32_scalar(0.0)
	hi := mlx.f32_scalar(1.0)
	if f.mechanism == 'linear' {
		a :=
			mlx.random_uniform(lo, hi, [n], .float32, k1).multiply(mlx.f32_scalar(4.0)).add(mlx.f32_scalar(-2.0))
		b :=
			mlx.random_uniform(lo, hi, [n], .float32, k2).multiply(mlx.f32_scalar(2.0)).add(mlx.f32_scalar(-1.0))
		return mlx.stack([a, b], 1)
	}
	amp :=
		mlx.random_uniform(lo, hi, [n], .float32, k1).multiply(mlx.f32_scalar(1.5)).add(mlx.f32_scalar(0.5))
	freq :=
		mlx.random_uniform(lo, hi, [n], .float32, k2).multiply(mlx.f32_scalar(3.0)).add(mlx.f32_scalar(2.0))
	phase :=
		mlx.random_uniform(lo, hi, [n], .float32, k3).multiply(mlx.f32_scalar(2.0 * math.pi)).add(mlx.f32_scalar(-math.pi))
	return mlx.stack([amp, freq, phase], 1)
}

// simulate maps (n,P) params → (n,T) observation sequences.
pub fn (f ToySeriesFamily) simulate(params mlx.Array) mlx.Array {
	x := toy_x().expand_dims(0)
	p0 := params.take_axis(sel1(0), 1)
	if f.mechanism == 'linear' {
		p1 := params.take_axis(sel1(1), 1)
		return p0.multiply(x).add(p1)
	}
	p1 := params.take_axis(sel1(1), 1)
	p2 := params.take_axis(sel1(2), 1)
	return p0.multiply(p1.multiply(x).add(p2).sin())
}

// residual returns RMSE(observed, simulate(params)).
pub fn (f ToySeriesFamily) residual(observation mlx.Array, params []f64) f64 {
	p := arr32(params, [1, params.len])
	sim := f.simulate(p)
	pred := sim.take_axis(sel1(0), 0).squeeze_axis(0)
	return f64(observation.subtract(pred).square().mean().sqrt().item_f32())
}

// encode maps a 1D sequence to a 16-d summary feature vector, computed with MLX
// float32 ops, kept in float32 for numerical consistency.
pub fn (f ToySeriesFamily) encode(y mlx.Array) mlx.Array {
	yf := y.astype(.float32)
	n := y.dim(0)
	last := yf.take_axis(sel1(n - 1), 0).squeeze_axis(0)
	first := yf.take_axis(sel1(0), 0).squeeze_axis(0)
	d := yf.take_axis(mlx.arange(1.0, f64(n), 1.0, .int32), 0).subtract(yf.take_axis(mlx.arange(0.0,
		f64(n - 1), 1.0, .int32), 0))
	dd := yf.take_axis(mlx.arange(2.0, f64(n), 1.0, .int32), 0).subtract(yf.take_axis(mlx.arange(1.0,
		f64(n - 1), 1.0, .int32), 0).multiply(mlx.f32_scalar(2.0))).add(yf.take_axis(mlx.arange(0.0,
		f64(n - 2), 1.0, .int32), 0))
	x := toy_x()
	mut feats := []mlx.Array{}
	feats << yf.mean()
	feats << yf.std(0)
	feats << yf.min()
	feats << yf.max()
	feats << last.subtract(first)
	feats << d.abs().mean()
	feats << d.std(0)
	feats << dd.std(0)
	for freq in [2.0, 3.0, 4.0, 5.0] {
		feats << yf.multiply(x.multiply(mlx.f32_scalar(f32(freq))).sin()).mean()
		feats << yf.multiply(x.multiply(mlx.f32_scalar(f32(freq))).cos()).mean()
	}
	return mlx.stack(feats, 0).astype(.float32)
}
