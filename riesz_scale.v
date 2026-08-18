module conger

// riesz_scale.v — single-scale monogenic wavelet response (V port of
// src/riesz_scale.py).
import mlx

struct RieszScale {
	b0     mlx.Array // bandpass response (even)
	b1     mlx.Array // Riesz-x response (odd along x)
	b2     mlx.Array // Riesz-y response (odd along y)
	amp    mlx.Array
	phase  mlx.Array
	ori    mlx.Array
	energy mlx.Array
}

// new_riesz_scale derives energy/amp/phase/ori from b0/b1/b2.
fn new_riesz_scale(b0 mlx.Array, b1 mlx.Array, b2 mlx.Array) RieszScale {
	r2 := b1.multiply(b1).add(b2.multiply(b2))
	energy := b0.multiply(b0).add(r2)
	amp := energy.sqrt()
	phase := r2.sqrt().arctan2(b0)
	ori := b2.arctan2(b1)
	return RieszScale{
		b0:     b0
		b1:     b1
		b2:     b2
		amp:    amp
		phase:  phase
		ori:    ori
		energy: energy
	}
}
