module conger

// feature_maps.v — RieszWavelet.features() output record (V port of
// src/feature_maps.py).
import mlx

struct FeatureMaps {
	log_mag   mlx.Array // local contrast (log Σe_s minus box mean)
	slope     mlx.Array // power-law log e_s vs octave slope
	residual  mlx.Array // power-law fit RMS residual
	bump      mlx.Array // argmax_s e_s, normalised to [0,1]
	centroid  mlx.Array // energy-distribution first moment (octave)
	spread    mlx.Array // second moment (std)
	skew      mlx.Array // third moment
	kurt      mlx.Array // fourth moment
	ori_r     mlx.Array // cross-scale orientation coherence (2θ resultant)
	mean_ori  mlx.Array // cross-scale mean normal (−π/2, π/2]
	phase_coh mlx.Array // cross-scale phase coherence
	log_e     mlx.Array // log per-scale energy (H,W,S)
}
