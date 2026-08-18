module conger

// feature_extractor.v — rendered frame → full-resolution features
// (V port of src/feature_extractor.py; the Riesz frontend is added separately).

import math

import mlx

// frame_lum returns the Rec601 luminance (H,W) float32 in [0,1].
fn frame_lum(frame mlx.Array) mlx.Array {
	rgb := frame.take_axis(mlx.arange(0.0, 3.0, 1.0, .int32), -1).astype(.float32).divide(mlx.f32_scalar(255.0))
	r := rgb.take_axis(sel1(0), -1).squeeze_axis(-1)
	g := rgb.take_axis(sel1(1), -1).squeeze_axis(-1)
	b := rgb.take_axis(sel1(2), -1).squeeze_axis(-1)
	return r.multiply(mlx.f32_scalar(0.299)).add(g.multiply(mlx.f32_scalar(0.587))).add(b.multiply(mlx.f32_scalar(0.114)))
}

// frame_hs returns the (hue, saturation) maps, each [0,1).
fn frame_hs(frame mlx.Array) (mlx.Array, mlx.Array) {
	rgb := frame.take_axis(mlx.arange(0.0, 3.0, 1.0, .int32), -1).astype(.float32).divide(mlx.f32_scalar(255.0))
	r := rgb.take_axis(sel1(0), -1).squeeze_axis(-1)
	g := rgb.take_axis(sel1(1), -1).squeeze_axis(-1)
	b := rgb.take_axis(sel1(2), -1).squeeze_axis(-1)
	mxv := r.maximum(g).maximum(b)
	mn := r.minimum(g).minimum(b)
	d := mxv.subtract(mn)
	s := mlx.where(mxv.greater(mlx.f32_scalar(1e-6)), d.divide(mxv.maximum(mlx.f32_scalar(1e-6))),
		mlx.f32_scalar(0.0))
	max_r := r.equal(mxv)
	max_g := g.equal(mxv)
	mut h6 := mlx.where(max_r, g.subtract(b).divide(d.maximum(mlx.f32_scalar(1e-9))),
		mlx.f32_scalar(0.0))
	h6 = mlx.where(max_g, b.subtract(r).divide(d.maximum(mlx.f32_scalar(1e-9))).add(mlx.f32_scalar(2.0)),
		h6)
	not_rg := max_r.logical_not().logical_and(max_g.logical_not())
	h6 = mlx.where(not_rg, r.subtract(g).divide(d.maximum(mlx.f32_scalar(1e-9))).add(mlx.f32_scalar(4.0)),
		h6)
	h := mlx.where(d.less(mlx.f32_scalar(1e-6)), mlx.f32_scalar(0.0), h6.divide(mlx.f32_scalar(6.0)))
	return h, s
}

// frame_chroma returns the complex-hue (real, imag) maps S·e^{i2πH}.
fn frame_chroma(frame mlx.Array) (mlx.Array, mlx.Array) {
	h, s := frame_hs(frame)
	ang := h.multiply(mlx.f32_scalar(f32(2.0 * math.pi)))
	return s.multiply(ang.cos()), s.multiply(ang.sin())
}

// FeatureExtractor holds the config for the Riesz-frontend feature assembly.
struct FeatureExtractor {
	cfg InverseConfig
}

// new_feature_extractor builds the extractor for a config.
fn new_feature_extractor(cfg InverseConfig) FeatureExtractor {
	return FeatureExtractor{
		cfg: cfg
	}
}

// feat_map_field selects one FeatureMaps channel by name.
fn feat_map_field(f FeatureMaps, ch string) mlx.Array {
	return match ch {
		'log_mag' { f.log_mag }
		'phase_coh' { f.phase_coh }
		'ori_R' { f.ori_r }
		'slope' { f.slope }
		'residual' { f.residual }
		'bump' { f.bump }
		'centroid' { f.centroid }
		'spread' { f.spread }
		'skew' { f.skew }
		'kurt' { f.kurt }
		'mean_ori' { f.mean_ori }
		else { f.log_mag }
	}
}

// of_frame returns the full-resolution feature vector (n_feat,) and the
// (threaded) RieszWavelet workspace.
fn (e FeatureExtractor) of_frame(frame mlx.Array, rw_opt ?RieszWavelet) (mlx.Array, RieszWavelet) {
	lum := frame_lum(frame)
	chr_re, chr_im := frame_chroma(frame)
	mut rw := rw_opt or { new_riesz_wavelet(lum, 3.0, 0, 1.0) }
	mut parts := []mlx.Array{}
	for spec in feat_spec_list() {
		img := match spec.src {
			'lum' { lum }
			'chr_re' { chr_re }
			else { chr_im }
		}
		if spec.ch == 'raw' {
			parts << img.reshape([int(img.size())])
			continue
		}
		rw.rz_update(img)
		gc := spec.src == 'lum'
		f := rw.rz_features(gc, 0)
		m := feat_map_field(f, spec.ch)
		parts << m.reshape([int(m.size())])
	}
	return mlx.concatenate(parts, 0), rw
}
