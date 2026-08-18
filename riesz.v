module conger

// riesz.v — Riesz (monogenic) wavelet feature frontend
// (V port of src/riesz.py; the visualise helpers are omitted).

import math

import mlx

// rz_fftfreq returns numpy-style sample frequencies (Nyquist = 0.5).
fn rz_fftfreq(n int) mlx.Array {
	k := mlx.arange(0.0, f64(n), 1.0, .float32)
	half := (n + 1) / 2
	k2 := mlx.where(k.less(mlx.f32_scalar(f32(half))), k, k.subtract(mlx.f32_scalar(f32(n))))
	return k2.divide(mlx.f32_scalar(f32(n)))
}

// rz_freqgrid returns (xgrid, ygrid) normalised frequency grids of shape (h,w).
fn rz_freqgrid(h int, w int) (mlx.Array, mlx.Array) {
	x := rz_fftfreq(w).expand_dims(0).broadcast_to([h, w])
	y := rz_fftfreq(h).expand_dims(1).broadcast_to([h, w])
	return x, y
}

// rz_crop_pad crops `p` px off each edge of a 2-D array.
fn rz_crop_pad(x mlx.Array, p int) mlx.Array {
	return x.take_axis(mlx.arange(f64(p), f64(x.dim(0) - p), 1.0, .int32), 0).take_axis(mlx.arange(f64(p),
		f64(x.dim(1) - p), 1.0, .int32), 1)
}

// rz_box_mean is a separable box mean (edge pad, k odd) via cumsum differences.
fn rz_box_mean(m mlx.Array, k int) mlx.Array {
	p := k / 2
	m2 := pad_edge(m, p)
	hp := m2.dim(0)
	wp := m2.dim(1)
	cs0 := m2.cumsum(0, false, true)
	c0 := mlx.concatenate([mlx.zeros([1, wp], .float32), cs0], 0)
	m3 := c0.take_axis(mlx.arange(f64(k), f64(hp + 1), 1.0, .int32), 0).subtract(c0.take_axis(mlx.arange(f64(0),
		f64(hp + 1 - k), 1.0, .int32), 0)).divide(mlx.f32_scalar(f32(k)))
	cs1 := m3.cumsum(1, false, true)
	c1 := mlx.concatenate([mlx.zeros([m3.dim(0), 1], .float32), cs1], 1)
	return c1.take_axis(mlx.arange(f64(k), f64(c1.dim(1)), 1.0, .int32), 1).subtract(c1.take_axis(mlx.arange(f64(0),
		f64(c1.dim(1) - k), 1.0, .int32), 1)).divide(mlx.f32_scalar(f32(k)))
}

struct RieszWavelet {
	lam_min    f64 = 3.0
	bandwidth  f64 = 1.0
	height     int
	width      int
	scale_size int
	pad        int
	xgrid      mlx.Array
	ygrid      mlx.Array
	radius     mlx.Array
	safe_r     mlx.Array
	m1         mlx.Array
	m2         mlx.Array
	dc_kernel  mlx.Array
	lams       []f64
	kernels    []mlx.Array
	n_freq     int
	k2         mlx.Array
mut:
	img    mlx.Array
	dc     mlx.Array
	scales []RieszScale
}

// new_riesz_wavelet builds the (shape-only) filter bank and runs update(img).
fn new_riesz_wavelet(img mlx.Array, lam_min f64, scale_size int, bandwidth f64) RieszWavelet {
	h := img.dim(0)
	w := img.dim(1)
	lam_max := fmin2(f64(h), f64(w)) / 2.0
	mut ssize := scale_size
	if ssize <= 0 {
		s := int(math.round(math.log2(lam_max / lam_min))) + 1
		ssize = if s > 4 { s } else { 4 }
	}
	mut lams := []f64{}
	if ssize == 1 {
		lams << lam_max
	} else {
		for i in 0 .. ssize {
			lams << lam_max * math.pow(2.0, -f64(i) * math.log2(lam_max / lam_min) / f64(ssize - 1))
		}
	}
	pad := int(lam_max)
	h2 := h + 2 * pad
	w2 := w + 2 * pad
	xgrid, ygrid := rz_freqgrid(h2, w2)
	sigma_f := 0.5 / lam_max
	radius := xgrid.multiply(xgrid).add(ygrid.multiply(ygrid))
	safe_r := radius.sqrt().maximum(mlx.f32_scalar(1e-12))
	dc_kernel := radius.divide(mlx.f32_scalar(f32(2.0 * sigma_f * sigma_f))).negative().exp()
	bw := bandwidth
	sigma_f_rel := (math.pow(2.0, bw) - 1.0) / ((math.pow(2.0, bw) + 1.0) * math.sqrt(2.0 * math.log(2.0)))
	mut kernels := []mlx.Array{}
	for lam in lams {
		f0 := 1.0 / lam
		sf := sigma_f_rel * f0
		kernel := radius.sqrt().subtract(mlx.f32_scalar(f32(f0))).square().divide(mlx.f32_scalar(f32(2.0 * sf * sf))).negative().exp()
		kernels << kernel
	}
	mut k2s := []mlx.Array{}
	for kk in kernels {
		k2s << kk.multiply(kk).sum()
	}
	k2 := mlx.stack(k2s, 0)
	n_freq := int(kernels[0].size())
	zero := mlx.zeros([h2, w2], .float32)
	m1 := complex_from(zero, xgrid.divide(safe_r).negative())
	m2 := complex_from(zero, ygrid.divide(safe_r).negative())
	mut rw := RieszWavelet{
		lam_min: lam_min
		bandwidth: bandwidth
		height: h
		width: w
		scale_size: ssize
		pad: pad
		xgrid: xgrid
		ygrid: ygrid
		radius: radius
		safe_r: safe_r
		m1: m1
		m2: m2
		dc_kernel: dc_kernel
		lams: lams
		kernels: kernels
		n_freq: n_freq
		k2: k2
	}
	rw.rz_update(img)
	return rw
}

// rz_update recomputes the image-dependent part (FFT, DC strip, per-scale responses).
fn (mut rw RieszWavelet) rz_update(img mlx.Array) {
	fft_arr := if rw.pad != 0 { fft2(pad_edge(img, rw.pad)) } else { fft2(img) }
	rw.dc = fft_arr.multiply(rw.dc_kernel)
	stripped := fft_arr.subtract(rw.dc)
	rw.scales = []RieszScale{}
	for kernel in rw.kernels {
		spec := stripped.multiply(kernel)
		mut b0 := ifft2(spec).real()
		mut b1 := ifft2(spec.multiply(rw.m1)).real()
		mut b2 := ifft2(spec.multiply(rw.m2)).real()
		if rw.pad > 0 {
			b0 = rz_crop_pad(b0, rw.pad)
			b1 = rz_crop_pad(b1, rw.pad)
			b2 = rz_crop_pad(b2, rw.pad)
		}
		rw.scales << new_riesz_scale(b0, b1, b2)
	}
	rw.img = img
}

// rz_median_all returns the median of all elements (mlx-c 0.6.0's mlx_median
// with no axes is broken — it trips a flatten axis assertion — so compute it
// via a full sort + middle take).
fn rz_median_all(x mlx.Array) f32 {
	n := int(x.size())
	sorted := x.reshape([n]).sort()
	if n % 2 == 1 {
		return sorted.take(sel1(n / 2)).item_f32()
	}
	lo := sorted.take(sel1(n / 2 - 1)).item_f32()
	hi := sorted.take(sel1(n / 2)).item_f32()
	return (lo + hi) / 2.0
}

// rz_features extracts the cross-scale spectral statistics.
fn (rw RieszWavelet) rz_features(gain_control bool, retinex_k int) FeatureMaps {
	mut e_arrs := []mlx.Array{}
	for s in rw.scales {
		e_arrs << s.energy
	}
	mut e := mlx.stack(e_arrs, -1) // (H,W,S)

	if gain_control {
		b0f := rw.scales[rw.scales.len - 1].b0
		b0f_med := rz_median_all(b0f)
		mad := rz_median_all(b0f.subtract(mlx.f32_scalar(b0f_med)).abs())
		k2_last := rw.k2.take(sel1(rw.k2.dim(0) - 1)).item_f32()
		sig2 := mlx.f32_scalar(mad).multiply(mlx.f32_scalar(1.4826)).square().multiply(mlx.f32_scalar(f32(rw.n_freq))).divide(mlx.f32_scalar(k2_last))
		floor := sig2.multiply(mlx.f32_scalar(3.0)).multiply(rw.k2).divide(mlx.f32_scalar(f32(rw.n_freq)))
		denom := e.add(floor)
		e = mlx.where(denom.greater(mlx.f32_scalar(0.0)), e.multiply(e).divide(denom), mlx.f32_scalar(0.0))
	}

	total := e.sum_axis(-1, false)
	safe_total := total.maximum(mlx.f32_scalar(1e-12))
	p := e.divide(safe_total.expand_dims(-1))
	log_e := e.maximum(mlx.f32_scalar(1e-12)).log()
	mut log_mag := safe_total.log()
	if gain_control {
		mut k := retinex_k
		if k == 0 {
			mut kk := int(rw.lam_max() / 4.0) | 1
			if 7 > kk {
				kk = 7
			}
			k = kk
		}
		log_mag = log_mag.subtract(rz_box_mean(log_mag, k))
	}

	lam_max := rw.lam_max()
	mut x_vals := []f64{}
	for lam in rw.lams {
		x_vals << math.log2(lam_max / lam)
	}
	x := arr32(x_vals, [x_vals.len])
	x_mean := x.mean()
	xc := x.subtract(x_mean)
	var_x := xc.multiply(xc).sum().item_f32()
	n_scales := rw.lams.len
	y := log_e
	y_mean := y.mean_axis(-1, true)
	slope := xc.multiply(y.subtract(y_mean)).sum_axis(-1, false).divide(mlx.f32_scalar(var_x))
	intercept := y_mean.squeeze_axis(-1).subtract(slope.multiply(x_mean))
	fit := intercept.expand_dims(-1).add(slope.expand_dims(-1).multiply(x))
	residual := y.subtract(fit).square().mean_axis(-1, false).sqrt()
	div := if n_scales - 1 > 1 { n_scales - 1 } else { 1 }
	bump := e.argmax_axis(-1, false).astype(.float32).divide(mlx.f32_scalar(f32(div)))

	mu := p.multiply(x).sum_axis(-1, false)
	d := x.subtract(mu.expand_dims(-1))
	vari := p.multiply(d.multiply(d)).sum_axis(-1, false)
	sd := vari.maximum(mlx.f32_scalar(1e-12)).sqrt()
	centroid := mu
	spread := sd
	skew := p.multiply(d.multiply(d).multiply(d)).sum_axis(-1, false).divide(sd.multiply(sd).multiply(sd))
	kurt := p.multiply(d.multiply(d).multiply(d).multiply(d)).sum_axis(-1, false).divide(sd.multiply(sd).multiply(sd).multiply(sd))

	mut ori_arrs := []mlx.Array{}
	for s in rw.scales {
		ori_arrs << s.ori
	}
	ori := mlx.stack(ori_arrs, -1)
	two_ori := ori.multiply(mlx.f32_scalar(2.0))
	m_re := e.multiply(two_ori.cos()).sum_axis(-1, false)
	m_im := e.multiply(two_ori.sin()).sum_axis(-1, false)
	mut ori_r := m_re.multiply(m_re).add(m_im.multiply(m_im)).sqrt().divide(safe_total)
	mean_ori := m_im.arctan2(m_re).multiply(mlx.f32_scalar(0.5))

	mut a_arrs := []mlx.Array{}
	mut ph_arrs := []mlx.Array{}
	for s in rw.scales {
		a_arrs << s.amp
		ph_arrs << s.phase
	}
	a := mlx.stack(a_arrs, -1)
	ph := mlx.stack(ph_arrs, -1)
	p_re := a.multiply(ph.cos()).sum_axis(-1, false)
	p_im := a.multiply(ph.sin()).sum_axis(-1, false)
	a_sum := a.sum_axis(-1, false).maximum(mlx.f32_scalar(1e-12))
	mut phase_coh := p_re.multiply(p_re).add(p_im.multiply(p_im)).sqrt().divide(a_sum)

	if gain_control {
		r_fl, p_fl := rz_coherence_floor(rw.height, rw.width, rw.lam_min, rw.scale_size, rw.bandwidth)
		ori_r = ori_r.subtract(mlx.f32_scalar(f32(r_fl))).maximum(mlx.f32_scalar(0.0)).divide(mlx.f32_scalar(f32(fmax2(1.0 - r_fl,
			1e-3))))
		phase_coh = phase_coh.subtract(mlx.f32_scalar(f32(p_fl))).maximum(mlx.f32_scalar(0.0)).divide(mlx.f32_scalar(f32(fmax2(1.0 - p_fl,
			1e-3))))
	}

	return FeatureMaps{
		log_mag: log_mag
		slope: slope
		residual: residual
		bump: bump
		centroid: centroid
		spread: spread
		skew: skew
		kurt: kurt
		ori_r: ori_r
		mean_ori: mean_ori
		phase_coh: phase_coh
		log_e: log_e
	}
}

// rz_coherence_floor calibrates the noise floor of the coherence stats.
fn rz_coherence_floor(h int, w int, lam_min f64, scale_size int, bandwidth f64) (f64, f64) {
	noise := mlx.random_normal([h, w], .float32, 0.0, 1.0, mlx.random_key(0))
	mut probe := new_riesz_wavelet(noise, lam_min, scale_size, bandwidth)
	f := probe.rz_features(false, 0)
	return f64(f.ori_r.mean().item_f32()), f64(f.phase_coh.mean().item_f32())
}

// rz_lam_max returns the coarsest wavelength supported by the image size.
fn (rw RieszWavelet) lam_max() f64 {
	return fmin2(f64(rw.height), f64(rw.width)) / 2.0
}
