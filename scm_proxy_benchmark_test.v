module conger

// scm_proxy_benchmark_test.v — rendering-backed appearance-mechanism calibration
// (V port of the core of src/scm_proxy_benchmark.py).
import mlx

fn test_rendered_appearance_mechanism_is_modular() {
	cb := new_codebook(InverseConfig{ scene_family: 'single' })
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	// fixed geometry (sphere), sweep the 54 appearance factors
	base := [0.0, 72.0, 72.0, 0.45, 3.2]
	mut rgb_flat := []f32{}
	for h in 0 .. n_hue {
		for lc in 0 .. light_colors_len {
			for ld in 0 .. light_dirs_len {
				prm := [base[0], base[1], base[2], base[3], base[4], f64(h), f64(lc), f64(ld)]
				scene := cb.to_scene(prm)
				fl := renderer.render(scene, cam_l)
				w := foreground_weights(fl)
				mean := foreground_mean_rgb(fl, w).data_f32()
				rgb_flat << mean[0]
				rgb_flat << mean[1]
				rgb_flat << mean[2]
			}
		}
	}
	rgb := mlx.array_f32(rgb_flat, [n_hue, light_colors_len, light_dirs_len, 3])
	mut mechanism := AppearanceMechanism{}
	mechanism.fit(rgb)
	err := mechanism.reconstruction_error(rgb)
	inv := mechanism.albedo_invariance(rgb)
	// the multiplicative albedo×lighting decomposition should be near-exact on
	// real MeshStandardMaterial renders (invariance ≈ 1, low reconstruction error)
	assert err < 0.05
	assert inv > 0.95
}
