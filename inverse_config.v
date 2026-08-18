module conger

// inverse_config.v — inverse-rendering run configuration (V port of
// src/inverse_config.py; only the fields used by the ported pipeline).

struct InverseConfig {
	use_cache          bool = true
	replicates         int  = 8
	sigma_rel_floor    f64  = 1e-2
	refine_appearance  bool = true
	refine_composite   bool = false
	kind_topk          int  = 3
	basis_dim          int  = 48
	scene_family       string
	n_textures         int  = 0
	em_refine          bool = false
	em_max_iters       int  = 2
	em_appearance_topk int  = 3
	em_freeze_sz       bool = true
	bg_color           int  = 0x141414
}

fn (c InverseConfig) family() string {
	return if c.scene_family != '' { c.scene_family } else { 'single' }
}

fn (c InverseConfig) textured() bool {
	return c.n_textures > 0
}

// FeatSpec is one (source, channel) entry of the Riesz feature spec.
struct FeatSpec {
	src string
	ch  string
}

// feat_spec_list returns the fixed Riesz feature set (11 entries).
fn feat_spec_list() []FeatSpec {
	return [
		FeatSpec{'lum', 'log_mag'},
		FeatSpec{'lum', 'phase_coh'},
		FeatSpec{'lum', 'ori_R'},
		FeatSpec{'chr_re', 'log_mag'},
		FeatSpec{'chr_re', 'phase_coh'},
		FeatSpec{'chr_re', 'ori_R'},
		FeatSpec{'chr_im', 'log_mag'},
		FeatSpec{'chr_im', 'phase_coh'},
		FeatSpec{'chr_im', 'ori_R'},
		FeatSpec{'chr_re', 'raw'},
		FeatSpec{'chr_im', 'raw'},
	]
}

// n_feat returns the full-resolution feature width (Riesz + stereo stats).
fn (c InverseConfig) n_feat() int {
	n := feat_spec_list().len * img_h * img_w
	return n + (if c.family() == 'single' { 2 } else { 8 })
}
