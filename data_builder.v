module conger

// data_builder.v — data construction (V port of src/data_builder.py).

import os

import mlx

// db_targets returns the continuous targets (4-d single / 8-d layered).
fn db_targets(p mlx.Array) mlx.Array {
	if p.dim(1) == 14 {
		return p.take_axis(mlx.array_i32([i32(1), i32(2), i32(3), i32(4), i32(7),
			i32(8), i32(9), i32(10)], [8]), 1)
	}
	return p.take_axis(mlx.arange(1.0, 5.0, 1.0, .int32), 1)
}

// db_scene_classes returns the discrete scene factors.
fn db_scene_classes(p mlx.Array) mlx.Array {
	if p.dim(1) == 14 {
		return p.take_axis(mlx.array_i32([i32(0), i32(6), i32(5), i32(11), i32(12),
			i32(13)], [6]), 1).astype(.int32)
	}
	return p.take_axis(mlx.array_i32([i32(0), i32(5), i32(6), i32(7)], [4]),
		1).astype(.int32)
}

// DataBuilder holds the config/codebook/extractor wiring (cache tag + data
// assembly).
struct DataBuilder {
	cfg       InverseConfig
	codebook  SceneFamily
	extractor FeatureExtractor
}

// cache_tag returns a deterministic cache-filename stem that changes with the
// family, geometry variant and child-template variant.
fn (d DataBuilder) cache_tag() string {
	mut tag := 'mix_${img_w}x${img_h}_fam${d.codebook.geometry_family()}'
	if d.codebook.template_variant() != '' {
		tag += '_tv${d.codebook.template_variant()}'
	}
	return tag
}

// db_feats_of renders each param row and returns (features (N,V), stereo stats).
fn (d DataBuilder) db_feats_of(params mlx.Array) (mlx.Array, mlx.Array) {
	n := params.dim(0)
	w := params.dim(1)
	pf := params.data_f32()
	mut renderer, cam_l, cam_r := make_renderer(stereo_base)
	app := InverseApp{
		cfg: d.cfg
		codebook: d.codebook
		data: d
		extractor: d.extractor
	}
	mut out := []mlx.Array{}
	mut stats_flat := []f64{}
	mut rw := ?RieszWavelet(none)
	for i in 0 .. n {
		mut prm := []f64{len: w}
		for j in 0 .. w {
			prm[j] = f64(pf[i * w + j])
		}
		scene := d.codebook.to_scene(prm)
		fl := renderer.render(scene, cam_l)
		fr := renderer.render(scene, cam_r)
		vec, st, nrw := sr_frame_features(app, fl, fr, rw)
		rw = nrw
		out << vec.take_axis(sel1(0), 0).squeeze_axis(0)
		sd := st.data_f32()
		for v in sd {
			stats_flat << f64(v)
		}
	}
	stats_w := if d.codebook.geometry_family() == 'single' { 3 } else { 8 }
	return mlx.stack(out, 0), arr32(stats_flat, [n, stats_w])
}

// DataSplit is the assembled train/interp/extrap data.
struct DataSplit {
	f_tr mlx.Array
	p_tr mlx.Array
	f_ti mlx.Array
	p_ti mlx.Array
	f_te mlx.Array
	p_te mlx.Array
	s_tr mlx.Array
	s_ti mlx.Array
	s_te mlx.Array
}

// db_block_feats renders one replicate block (with per-block safetensors cache).
fn (d DataBuilder) db_block_feats(split string, r int) (mlx.Array, mlx.Array, mlx.Array) {
	cache_dir := 'artifacts'
	os.mkdir_all(cache_dir, os.MkdirParams{}) or {}
	path := os.join_path(cache_dir, '${d.cache_tag()}_${split}${r}.safetensors')
	if d.cfg.use_cache && os.exists(path) {
		tens, _ := mlx.load_safetensors(path)
		return tens.get('P'), tens.get('F'), tens.get('S')
	}
	seed := match split {
		'tr' { u64(42) }
		'ti' { u64(99) }
		else { u64(7) }
	}
	p := d.codebook.sample(1, seed + u64(r), split == 'te')
	f, s := d.db_feats_of(p)
	mut tens := mlx.new_map_string_to_array()
	tens.insert('P', p)
	tens.insert('F', f)
	tens.insert('S', s)
	mlx.save_safetensors(path, tens, mlx.new_map_string_to_string())
	return p, f, s
}

// db_build assembles the train/interp/extrap features, params and stats.
fn (d DataBuilder) db_build(n_rep int) DataSplit {
	mut ptr := []mlx.Array{}
	mut ftr := []mlx.Array{}
	mut str := []mlx.Array{}
	for r in 0 .. n_rep {
		p, f, s := d.db_block_feats('tr', r)
		ptr << p
		ftr << f
		str << s
	}
	n_test := 2
	mut pti := []mlx.Array{}
	mut fti := []mlx.Array{}
	mut sti := []mlx.Array{}
	mut pte := []mlx.Array{}
	mut fte := []mlx.Array{}
	mut ste := []mlx.Array{}
	for r in 0 .. n_test {
		p, f, s := d.db_block_feats('ti', r)
		pti << p
		fti << f
		sti << s
		p2, f2, s2 := d.db_block_feats('te', r)
		pte << p2
		fte << f2
		ste << s2
	}
	return DataSplit{
		f_tr: mlx.concatenate(ftr, 0)
		p_tr: mlx.concatenate(ptr, 0)
		f_ti: mlx.concatenate(fti, 0)
		p_ti: mlx.concatenate(pti, 0)
		f_te: mlx.concatenate(fte, 0)
		p_te: mlx.concatenate(pte, 0)
		s_tr: mlx.concatenate(str, 0)
		s_ti: mlx.concatenate(sti, 0)
		s_te: mlx.concatenate(ste, 0)
	}
}
