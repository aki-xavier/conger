module conger

// uci_wine_test.v — quantitative regression for the MixtureSPN classification
// path on a second UCI dataset (wine: 178 samples × 13 features, 3 classes),
// guarding against silent accuracy regressions in the core SPN. Mirrors the
// iris example's wiring: cat contract [3], stratum = class index (per-class
// tied variance), deterministic stratified split (every 5th per class →
// test). Data: testdata/wine.data (UCI ML repository).
import os
import mlx

fn load_wine() ([]f64, []int) {
	lines := os.read_lines('testdata/wine.data') or { panic(err) }
	mut feats := []f64{cap: 178 * 13}
	mut labels := []int{cap: 178}
	for line in lines {
		if line.trim_space() == '' {
			continue
		}
		parts := line.split(',')
		if parts.len != 14 {
			panic('wine.data: expected 14 columns, got ${parts.len}')
		}
		labels << parts[0].int() - 1 // classes 1/2/3 → 0/1/2
		for p in parts[1..] {
			feats << p.f64()
		}
	}
	return feats, labels
}

fn test_uci_wine_classification() {
	feats, labels := load_wine()
	assert labels.len == 178
	// deterministic stratified split: every 5th sample within each class → test
	mut train_f := []f64{cap: 140 * 13}
	mut test_f := []f64{cap: 38 * 13}
	mut train_y := []int{cap: 140}
	mut test_y := []int{cap: 38}
	mut seen := map[int]int{}
	for i, y in labels {
		k := seen[y]
		seen[y] = k + 1
		row := feats[i * 13..(i + 1) * 13]
		if k % 5 == 0 {
			test_f << row
			test_y << y
		} else {
			train_f << row
			train_y << y
		}
	}
	f_train := mlx.arr32(train_f, [train_y.len, 13])
	t_train := mlx.zeros([train_y.len, 1], .float32)
	mut sc_i32 := []i32{cap: train_y.len}
	for y in train_y {
		sc_i32 << i32(y)
	}
	scene := mlx.array_i32(sc_i32, [train_y.len, 1])
	stratum := mlx.col(scene, 0)
	m := fit_mixture_spn(f_train, t_train, stratum, 1e-3, scene, [3], 0)

	f_test := mlx.arr32(test_f, [test_y.len, 13])
	_, cat_p, _ := m.predict(f_test)
	cp := cat_p.data_f32()
	mut correct := 0
	for i, y in test_y {
		mut best := 0
		for c in 1 .. 3 {
			if cp[i * 3 + c] > cp[i * 3 + best] {
				best = c
			}
		}
		if best == y {
			correct++
		}
	}
	acc := f64(correct) / f64(test_y.len)
	println('wine: ${correct}/${test_y.len} = ${acc * 100.0:.1f}%')
	// wine is nearly linearly separable; anything below 85% signals a
	// regression in fit/predict, not data difficulty
	assert acc >= 0.85
}
