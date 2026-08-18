module conger

// retinex_test.v — RetinexModel (albedo↔lighting) black-box test.

import math

fn test_retinex_decomposes_albedo_and_light() {
	n := 96
	mut segments := []int{len: n}
	for i in 0 .. n {
		if i > n / 2 {
			segments[i] = 1
		}
	}
	model := new_retinex_model(segments, 0.02)
	true_log_a := [0.3, -0.4]
	true_l := [0.2, 0.5]
	obs := model.render(true_log_a, true_l, 0)

	mut loop := EMLoop[RetinexModel, []f64, []f64]{
		model: model
		max_iters: 20
		tol: 1e-10
	}
	result := loop.run(obs, [0.0, 0.0])

	mut log_i := []f64{len: n}
	for i in 0 .. n {
		log_i[i] = math.log(obs[i] + 1e-12)
	}
	log_a := result.responsibilities
	l0 := result.params[0]
	l1 := result.params[1]
	mut s := 0.0
	for i in 0 .. n {
		recon := log_a[i] + l0 + l1 * model.x[i]
		d := recon - log_i[i]
		s += d * d
	}
	assert math.sqrt(s / f64(n)) < 0.05

	// albedo contrast (segment mean difference) matches ground truth
	mut mean0 := 0.0
	mut mean1 := 0.0
	mut c0 := 0
	mut c1 := 0
	for i in 0 .. n {
		if segments[i] == 1 {
			mean1 += log_a[i]
			c1++
		} else {
			mean0 += log_a[i]
			c0++
		}
	}
	contrast := mean1 / f64(c1) - mean0 / f64(c0)
	assert math.abs(contrast - (true_log_a[1] - true_log_a[0])) < 0.05
}
