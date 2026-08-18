module conger

// structure_gate_test.v — V port of tests/test_structure_gate.py.
import mlx

fn sg_estimate(params []f64, cb Codebook) StructuredHypothesis {
	return StructuredHypothesis{
		scene:         cb.to_scene(params)
		params:        params
		spn_posterior: mlx.zeros([15], .float32)
	}
}

fn test_structure_gate_posterior_and_birth() {
	cb := new_codebook(InverseConfig{})
	good_prm := [0.0, 72.0, 72.0, 0.45, 3.2, 2.0, 0.0, 1.0]
	bad_a := [1.0, 30.0, 40.0, 0.55, 3.8, 4.0, 1.0, 2.0]
	bad_b := [2.0, 110.0, 100.0, 0.35, 2.8, 5.0, 2.0, 0.0]
	mut renderer, cam_l, cam_r := sr_rig()
	gt := cb.to_scene(good_prm)
	fl := renderer.render(gt, cam_l)
	fr := renderer.render(gt, cam_r)
	gate := new_structure_gate()
	good := sg_estimate(good_prm, cb)
	bad := sg_estimate(bad_a, cb)
	out := gate.decide({
		'good': good
		'bad':  bad
	}, fl, fr)
	assert out.estimate.structure_id == 'good'
	assert out.posterior['good'] > 0.99
	assert !out.needs_new_structure

	born := gate.decide({
		'bad_a': sg_estimate(bad_a, cb)
		'bad_b': sg_estimate(bad_b, cb)
	}, fl, fr)
	assert born.needs_new_structure
	mut maxp := -1e18
	for _, v in born.posterior {
		if v > maxp {
			maxp = v
		}
	}
	assert maxp < 0.8
}
