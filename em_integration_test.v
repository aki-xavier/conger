module conger

// em_integration_test.v — V port of tests/test_em_integration.py
// (geometry <-> illumination ECM contract tests; ECM disabled by default).

fn test_em_refine_defaults_off() {
	cfg := InverseConfig{}
	assert cfg.em_refine == false
	assert cfg.em_max_iters == 2
	assert cfg.em_appearance_topk == 3

	cfg_on := InverseConfig{
		em_refine:          true
		em_max_iters:       4
		em_appearance_topk: 2
	}
	assert cfg_on.em_refine == true
	assert cfg_on.em_max_iters == 4
	assert cfg_on.em_appearance_topk == 2
}

fn test_hypothesis_carries_em_trajectory() {
	// StructuredHypothesis should record the per-round ECM log-likelihood trajectory.
	h := StructuredHypothesis{
		structure_id:  'single'
		em_trajectory: [1.0, 2.0]
	}
	assert h.em_trajectory.len == 2
	assert h.em_trajectory[0] == 1.0
	assert h.em_trajectory[1] == 2.0

	// Default carries no trajectory (empty slice == None in the Python source).
	d := new_hypothesis()
	assert d.em_trajectory.len == 0
}
