module conger

// structure_benchmark_test.v — V port of tests/test_structure_benchmark.py.
import math

fn sb_test_result(true_fam string, pred string, p f64) StructureCaseResult {
	names := ['single', 'layered', 'composite']
	mut posterior := map[string]f64{}
	mut residuals := map[string]f64{}
	mut scores := map[string]f64{}
	for k in names {
		posterior[k] = if k == pred { p } else { (1.0 - p) / 2.0 }
		residuals[k] = 1.0
		scores[k] = 1.0
	}
	return StructureCaseResult{
		true_family:         true_fam
		predicted:           pred
		posterior:           posterior
		residuals:           residuals
		scores:              scores
		needs_new_structure: false
	}
}

fn test_structure_benchmark_summary() {
	results := [
		sb_test_result('single', 'single', 0.8),
		sb_test_result('layered', 'composite', 0.6),
		sb_test_result('composite', 'composite', 0.7),
	]
	out := sb_summarize(results)
	assert out.n == 3
	assert math.abs(out.accuracy - 2.0 / 3.0) < 1e-12
	assert out.confusion['layered']['composite'] == 1
	assert out.posterior_mean.len == 3
	assert 'single' in out.posterior_mean
	assert 'layered' in out.posterior_mean
	assert 'composite' in out.posterior_mean
}
