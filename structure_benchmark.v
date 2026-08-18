module conger

// structure_benchmark.v — cross-family structure-gate benchmark aggregation
// (V port of src/structure_benchmark.py; the render/CLI loop is added with the
// inverse-app integration, the summariser is standalone).
import math

// StructureCaseResult records one ground-truth sample's gating outcome.
struct StructureCaseResult {
	true_family         string
	predicted           string
	posterior           map[string]f64
	residuals           map[string]f64
	scores              map[string]f64
	needs_new_structure bool
}

// StructureBenchmarkSummary is the aggregated benchmark output.
struct StructureBenchmarkSummary {
	n              int
	accuracy       f64
	confusion      map[string]map[string]int
	posterior_mean map[string]f64
	ece            f64
}

// sb_ece returns the winner-confidence expected calibration error.
fn sb_ece(results []StructureCaseResult) f64 {
	if results.len == 0 {
		return 0.0
	}
	bins := 10
	mut acc := []f64{len: bins}
	mut conf := []f64{len: bins}
	mut cnt := []int{len: bins}
	for r in results {
		c := r.posterior[r.predicted]
		mut idx := int(c * f64(bins))
		if idx > bins - 1 {
			idx = bins - 1
		}
		if r.predicted == r.true_family {
			acc[idx] += 1.0
		}
		conf[idx] += c
		cnt[idx]++
	}
	total := f64(results.len)
	mut ece := 0.0
	for i in 0 .. bins {
		if cnt[i] == 0 {
			continue
		}
		ece += (f64(cnt[i]) / total) * math.abs(conf[i] / f64(cnt[i]) - acc[i] / f64(cnt[i]))
	}
	return ece
}

// sb_summarize aggregates per-sample results → accuracy/confusion/mean posterior/ECE.
fn sb_summarize(results []StructureCaseResult) StructureBenchmarkSummary {
	mut confusion := map[string]map[string]int{}
	mut correct := 0
	mut posterior_sum := map[string]f64{}
	for r in results {
		if r.true_family !in confusion {
			confusion[r.true_family] = map[string]int{}
		}
		confusion[r.true_family][r.predicted]++
		if r.predicted == r.true_family {
			correct++
		}
		for k, v in r.posterior {
			posterior_sum[k] += v
		}
	}
	n := results.len
	mut posterior_mean := map[string]f64{}
	for k, v in posterior_sum {
		posterior_mean[k] = v / f64(n)
	}
	return StructureBenchmarkSummary{
		n:              n
		accuracy:       f64(correct) / f64(n)
		confusion:      confusion
		posterior_mean: posterior_mean
		ece:            sb_ece(results)
	}
}
