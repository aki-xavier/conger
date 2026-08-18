module conger

// roughness_head.v — shape-descriptor → roughness 1-NN kernel regression head
// (V port of src/roughness_head.py; numpy replaced with pure-V f64 arrays).
import math

const roughness_default = 0.55

struct RoughnessHead {
mut:
	x  [][]f64 // (N,16)
	y  []f64   // (N,)
	mu []f64
	sd []f64
}

// rh_fit stores sphere samples' shape descriptors + roughness labels.
fn (mut r RoughnessHead) rh_fit(x [][]f64, y []f64) {
	r.x = x.clone()
	r.y = y.clone()
	n := x.len
	d := x[0].len
	r.mu = []f64{len: d}
	for j in 0 .. d {
		mut s := 0.0
		for i in 0 .. n {
			s += x[i][j]
		}
		r.mu[j] = s / f64(n)
	}
	r.sd = []f64{len: d}
	for j in 0 .. d {
		mut s := 0.0
		for i in 0 .. n {
			dd := x[i][j] - r.mu[j]
			s += dd * dd
		}
		r.sd[j] = math.sqrt(s / f64(n)) + 1e-9
	}
}

// rh_predict returns the nearest-neighbour roughness for each row.
fn (r RoughnessHead) rh_predict(x [][]f64) []f64 {
	n := x.len
	d := x[0].len
	mut out := []f64{len: n}
	for i in 0 .. n {
		mut best := 1e18
		mut best_idx := 0
		for k in 0 .. r.x.len {
			mut dist := 0.0
			for j in 0 .. d {
				dd := (r.x[k][j] - r.mu[j]) / r.sd[j] - (x[i][j] - r.mu[j]) / r.sd[j]
				dist += dd * dd
			}
			if dist < best {
				best = dist
				best_idx = k
			}
		}
		out[i] = r.y[best_idx]
	}
	return out
}

// roughness_r2 returns 1 − SS_res/SS_base (baseline = gt mean).
fn roughness_r2(pred []f64, gt []f64) f64 {
	mut mean := 0.0
	for v in gt {
		mean += v
	}
	mean /= f64(gt.len)
	mut ss_res := 0.0
	mut ss_base := 0.0
	for i in 0 .. gt.len {
		ss_res += (gt[i] - pred[i]) * (gt[i] - pred[i])
		ss_base += (gt[i] - mean) * (gt[i] - mean)
	}
	return 1.0 - ss_res / fmax2(ss_base, 1e-12)
}
