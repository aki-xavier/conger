module conger

// toy_series_expert.v — MixtureSPN-wrapped non-visual structure expert
// (V port of src/toy_series_expert.py).
import math
import mlx

struct ToySeriesExpert {
	family ToySeriesFamily
	net    MixtureSPN
}

// train_toy_expert trains an instance-level MixtureSPN expert for one mechanism.
fn train_toy_expert(mechanism string, n int, seed u64) ToySeriesExpert {
	family := new_toy_series_family(mechanism)
	p := family.sample(n, seed)
	y := family.simulate(p)
	mut feats := []mlx.Array{}
	for i in 0 .. n {
		row := y.take_axis(sel1(i), 0).squeeze_axis(0)
		feats << family.encode(row)
	}
	f := mlx.stack(feats, 0)
	zeros := mlx.zeros([n], .int32)
	classes := zeros.expand_dims(1)
	net := fit_mixture_spn(f, p, zeros, 1e-3, classes, [1], 0)
	return ToySeriesExpert{
		family: family
		net:    net
	}
}

// estimate returns a StructuredHypothesis for one observation sequence.
fn (e ToySeriesExpert) estimate(observation mlx.Array) StructuredHypothesis {
	f := e.family.encode(observation).expand_dims(0)
	tm, _, r := e.net.predict(f)
	params := tm.take_axis(sel1(0), 0).squeeze_axis(0).data_f32()
	mut pf := []f64{len: params.len}
	for i, v in params {
		pf[i] = f64(v)
	}
	residual := e.family.residual(observation, pf)
	max_r := f64(r.max().item_f32()) + 1e-12
	// responsibility-concentration novelty, normalised by log(K); a single
	// component (K == 1) is fully concentrated by construction, so that term
	// is 0 and only the residual contributes.
	k := r.dim(1)
	mut resp_novelty := 0.0
	if k > 1 {
		resp_novelty = -math.log(max_r) / math.log(f64(k))
	}
	novelty := resp_novelty + math.log(1.0 + residual)
	return StructuredHypothesis{
		structure_id:       e.family.mechanism
		params:             pf
		responsibility_max: max_r
		posterior_entropy:  0.0
		residual:           residual
		complexity:         f64(e.family.n_params())
		novelty_score:      novelty
	}
}
