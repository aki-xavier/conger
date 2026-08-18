module conger

// evaluator.v — continuous regression metrics + discrete scene-factor accuracy
// (V port of src/evaluator.py; single / layered-composite / textured).

import mlx

struct Evaluator {}

// FactorSpec is one discrete scene factor (name, param column).
struct FactorSpec {
	name string
	col  int
}

// evaluator_cols returns the target/factor layouts for a param width.
fn evaluator_cols(width int) ([]int, []string, []FactorSpec) {
	if width == 14 {
		return [1, 2, 3, 4, 7, 8, 9, 10], ['u0', 'v0', 's0', 'z0', 'u1', 'v1',
			's1', 'z1'], [FactorSpec{'kind0', 0}, FactorSpec{'kind1', 6},
			FactorSpec{'hue0', 5}, FactorSpec{'hue1', 11}, FactorSpec{'lcol', 12},
			FactorSpec{'ldir', 13}]
	}
	if width == 10 {
		return [1, 2, 3, 4], ['u', 'v', 's', 'z'], [FactorSpec{'kind', 0},
			FactorSpec{'hue', 5}, FactorSpec{'lcol', 6}, FactorSpec{'ldir', 7},
			FactorSpec{'tex', 8}]
	}
	return [1, 2, 3, 4], ['u', 'v', 's', 'z'], [FactorSpec{'kind', 0},
		FactorSpec{'hue', 5}, FactorSpec{'lcol', 6}, FactorSpec{'ldir', 7}]
}

// report returns the regression + factor-accuracy metric dict for a scene family.
fn (e Evaluator) report(name string, p_gt mlx.Array, t_pred mlx.Array, scene_pred [][]f64, p_train mlx.Array) map[string]f64 {
	cols, targets, factors := evaluator_cols(p_gt.dim(1))
	gt := p_gt.take_axis(mlx.array_i32(ints32(cols), [cols.len]), 1)
	base := p_train.take_axis(mlx.array_i32(ints32(cols), [cols.len]), 1).mean_axis(0,
		true)
	ss_base := gt.subtract(base).square().sum_axis(0, false)
	tp := t_pred.take_axis(mlx.arange(0.0, f64(cols.len), 1.0, .int32), 1)
	ss_res := gt.subtract(tp).square().sum_axis(0, false)
	rmse := gt.subtract(tp).square().mean_axis(0, false).sqrt()
	r2 := mlx.ones([cols.len], .float32).subtract(ss_res.divide(ss_base.maximum(mlx.f32_scalar(1e-12))))
	mut pred_flat := []f32{}
	for r in scene_pred {
		for v in r {
			pred_flat << f32(v)
		}
	}
	pred := mlx.array_f32(pred_flat, [scene_pred.len, scene_pred[0].len])
	mut out := map[string]f64{}
	for f in factors {
		acc := pred.take_axis(sel1(f.col), 1).equal(p_gt.take_axis(sel1(f.col), 1)).astype(.float32).mean().item_f32()
		out[f.name] = f64(acc)
	}
	rmse_vals := rmse.data_f32()
	r2_vals := r2.data_f32()
	for j, nm in targets {
		out['${nm}_rmse'] = f64(rmse_vals[j])
		out['${nm}_r2'] = f64(r2_vals[j])
	}
	return out
}

// ints32 converts []int to []i32.
fn ints32(a []int) []i32 {
	mut out := []i32{len: a.len}
	for i, v in a {
		out[i] = i32(v)
	}
	return out
}
