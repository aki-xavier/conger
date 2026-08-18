module conger

// structure_gate.v — visual structure-expert gate (render-residual fusion),
// V port of src/structure_gate.py on top of the generic gate.

import mlx

struct StructureGate {
	gate GenericStructureGate
}

// new_structure_gate returns the default visual gate (high birth residual,
// 0.6 posterior floor, complexity/geometry weights per the visual calibration).
fn new_structure_gate() StructureGate {
	return StructureGate{
		gate: GenericStructureGate{
			birth_residual: 10000.0
			posterior_floor: 0.6
			complexity_weight: 1.0
			geometry_weight: 5000.0
			temperature_scale: 1.0
		}
	}
}

// residual re-renders the candidate scene and returns the foreground-weighted MSE.
fn (g StructureGate) residual(est StructuredHypothesis, fl mlx.Array, fr mlx.Array) f64 {
	mut renderer, cam_l, cam_r := sr_rig()
	sc := est.scene or { return 1e18 }
	cl := renderer.render(sc, cam_l)
	cr := renderer.render(sc, cam_r)
	wl := foreground_weights(fl)
	wr := foreground_weights(fr)
	return 0.5 * (sr_masked_mse(fl, cl, wl) + sr_masked_mse(fr, cr, wr))
}

// decide fuses multiple visual structure hypotheses by render residual.
fn (g StructureGate) decide(estimates map[string]StructuredHypothesis, fl mlx.Array, fr mlx.Array) GenericStructureDecision {
	geometry_costs := sg_costs(fl, fr)
	mut stats_cache := map[string][]f64{}
	mut with_residual := map[string]StructuredHypothesis{}
	for name, est in estimates {
		family := if est.geometry_family != '' { est.geometry_family } else { name }
		mut geometry_cost := if family in geometry_costs { geometry_costs[family] } else { 0.0 }
		if est.template_delta.len > 0 {
			mut stats := ?[]f64(none)
			if family in stats_cache {
				stats = stats_cache[family]
			} else {
				stats = sg_geometry_stats(family, fl, fr)
				if sv := stats {
					stats_cache[family] = sv
				}
			}
			geometry_cost += sg_delta_cost(family, est.template_delta, stats, fl, fr)
		}
		r := g.residual(est, fl, fr)
		with_residual[name] = est.with_residual_geometry(r, geometry_cost)
	}
	return g.gate.decide_hierarchical(with_residual)
}
