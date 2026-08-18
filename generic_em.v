module conger

// generic_em.v — domain-independent expectation-maximisation (EM) loop.
//
// The generation model is abstracted as "observation X ← (latent Z, params θ)";
// EM iterates E/M between them:
//
//   E step: q(Z) = P(Z | X, θ_t)            (soft posterior / responsibilities)
//   M step: θ_{t+1} = argmax E_q[log P(X, Z | θ)]
//
// The loop only handles iteration, temperature (E-step sharpening), damping
// (M-step stabilisation) and convergence monitoring; the concrete generative
// model is injected as a generic struct type M with observation type O and
// responsibility type R.
import math

// EMResult captures one EM run.
pub struct EMResult[T] {
pub:
	params           []f64
	responsibilities T
	log_likelihood   f64
	iterations       int
	trajectory       []f64
}

// EMLoop iterates E/M to convergence. M is the model struct, O the observation
// type, R the responsibility type (both inferred from the concrete model's
// method signatures at each instantiation).
pub struct EMLoop[M, O, R] {
pub:
	max_iters   int = 50
	tol         f64 = 1e-6
	temperature f64 = 1.0
	damping     f64 = 0.0
pub mut:
	model M
}

// run iterates E/M from init_params and returns the converged state + trajectory.
pub fn (mut e EMLoop[M, O, R]) run(observation O, init_params []f64) EMResult[R] {
	// NB: explicit panic, not `assert` — V strips asserts in `-prod` builds.
	if e.max_iters < 1 {
		panic('EMLoop.run: max_iters must be >= 1 (got ${e.max_iters})')
	}
	mut params := init_params.clone()
	mut resp := R{}
	mut trajectory := []f64{}
	mut prev_ll := -1e300
	for _ in 0 .. e.max_iters {
		resp = e.model.responsibilities(params, observation, e.temperature)
		params = e.model.maximize(resp, observation, params, e.damping)
		ll := e.model.log_likelihood(params, observation)
		trajectory << ll
		if trajectory.len > 1 && math.abs(ll - prev_ll) < e.tol {
			break
		}
		prev_ll = ll
	}
	return EMResult[R]{
		params:           params
		responsibilities: resp
		log_likelihood:   trajectory[trajectory.len - 1]
		iterations:       trajectory.len
		trajectory:       trajectory
	}
}
