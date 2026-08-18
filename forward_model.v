module conger

// forward_model.v — domain-independent forward model / simulator protocol
// (V port of src/forward_model.py).

pub interface ForwardModel {
	residual(observation voidptr, params []f64) f64
}
