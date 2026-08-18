module conger

// forward_model.v — domain-independent forward model / simulator protocol
// (V port of src/forward_model.py).

interface ForwardModel {
	residual(observation voidptr, params []f64) f64
}
