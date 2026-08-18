module conger

// layered_child_reconstructor.v — constrained layer child-template full-residual
// decoding (V port of src/layered_child_reconstructor.py).

// lrc_residual_scale_constrained allows all 8 geometry residuals to be learned
// for constrained layer child templates (vs the parent's [1,1,1,1,1,1,0,0]).
const lrc_residual_scale_constrained = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
