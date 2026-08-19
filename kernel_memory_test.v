module conger

// kernel_memory_test.v — lazy on-demand loading and LRU eviction of kernel
// nodes: load-on-first-use, recency-ordered eviction, transparent reload,
// and output equivalence with an eagerly-loaded reference graph.
import math

// CountKernel is the inner kernel: self-feedback affine a_{t+1} = c + k·a_t.
struct CountKernel {
	c f64
	k f64
}

fn (k CountKernel) out_dim() int {
	return 1
}

fn (k CountKernel) step(ctx KernelContext) []f64 {
	return [k.c + k.k * ctx.back[0]]
}

fn test_lazy_kernel_loads_on_first_use() {
	mut loads := 0
	lp := &loads
	lk := new_lazy_kernel(1, fn [lp] () LikelihoodKernel {
		unsafe {
			*lp = *lp + 1
		}
		return CountKernel{
			c: 1.0
			k: 0.5
		}
	})
	g := KernelGraph{
		nodes: {
			'a': KernelNode{
				kernel:   lk
				feedback: ['a']
			}
		}
	}
	assert loads == 0 // not materialised at construction
	tr := run_recurrent_opts(g, []map[string][]f64{len: 50, init: map[string][]f64{}}, RecurrentOptions{
		tol: 1e-9
	}) or { panic(err) }
	assert loads == 1 // exactly one load across the whole run
	last := tr.steps.len - 1
	assert math.abs(tr.output(last, 'a')[0] - 2.0) < 1e-6
}

fn test_memory_manager_evicts_lru_and_reloads() {
	mut loads_a, mut loads_b, mut loads_c := 0, 0, 0
	pa, pb, pc := &loads_a, &loads_b, &loads_c
	ka := new_lazy_kernel(1, fn [pa] () LikelihoodKernel {
		unsafe {
			*pa = *pa + 1
		}
		return CountKernel{
			c: 1.0
			k: 0.5
		}
	})
	kb := new_lazy_kernel(1, fn [pb] () LikelihoodKernel {
		unsafe {
			*pb = *pb + 1
		}
		return CountKernel{
			c: 1.0
			k: 0.5
		}
	})
	kernel_c := new_lazy_kernel(1, fn [pc] () LikelihoodKernel {
		unsafe {
			*pc = *pc + 1
		}
		return CountKernel{
			c: 1.0
			k: 0.5
		}
	})
	mut mgr := new_kernel_memory_manager(2)
	mgr.register(ka)
	mgr.register(kb)
	mgr.register(kernel_c)
	// touch a (t=0), b (t=1), c (t=2): a is now the least recently used
	ka.step(KernelContext{
		t:    0
		back: [0.0]
	})
	kb.step(KernelContext{
		t:    1
		back: [0.0]
	})
	kernel_c.step(KernelContext{
		t:    2
		back: [0.0]
	})
	assert ka.loaded() && kb.loaded() && kernel_c.loaded()
	assert mgr.evict() == 1
	assert !ka.loaded() // evicted (oldest)
	assert kb.loaded() && kernel_c.loaded()
	assert mgr.evict() == 0 // at capacity now
	// touching a again reloads it transparently, with the same result
	out := ka.step(KernelContext{
		t:    3
		back: [0.0]
	})
	assert loads_a == 2 // initial load + reload after eviction
	assert out == [1.0]
	assert mgr.unloads == 1
}

fn test_lazy_matches_eager_reference() {
	// a lazy graph must produce the same trace as its eager reference
	mut loads := 0
	lp := &loads
	lk := new_lazy_kernel(1, fn [lp] () LikelihoodKernel {
		unsafe {
			*lp = *lp + 1
		}
		return CountKernel{
			c: 1.0
			k: 0.5
		}
	})
	gl := KernelGraph{
		nodes: {
			'a': KernelNode{
				kernel:   lk
				feedback: ['a']
			}
		}
	}
	ge := KernelGraph{
		nodes: {
			'a': KernelNode{
				kernel:   CountKernel{
					c: 1.0
					k: 0.5
				}
				feedback: ['a']
			}
		}
	}
	obs := []map[string][]f64{len: 20, init: map[string][]f64{}}
	tl := run_recurrent_opts(gl, obs, RecurrentOptions{}) or { panic(err) }
	te := run_recurrent_opts(ge, obs, RecurrentOptions{}) or { panic(err) }
	for t in 0 .. 20 {
		assert tl.output(t, 'a') == te.output(t, 'a')
	}
}
