module conger

// kernel_memory.v — model-memory management for kernel networks: lazy
// on-demand loading and LRU forgetting of kernel nodes (`model_memory.v`
// gives MixtureSPN split/load/forget at the persistence layer; this file is
// the kernel-graph counterpart at the execution layer).
//
// `LazyKernel` wraps a loader closure: the inner kernel is materialised on
// the first `step` that needs it and stays resident until evicted. Usage
// recency is tracked with `ctx.t` (the graph's step index), so no global
// clock is needed. `KernelMemoryManager` holds the lazy kernels of a graph
// and evicts the least-recently-used ones beyond a capacity — the natural
// call points are between recurrent runs, or between run_residual batches
// (residual scheduling already concentrates updates on hot nodes, so the
// LRU order tracks actual compute demand).
//
// The loader closure is where the persistence layer plugs in: it can call
// `load_components` / assemble a MixtureSPN from a split archive, and the
// manager's eviction is where `forget_components`-style cleanup belongs.

// LazyState is the shared mutable cell behind a LazyKernel (interface
// method receivers are values, so the mutable state lives in a `shared`
// struct accessed under `lock`/`rlock`).
pub struct LazyState {
pub mut:
	inner     ?LikelihoodKernel
	last_used u64 // ctx.t of the most recent step call
	loads     int // materialisation count (stats)
}

// LazyKernel is a LikelihoodKernel that loads its inner kernel on first use.
pub struct LazyKernel {
pub:
	dim   int
	load  fn () LikelihoodKernel @[required]
	state shared LazyState
}

// new_lazy_kernel wraps `load`; `dim` must equal the inner kernel's out_dim
// (declared eagerly because the graph validates widths before first use).
pub fn new_lazy_kernel(dim int, load fn () LikelihoodKernel) LazyKernel {
	if dim < 1 {
		panic('new_lazy_kernel: dim must be >= 1')
	}
	return LazyKernel{
		dim:   dim
		load:  load
		state: LazyState{}
	}
}

pub fn (k LazyKernel) out_dim() int {
	return k.dim
}

pub fn (k LazyKernel) step(ctx KernelContext) []f64 {
	lock k.state {
		k.state.last_used = u64(ctx.t)
		if inner := k.state.inner {
			return inner.step(ctx)
		}
		inner := k.load()
		k.state.loads++
		k.state.inner = inner
		return inner.step(ctx)
	}
	panic('unreachable')
}

// loaded reports whether the inner kernel is currently resident.
pub fn (k LazyKernel) loaded() bool {
	rlock k.state {
		return k.state.inner != none
	}
	panic('unreachable')
}

// last_used returns the recency stamp (0 if never used).
pub fn (k LazyKernel) last_used() u64 {
	rlock k.state {
		return k.state.last_used
	}
	panic('unreachable')
}

// KernelMemoryManager evicts least-recently-used lazy kernels beyond a
// capacity.
pub struct KernelMemoryManager {
pub:
	capacity int
pub mut:
	kernels []LazyKernel
	unloads int // cumulative eviction count (stats)
}

// new_kernel_memory_manager creates a manager keeping at most `capacity`
// kernels resident.
pub fn new_kernel_memory_manager(capacity int) KernelMemoryManager {
	if capacity < 1 {
		panic('new_kernel_memory_manager: capacity must be >= 1')
	}
	return KernelMemoryManager{
		capacity: capacity
	}
}

// register adds a lazy kernel to the manager.
pub fn (mut m KernelMemoryManager) register(k LazyKernel) {
	m.kernels << k
}

// evict unloads resident kernels, least-recently-used first, until at most
// `capacity` remain. Returns the number of kernels unloaded.
pub fn (mut m KernelMemoryManager) evict() int {
	mut resident := []int{}
	for i, k in m.kernels {
		if k.loaded() {
			resident << i
		}
	}
	if resident.len <= m.capacity {
		return 0
	}
	// sort resident indices by recency ascending (oldest first)
	resident.sort_with_compare(fn [m] (a &int, b &int) int {
		la, lb := m.kernels[*a].last_used(), m.kernels[*b].last_used()
		return if la < lb {
			-1
		} else if la > lb {
			1
		} else {
			0
		}
	})
	mut n := 0
	for i in resident[..resident.len - m.capacity] {
		lock m.kernels[i].state {
			m.kernels[i].state.inner = none
		}
		n++
	}
	m.unloads += n
	return n
}
