# V port build helper.
#
# The V module resolves its `cga` and `mlx` dependencies through the
# project-local `.vmodules` directory, so every build sets VMODULES.

VMODULES := $(CURDIR)/.vmodules

.PHONY: test fmt

# `-no-memory-limit` mirrors the cga V port: the v3 compiler's default 2.3 GiB
# guard can trip on large generated tables when compiling the full module.
test:
	VMODULES=$(VMODULES) v -no-memory-limit test .

fmt:
	v fmt -w .
