# V port build helper.
#
# `cga` and `mlx` resolve from the default V module dir `~/.vmodules`
# (symlinks set up once per machine; see README § 依赖).

.PHONY: test fmt

# `-no-memory-limit` mirrors the cga V port: the v3 compiler's default 2.3 GiB
# guard can trip on large generated tables when compiling the full module.
test:
	v -no-memory-limit test .

fmt:
	v fmt -w .
