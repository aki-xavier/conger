# Build helper.
#
# `mlx` resolves from the default V module dir `~/.vmodules`
# (symlink set up once per machine; see README § 依赖).

.PHONY: test fmt

# `-no-memory-limit`: the v3 compiler's default 2.3 GiB guard can trip on
# large generated tables when compiling the full module.
test:
	v -gc boehm -no-memory-limit test .

fmt:
	v fmt -w .
