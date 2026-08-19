# Build helper.
#
# `mlx` resolves from the default V module dir `~/.vmodules`
# (symlink set up once per machine; see README § 依赖).

.PHONY: test smoke fmt

# `-no-memory-limit`: the v3 compiler's default 2.3 GiB guard can trip on
# large generated tables when compiling the full module.
test:
	v -gc boehm -no-memory-limit test .

# smoke: build and run every example end-to-end (catches example rot that
# `make test` misses — examples are not part of the *_test.v suite).
smoke:
	v -gc boehm -no-memory-limit run examples/main_pipeline.v > /dev/null
	v -gc boehm -no-memory-limit run examples/iris_classification.v > /dev/null
	v -gc boehm -no-memory-limit run examples/mrf_lattice.v > /dev/null
	@echo "smoke: 3 examples OK"

fmt:
	v fmt -w .
