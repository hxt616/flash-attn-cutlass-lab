PY ?= .venv/bin/python
BUILD_DIR ?= build
CUDA_ARCH ?= 89
BENCH_ARGS ?= --dims 64,128 --seqlens 512,1024,2048 --iters 50 --warmup 10 --repeats 3 --baselines sdpa

.PHONY: all build ext test bench bench-causal bench-decode sanitize clean

all: build ext

build:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
	cmake --build $(BUILD_DIR) -j

ext:
	$(PY) setup.py build_ext --inplace --force

test:
	$(PY) -m pytest -q

bench:
	$(PY) benchmarks/bench_fwd.py $(BENCH_ARGS) --out benchmarks/results/bench_fwd_noncausal.csv
	$(PY) benchmarks/plot.py benchmarks/results/bench_fwd_noncausal.csv

bench-causal:
	$(PY) benchmarks/bench_fwd.py $(BENCH_ARGS) --causal --out benchmarks/results/bench_fwd_causal.csv
	$(PY) benchmarks/plot.py benchmarks/results/bench_fwd_causal.csv

bench-decode:
	$(PY) benchmarks/bench_decode.py

sanitize:
	bash scripts/sanitize.sh

clean:
	rm -rf $(BUILD_DIR) .pytest_cache build *.egg-info
