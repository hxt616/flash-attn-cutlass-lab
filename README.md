# flash-attn-cutlass-lab

A from-scratch-in-spirit FlashAttention-2 **forward** kernel built with
[CUTLASS CuTe](https://github.com/NVIDIA/cutlass), packaged with a reproducible
correctness and performance harness. It is my rework of the excellent
[tiny-flash-attention](https://github.com/66RING/tiny-flash-attention) learning
repo: the upstream project supplies the CuTe skeleton and tutorials, while this
repository adds correctness fixes, arbitrary-length/varlen support, GQA,
a PyTorch extension, a pytest matrix, benchmarks, and documentation.

> Attribution: see [`NOTICE`](NOTICE). Upstream is MIT; CUTLASS and the
> FlashAttention algorithm/reference patterns are BSD-3. Files derived from
> upstream keep their copyright notice. This is not a claim of original
> authorship over the upstream skeleton.

## Features

- FlashAttention-2 forward in CuTe (`SM80_16x8x16` MMA, `ldmatrix`, `cp.async`).
- `fp16` and `bf16`; head dims 32, 64, 128.
- Arbitrary sequence length, including non-multiples of the 64-row tile (an
  element-wise predicate handles ragged 8-row copy tiles).
- Causal and non-causal masks; the causal path skips fully-masked KV blocks.
- GQA / MQA (`nheads_k < nheads`) with correct KV-head mapping.
- Variable-length attention (`cu_seqlens`) without padding. The varlen path
  currently launches one kernel per sequence; it is correct but not a single
  fused persistent kernel.
- PyTorch extension that launches on the current CUDA stream, returns
  `(out, lse)`, and validates dtype/shape/contiguity.
- 219 pytest cases (shapes, causal, GQA, varlen, bf16, edge cases, API
  contracts, and an official-style error-vs-PyTorch criterion).
- Reproducible benchmark harness with CSV/PNG output and SDPA comparison.

## Repo layout

```text
include/
  flash_fwd_kernel_impl.cuh   # framework-agnostic kernel (CuTe)
  flash_fwd_launch.h          # grid/smem launch
  flash_params.h              # shared launch parameters
  kernel_traits.h, utils.h    # upstream-derived CuTe helpers
csrc/
  flash_fwd_kernel.cu         # PyTorch host wrapper (fixed + varlen)
  flash_api.cpp               # pybind11 module
  static_switch.h
standalone/
  main.cu                     # correctness vs naive + cudaEvent benchmark
  reference.cuh               # simple, auditable reference kernel
tests/                        # pytest matrix + fp64/fp32 reference
benchmarks/                   # bench_fwd.py, plot.py, results/
docs/                         # design, tuning, debugging, interview notes
```

## Quick start

```bash
# 1. Environment (uv is fast; a plain venv + pip works too)
bash scripts/setup_env.sh

# 2. Standalone binary (no PyTorch needed)
make build

# 3. PyTorch extension (in-place)
make ext

# 4. Tests
make test

# 5. Benchmarks (non-causal, then causal)
make bench
make bench-causal
# or install editable: pip install -e . --no-build-isolation
```

Standalone quick checks:

```bash
./build/flash_attn_lab --B 1 --H 2 --Hk 2 --S 97 --D 64 --causal
./build/flash_attn_lab --B 2 --H 8 --Hk 8 --S 2048 --D 64 --bn 128 --iters 200
./build/flash_attn_lab --B 1 --H 2 --Hk 2 --S 129 --D 128 --causal --no-bench
```

Python API (also available as `flash_attn_lab.flash_attn_func`):

```python
from flash_attn_lab import flash_attn_func, flash_attn_varlen_func

out, lse = flash_attn_func(q, k, v, causal=True)   # q: (B,S,H,D)
out, lse = flash_attn_varlen_func(q, k, v, cu_q, cu_k, causal=True)
```

`q` is `(batch, seqlen, nheads, headdim)`; `k/v` are
`(batch, seqlen_k, nheads_k, headdim)`; inputs must be contiguous
`fp16`/`bf16`. `lse` is `float32` with shape `(batch, nheads, seqlen_q)`.
Varlen inputs are `(total_tokens, nheads, headdim)` and `lse` is
`(total_tokens, nheads)`.

## Correctness

Reference: an independent, easy-to-audit CUDA kernel
(`standalone/reference.cuh`) and a PyTorch materialized reference
(`tests/reference.py`). The pytest matrix covers:

- `dtype ∈ {fp16, bf16}`, `D ∈ {32, 64, 128}`
- `S ∈ {1, 2, 17, 63, 64, 65, 97, 127, 128, 129, 200, 257, 512, 1024}`
- causal / non-causal, custom scales
- GQA ratios `Hk ∈ {H, H/2, H/4, 1}`
- varlen with zero-length and ragged sequences
- API rejection of non-contiguous / wrong-dtype / invalid head dims
- official-style check: `our_error <= 2 * pytorch_error + 1e-3`

```text
219 passed
```

Run `make sanitize` to exercise small ragged shapes under
`compute-sanitizer` (`memcheck`, `racecheck`, `initcheck`, `synccheck`).

## Performance (RTX 4060 Laptop 8 GB, WSL2, CUDA 12.4)

Method: fp16, `B=2, H=8`, median of 3 repeats of 50 timed iterations after 10
warmup iterations; PyTorch SDPA is the baseline. Laptop GPUs throttle, so treat
±10% as run-to-run noise. Raw data:
[`benchmarks/results/`](benchmarks/results/).

**Non-causal (TFLOPS)**

| D | S | this kernel | SDPA |
|---:|---:|---:|---:|
| 64 | 512 | 20.8 | 21.8 |
| 64 | 1024 | 24.3 | 23.7 |
| 64 | 2048 | 25.2 | 29.7 |
| 128 | 512 | 24.8 | 24.9 |
| 128 | 1024 | 28.5 | 26.3 |
| 128 | 2048 | 29.3 | 29.1 |

**Causal (TFLOPS)**

| D | S | this kernel | SDPA |
|---:|---:|---:|---:|
| 64 | 512 | 15.3 | 14.2 |
| 64 | 1024 | 19.8 | 18.2 |
| 64 | 2048 | 22.9 | 23.3 |
| 128 | 512 | 18.3 | 18.9 |
| 128 | 1024 | 23.0 | 23.6 |
| 128 | 2048 | 26.5 | 27.3 |

The kernel is competitive with SDPA's flash backend on this laptop GPU; D=128
is often ahead because the extra K-dimension work amortizes the non-matmul
softmax overhead. See [`docs/tuning.md`](docs/tuning.md) for the block-N
experiment and bottleneck analysis.

## What I changed vs upstream

See [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md). Highlights:

- Rebuilt the kernel as a framework-agnostic header and added a PyTorch
  extension with stream-safe launch and input validation.
- Fixed the standalone reference (it never executed softmax/PV) and the
  `exp2` scale (missing `log2(e)`).
- Added ragged-tile predication, arbitrary sequence lengths, GQA, bf16, varlen,
  and LSE.
- Added the pytest matrix, sanitizer script, benchmark harness, and docs.
- Tuned `kBlockN` per dtype/head-dim/causal case (measured, not guessed).

## Known limitations

- Forward only; no backward, dropout, ALiBi/RoPE/window, paged-KV decode.
- SM80/SM89 path only; no Hopper TMA/wgmma or warp specialization.
- Varlen is a host loop over sequences, not one fused persistent kernel.
- Single-stage `cp.async` pipeline (K/V loads overlap the QK GEMM, but there is
  no multi-stage ring buffer yet).
- Only contiguous inputs; head dim must be a multiple of 32 and ≤ 256.
- The official `flash-attn` package was not importable in this Python 3.10
  environment, so SDPA is the reported baseline. The benchmark script will add
  official FA automatically when it is installed.

## License

MIT for this repository; upstream MIT copyright retained. CUTLASS and
FlashAttention-derived parts are BSD-3. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
