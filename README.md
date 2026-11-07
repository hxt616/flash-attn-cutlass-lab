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
- Variable-length attention (`cu_seqlens`) without padding. Varlen now uses a
  single fused kernel: a small device-side kernel builds a `tile -> (sequence,
  m_block)` map, and all sequences' Q tiles share one grid. No per-sequence
  launch and no `cu_seqlens` device-to-host copy.
- Paged-KV decode: `flash_attn_with_kvcache`-style API with `block_table`,
  `cache_seqlens`, GQA and page sizes 64/128/256. The kernel reads K/V tiles
  directly from paged cache (no gather into a contiguous buffer).
- PyTorch extension that launches on the current CUDA stream, returns
  `(out, lse)`, and validates dtype/shape/contiguity.
- 231 pytest cases (shapes, causal, GQA, varlen, decode/paged KV, bf16, edge
  cases, API contracts, and an official-style error-vs-PyTorch criterion).
- Reproducible benchmark harness with CSV/PNG output and SDPA/official
  comparisons.

## Repo layout

```text
include/
  flash_fwd_kernel_impl.cuh   # framework-agnostic kernel (CuTe)
  flash_fwd_launch.h          # grid/smem launch
  flash_params.h              # shared launch parameters
  kernel_traits.h, utils.h    # upstream-derived CuTe helpers
csrc/
  flash_fwd_kernel.cu         # PyTorch host wrapper (fixed + varlen + decode)
  flash_api.cpp               # pybind11 module
  static_switch.h
standalone/
  main.cu                     # correctness vs naive + cudaEvent benchmark
  reference.cuh               # simple, auditable reference kernel
tests/                        # pytest matrix + fp64/fp32 reference (incl. decode)
benchmarks/                   # bench_fwd.py, bench_official.py, bench_decode.py, results/
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

Python API (also available as `flash_attn_lab.*`):

```python
from flash_attn_lab import (
    flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache,
)

out, lse = flash_attn_func(q, k, v, causal=True)   # q: (B,S,H,D)
out, lse = flash_attn_varlen_func(q, k, v, cu_q, cu_k, causal=True)

# paged-KV decode: q (B,1,H,D), cache (num_pages,page_size,Hk,D)
out, lse = flash_attn_with_kvcache(
    q, k_cache, v_cache, block_table, cache_seqlens
)
```

`q` is `(batch, seqlen, nheads, headdim)`; `k/v` are
`(batch, seqlen_k, nheads_k, headdim)`; inputs must be contiguous
`fp16`/`bf16`. `lse` is `float32` with shape `(batch, nheads, seqlen_q)`.
Varlen inputs are `(total_tokens, nheads, headdim)` and `lse` is
`(total_tokens, nheads)`. Decode `block_table` is `(batch, max_pages)` int32
and `cache_seqlens` is `(batch,)` int32.

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
231 passed
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
| 64 | 512 | 19.6 | 21.8 |
| 64 | 1024 | 22.8 | 23.7 |
| 64 | 2048 | 23.5 | 29.7 |
| 128 | 512 | 24.6 | 25.1 |
| 128 | 1024 | 28.1 | 26.4 |
| 128 | 2048 | 28.7 | 29.1 |

**Causal (TFLOPS)**

| D | S | this kernel | SDPA |
|---:|---:|---:|---:|
| 64 | 512 | 14.8 | 14.1 |
| 64 | 1024 | 19.4 | 18.2 |
| 64 | 2048 | 22.4 | 22.1 |
| 128 | 512 | 18.4 | 19.7 |
| 128 | 1024 | 23.3 | 24.5 |
| 128 | 2048 | 26.1 | 27.4 |

The kernel is competitive with SDPA's flash backend on this laptop GPU; D=128
is often ahead because the extra K-dimension work amortizes the non-matmul
softmax overhead. See [`docs/tuning.md`](docs/tuning.md) for the block-N
experiment and bottleneck analysis, and
[`docs/performance-gap-analysis-zh.md`](docs/performance-gap-analysis-zh.md)
for a detailed explanation of every performance gap (prefill/varlen/decode).

**vs official flash-attn 2.8.3** (same GPU, bf16, Qwen3-0.6B shape:
`B=1, H=16, Hk=8, D=128`; ratio = our TFLOPS / official TFLOPS)

| mode | S | causal | ratio |
|---|---:|---:|---:|
| fixed | 512 | no | 0.98 |
| fixed | 1024 | no | 1.06 |
| fixed | 2048 | no | 0.97 |
| varlen (fused) | 512/1024/2048 | no | 0.96 |
| fixed | 512 | yes | 0.93 |
| fixed | 1024 | yes | 0.94 |
| fixed | 2048 | yes | 0.94 |
| varlen (fused) | 512/1024/2048 | yes | 0.92 |

Max output difference vs official is ~1e-3 in bf16. Fixed-length and fused
varlen are now both essentially at parity (0.92–1.06×). Many short sequences
(`32 × 16`) land at 0.88 (non-causal) / 0.86 (causal); the remaining gap is the
causal tail and the static tile order. The official baseline runs in the
nano-vllm Python 3.12 environment (`torch 2.6.0+cu124`, official
`flash_attn 2.8.3`). Raw data:
`benchmarks/results/bench_vs_official.csv`.

**Paged-KV decode** (bf16, `H=16, Hk=8, D=128`, page size 256, no split-KV)

| B | context | ours ms | official ms | speed vs official |
|---:|---:|---:|---:|---:|
| 1 | 512 | 0.0256 | 0.0184 | 0.72× |
| 1 | 2048 | 0.0888 | 0.0350 | 0.39× |
| 1 | 4096 | 0.1727 | 0.0587 | 0.34× |
| 8 | 512 | 0.1054 | 0.0552 | 0.52× |
| 8 | 2048 | 0.3950 | 0.2102 | 0.53× |
| 8 | 4096 | 0.6970 | 0.4050 | 0.58× |

Correctness matches official within ~1e-3. This is a **first paged-decode
version**: K/V are read directly through `block_table`, but it uses one CTA per
`(batch, head)` and the same 64-row Q tile as prefill. Official uses split-KV
and a decode-specific schedule, so our latency is 0.34–0.72× official. Split-KV
+ combine is the next optimization; raw data:
`benchmarks/results/bench_decode.csv`.

## What I changed vs upstream

See [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md). Highlights:

- Rebuilt the kernel as a framework-agnostic header and added a PyTorch
  extension with stream-safe launch and input validation.
- Fixed the standalone reference (it never executed softmax/PV) and the
  `exp2` scale (missing `log2(e)`).
- Added ragged-tile predication, arbitrary sequence lengths, GQA, bf16, fused
  varlen (device-side tile map), and LSE.
- Added the pytest matrix, sanitizer script, benchmark harness, and docs.
- Tuned `kBlockN` per dtype/head-dim/causal case (measured, not guessed).

## Known limitations

- Forward only; no backward, dropout, ALiBi/RoPE/window.
- SM80/SM89 path only; no Hopper TMA/wgmma or warp specialization.
- Varlen uses a static fused grid with an atomic device-side tile map; a
  persistent work-queue scheduler would further reduce the causal tail.
- Decode is a first paged-KV version: no split-KV/combine yet, and it uses the
  64-row prefill Q tile; this is why decode is 0.34–0.72× official.
- Prefill does not yet support `block_table`/prefix-cache indexing; only the
  no-prefix-cache prefill path is supported. Decode does support `block_table`.
- Single-stage `cp.async` pipeline (K/V loads overlap the QK GEMM, but there is
  no multi-stage ring buffer yet).
- Only contiguous inputs; head dim must be a multiple of 32 and ≤ 256.
- The official comparison uses a separate Python 3.12 environment because the
  project's dev venv is Python 3.10; the kernel sources are identical. See
  `benchmarks/bench_official.py`.

## License

MIT for this repository; upstream MIT copyright retained. CUTLASS and
FlashAttention-derived parts are BSD-3. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
