# Contributions

This repository is a rework of the MIT-licensed
[tiny-flash-attention](https://github.com/66RING/tiny-flash-attention) skeleton.
The upstream project provides the CuTe FlashAttention-2 learning kernel and
tutorials. The list below separates upstream material from my changes so the
work can be discussed honestly in interviews.

## Upstream material (kept, with attribution)

- `include/kernel_traits.h`, `include/utils.h`
- The overall QK -> online-softmax -> PV CuTe structure
- `docs/cutlass_cute_tutorial_{en,zh}.md`
- The original parameter structs and the idea of a standalone + pybind build

## My changes

### Correctness and kernel

- Moved the kernel into a framework-agnostic header (`include/flash_fwd_kernel_impl.cuh`)
  so the standalone binary and the PyTorch extension share exactly one
  implementation.
- Fixed the upstream standalone reference: the softmax and PV kernels were
  commented out, so the old `all_close` compared an uninitialized buffer. Added
  an independent reference kernel (`standalone/reference.cuh`) and a host-side
  comparison with max/mean/relative error reporting.
- Fixed the base-2 softmax bug in the standalone path: `exp2` requires
  `softmax_scale * log2(e)`. The kernel now carries `scale_log2` and computes
  LSE as `max * scale + log(sum)`.
- Added row-level predicates for ragged 8-row copy tiles. The tile-level
  predicate alone writes up to 7 rows past the logical sequence, which
  corrupted packed varlen buffers and could segfault on the last sequence.
- Added arbitrary-length handling for Q/K/V and output (S not a multiple of
  64, and not a multiple of 8 inside the last tile).
- Added GQA/MQA by mapping `q_head -> kv_head = q_head / (nheads / nheads_k)`
  for K/V pointers; upstream only read `h_k` fields and used the q head offset.
- Added LSE output with the same empty-row convention as FlashAttention
  (`+inf` for all-masked rows).
- Added bf16 through the CuTe atom selection, a generic fp32 fragment
  conversion, and a bf16 test path.
- Added the fused varlen API. A device-side kernel builds a
  `tile -> (sequence, m_block)` map; all sequences' Q tiles live in one grid
  (`total_q_tiles_upper × nheads`), so the hardware scheduler backfills idle
  SMs across sequence boundaries. No per-sequence launch and no `cu_seqlens`
  device-to-host copy. Static fused version measures 0.96× (non-causal) /
  0.92× (causal) official FA 2.8.3; a persistent work queue is still future
  work.
- Added `kBlockN` dispatch from measurements: non-causal D=64 uses 128,
  everything else uses 64.
- Added a paged-KV decode kernel and `flash_attn_with_kvcache`-style API. K/V
  tiles are addressed through `block_table` (no gather into a contiguous
  buffer), with GQA and page sizes 64/128/256. Decode uses split-KV + a
  log-sum-exp combine kernel with an auto heuristic (`num_splits` override
  available), and measures 0.50–0.76× official `flash_attn_with_kvcache`.

### PyTorch integration

- Stream-safe launch: `at::cuda::getCurrentCUDAStream()`, `CUDAGuard`,
  `C10_CUDA_KERNEL_LAUNCH_CHECK()`, no `cudaDeviceSynchronize()` inside the
  library.
- Input contract: CUDA, contiguous, fp16/bf16, matching dtype/head dim,
  head dim multiple of 32 and <= 256, GQA ratio integral.
- Fixed upstream's `void*` arithmetic and `tools/utils` include-path typo;
  the extension now builds with `setup.py build_ext --inplace`.
- Device-side varlen tile map (no host prefix-sum copy; the upstream code
  dereferenced a device pointer from host).

### Build and reproducibility

- Forced `CMAKE_CUDA_ARCHITECTURES` away from CMake's default 52; the default
  silently selected SM75 CuTe atoms and crashed at runtime on sm_89.
- Added automatic include paths for CUTLASS, `makefile`, `scripts/setup_env.sh`,
  `scripts/run_all.sh`, and `scripts/sanitize.sh`.
- Added CSV/PNG benchmark artifacts and a plot script.

### Tests and docs

- 233 pytest cases: shape matrix, causal, custom scale, LSE, GQA, varlen,
  zero-length sequences, bf16, API rejection tests, and an
  error-vs-PyTorch criterion (`our_error <= 2 * torch_error + 1e-3`).
- `compute-sanitizer` script for ragged shapes.
- Design, tuning, debugging, and interview notes under `docs/`.

## Not implemented (deliberately out of scope)

- Backward pass, dropout, ALiBi/RoPE/window masks.
- A decode-specific small-Q-tile schedule and a multi-stage decode pipeline;
  the current paged decode is correct and split-KV, but reuses the 64-row
  prefill Q tile.
- A persistent/dynamic-work-queue varlen scheduler (the current fused varlen
  uses a static atomic tile map); prefill `block_table` / prefix-cache
  indexing.
- Hopper TMA/wgmma, warp specialization.
- Multi-stage `cp.async` double buffering (the current kernel overlaps V/K
  loads but uses a single K/V buffer).
