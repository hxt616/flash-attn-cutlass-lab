# Tuning notes

## Methodology

- Isolate the kernel with the standalone binary (no Python allocation):
  `./build/flash_attn_lab --B 2 --H 8 --Hk 8 --S ... --D ... --bn ...`.
- Use `cudaEvent` around a warmup + 100-300 iteration loop; report ms/iter and
  TFLOPS. Re-run suspicious cases because the 4060 Laptop GPU throttles.
- Confirm correctness at every new block configuration (`--no-bench` uses the
  independent reference).
- The Python benchmark adds `torch::empty` allocations and LSE writes; use it
  for end-to-end comparisons with SDPA, and the standalone binary to attribute
  changes to the kernel.

## Block-N experiment (standalone, fp16, B=2, H=8)

| D | S | causal | kBlockN | ms/iter | TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 64 | 2048 | no | 64 | 0.6758 | 25.42 |
| 64 | 2048 | no | 128 | 0.6198 | 27.72 |
| 64 | 2048 | yes | 64 | 0.3359 | 25.59 |
| 64 | 2048 | yes | 128 | 0.3664 | 23.45 |
| 32 | 2048 | no | 64 | 0.3108 | 27.63 |
| 32 | 2048 | no | 128 | 0.3701 | 23.21 |
| 128 | 2048 | no | 64 | 1.1733 | 29.28 |
| 128 | 2048 | no | 128 | 1.4089 | 24.39 |
| 128 | 1024 | yes | 64 | 0.2025 | 21.23 |
| 128 | 1024 | yes | 128 | 0.2233 | 19.25 |

Decision: use `kBlockN=128` only for `D=64` non-causal; use 64 for all other
combinations. The dispatch is in `run_flash_fwd_` in
`csrc/flash_fwd_kernel.cu`.

Why:

- A larger `BlockN` halves the number of loop iterations and softmax rescale
  steps, but increases shared memory (K/V tiles) and register pressure. For
  D=64 non-causal the trade is positive; for D=32 the smaller K tile already
  fits more CTAs, and for D=128 the 128-wide K/V tile cuts occupancy too much.
- Causal kernels do fewer KV blocks per Q tile; the tail blocks are partially
  masked, so the larger N tile adds work without enough reuse.

## End-to-end vs PyTorch SDPA (fp16, B=2, H=8)

See `benchmarks/results/bench_fwd_{noncausal,causal}.csv` for raw data. The
kernel is roughly at parity with SDPA's flash backend on this laptop GPU;
D=128 is often ahead, D=64 non-causal at S=2048 is behind. Run-to-run variance
is about ±10% due to laptop clock behavior.

## Bottleneck hypotheses

1. **Single-stage cp.async.** `cp_async_wait<0>` at the top of every KV
   iteration drains the pipeline. A 2-stage double buffer for K/V should hide
   more of the gmem latency. This is the top next step; it is not implemented
   here.
2. **Causal load imbalance.** Q tiles near the top of the sequence process one
   KV block; tiles near the bottom process all of them. With a fixed CTA per Q
   tile, the tail of the grid is imbalanced. A persistent scheduler or
   reversing/swizzling the block order can help.
3. **Non-matmul softmax cost.** The rescale of the `O` accumulator and the row
   reductions are scalar/vector work that does not use tensor cores. At small
   S and D this is a larger fraction of runtime.
4. **Occupancy vs shared memory.** D=128 with `BlockN=128` needs 16 KB Q +
   32 KB K + 32 KB V = 80 KB per CTA, which limits resident CTAs. D=64 is much
   lighter (8+8+8 KB with BlockN=64).
5. **Register pressure from the fused two-GEMM accumulator.** `kBlockM=64`
   keeps the `O` accumulator and the `P` fragment live across the inner loop.
   A larger `BlockM` would increase reuse but likely spill.

## What I would do next

- Double-buffered K/V with `cp.async.commit_group` / `wait_group<1>` and a
  circular shared-memory buffer.
- Persistent causal scheduler to flatten the tail.
- Tune `(BlockM, BlockN, Warps)` per D with a compile-time dispatch table and
  autotune on the target GPU.
- For Hopper: TMA loads + WGMMA + warp-specialized producer/consumer.
- A fused varlen kernel with device-side `cu_seqlens` and a work queue.
