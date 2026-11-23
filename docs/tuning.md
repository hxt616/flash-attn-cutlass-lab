# Tuning notes

[English](tuning.md) | [简体中文](tuning_zh.md)

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

## Same-GPU vs official flash-attn 2.8.3

Environment: nano-vllm Python 3.12 venv (`torch 2.6.0+cu124`), official
`flash_attn 2.8.3` editable checkout, RTX 4060 Laptop. Shape follows
Qwen3-0.6B: `B=1, H=16, Hk=8, D=128, bf16`, `causal=True/False`.

| mode | S | causal | ours ms | official ms | ours/official |
|---|---:|---:|---:|---:|---:|
| fixed | 512 | no | 0.0977 | 0.0957 | 0.98 |
| fixed | 1024 | no | 0.3427 | 0.3616 | 1.06 |
| fixed | 2048 | no | 1.1930 | 1.1606 | 0.97 |
| varlen (fused, 512/1024/2048) | 2048 | no | 1.6156 | 1.5529 | 0.96 |
| fixed | 512 | yes | 0.0589 | 0.0545 | 0.93 |
| fixed | 1024 | yes | 0.1864 | 0.1748 | 0.94 |
| fixed | 2048 | yes | 0.6618 | 0.6230 | 0.94 |
| varlen (fused, 512/1024/2048) | 2048 | yes | 0.8941 | 0.8230 | 0.92 |
| varlen (fused, 32 × 16) | 16 | no | 0.0522 | 0.0460 | 0.88 |
| varlen (fused, 32 × 16) | 16 | yes | 0.0521 | 0.0447 | 0.86 |

Raw data: `benchmarks/results/bench_vs_official.csv`; script:
`benchmarks/bench_official.py`.

Interpretation:

- Fixed-length is at parity: 0.93–1.06× official. The implementation is
  competitive with a production FA2 kernel on this sm89 GPU, which is a
  stronger result than the initial "learning project" expectation.
- Fusing varlen closed most of the old 0.78–0.89× gap: it now measures 0.92×
  (causal) / 0.96× (non-causal) for the mixed-length case, and 0.86–0.88× for
  many short sequences. The remaining gap is the causal tail from the static
  tile order, not per-sequence launches anymore.
- The kernel remains forward-only; official also provides paged-KV decode
  (`flash_attn_with_kvcache`) which this project does not implement.

## Paged-KV decode

Decode addresses K/V tiles directly through `block_table` and now uses
split-KV + a log-sum-exp combine kernel. The auto heuristic picks 3 splits for
`B*H=16` and 2 for `B*H=128`; `num_splits` can be overridden.

Environment: same bf16/Qwen3 shape (`H=16, Hk=8, D=128`), page size 256,
`torch 2.6.0+cu124`, official `flash_attn 2.8.3`.

| B | context | auto splits | ours ms | official ms | speed |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 3 | 0.0239 | 0.0181 | 0.76× |
| 1 | 2048 | 3 | 0.0581 | 0.0352 | 0.61× |
| 1 | 4096 | 3 | 0.1058 | 0.0589 | 0.56× |
| 8 | 512 | 2 | 0.1105 | 0.0553 | 0.50× |
| 8 | 2048 | 2 | 0.3765 | 0.2069 | 0.55× |
| 8 | 4096 | 2 | 0.7286 | 0.3946 | 0.54× |

Raw data: `benchmarks/results/bench_decode.csv`; script:
`benchmarks/bench_decode.py`.

For a code-level explanation of prefill/varlen/decode, see
[`design.md`](design.md) / [`design_zh.md`](design_zh.md).

Remaining decode work: a smaller Q tile (1/16 rows) so tensor-core work matches
the one valid query row, a multi-stage `cp.async` pipeline, and possibly sharing
K/V reads across the query heads in a GQA group. Official still has a more
mature decode schedule.

## Bottleneck hypotheses

1. **Single-stage cp.async.** `cp_async_wait<0>` at the top of every KV
   iteration drains the pipeline. A 2-stage double buffer for K/V should hide
   more of the gmem latency. Fixed-length is already at parity, so this is a
   second-order optimization after the varlen scheduler work.
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
6. **Varlen static tile order / causal tail.** Fused varlen removed the
   per-sequence launch and host copy, but the atomic tile map orders tiles
   nondeterministically and does not prioritize long causal tiles. The measured
   remaining gap is 4–7% for mixed lengths and 10–16% for many short sequences.

## What I would do next

- **Persistent varlen scheduler / causal ordering**: the static fused grid is
  in place; sort long causal tiles first or use a persistent work queue to
  close the remaining 4–7% (mixed) and 10–16% (short-sequence) gap. `block_table`
  support is also required for prefix-cache prefill.
- Double-buffered K/V with `cp.async.commit_group` / `wait_group<1>` and a
  circular shared-memory buffer. Fixed-length is already at parity, so expect
  smaller returns here than in a typical "from scratch" kernel.
- Persistent causal scheduler to flatten the tail.
- Tune `(BlockM, BlockN, Warps)` per D with a compile-time dispatch table and
  autotune on the target GPU.
- For Hopper: TMA loads + WGMMA + warp-specialized producer/consumer.
- Add `block_table` / prefix-cache indexing if the kernel is to replace
  nano-vllm's prefill path with prefix caching enabled.
