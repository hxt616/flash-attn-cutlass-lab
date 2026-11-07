# Design notes

## 1. What FlashAttention computes

For one head, with `Q, K, V` of shape `(S, D)`:

```text
S_ij = scale * (Q_i . K_j)
P     = softmax(S, dim=-1)
O     = P V
```

Materializing `S` and `P` costs `O(S^2)` HBM traffic. FlashAttention tiles the
sequence, keeps the current `Q` block and the output accumulator in registers /
SRAM, and streams `K/V` blocks. The softmax normalizer is accumulated online, so
`P` is never written to HBM.

## 2. Online softmax (the math the kernel implements)

For KV block `j`, let `S_j = scale * Q_blk K_j^T` and let `m_j, l_j, O_j` be the
running max, denominator, and unnormalized output. The update is

```text
m_j = max(m_{j-1}, rowmax(S_j))
l_j = exp2((m_{j-1} - m_j) * scale_log2) * l_{j-1}
      + rowsum(exp2(S_j * scale_log2 - m_j * scale_log2))
O_j = exp2((m_{j-1} - m_j) * scale_log2) * O_{j-1}
      + exp2(S_j * scale_log2 - m_j * scale_log2) V_j
```

At the end, `O = O_last / l_last` and

```text
lse = m_last * scale + log(l_last)
```

`scale_log2 = scale * log2(e)`, so the kernel uses `exp2f` and folds the scale
into the max term. `exp2` lets the compiler emit an FFMA instead of a separate
multiply/add, and avoids the `exp` range-reduction overhead.

The first KV block is special (`Is_first=true`): there is no old accumulator to
rescale, so it only computes the max, applies `exp2`, and sums.

## 3. Kernel structure

Grid:

```text
grid.x = ceil(S_q / kBlockM)       # one Q tile per CTA
grid.y = batch * nheads_q          # one head per CTA
block  = kNWarps * 32              # 4 warps = 128 threads
```

Each CTA:

1. Loads its `Q` tile to shared memory.
2. Loads the first `K` tile; starts `cp.async`.
3. For each KV block `nbi`:
   - wait for `Q/K`, sync;
   - issue async load of `V[nbi]` while the QK GEMM runs;
   - compute `S = Q K^T` with the SM80 MMA atom;
   - apply the causal/length mask to the register fragment;
   - wait for `V`, sync;
   - issue async load of `K[nbi+1]`;
   - online-softmax update + rescale the output accumulator;
   - convert `P` fp32 -> fp16/bf16, run `O += P V`.
4. Normalize by `l`, convert to the output dtype, store via shared memory,
   optionally write `lse`.

The current pipeline is single-stage: `V` overlaps the QK GEMM and `K[nbi+1]`
overlaps softmax + PV, but there is no double-buffered `K/V` ring. That is the
main remaining latency-hiding opportunity.

## 4. CuTe objects and why they fit together

- `Flash_fwd_kernel_traits<D, BlockM, BlockN, Warps, T>` in
  `include/kernel_traits.h` chooses:
  - `SM80_16x8x16_F32F16F16F32_TN` for fp16 (or the BF16 atom);
  - `TiledMMA` with `Tile<16 * Warps, 16, 16>`, so 4 warps cover 64 rows;
  - swizzled shared-memory layouts (`Swizzle<3,3,3>` for `kBlockKSmem=64`);
  - `SM75_U32x4_LDSM_N` for smem -> register loads and the transposed
    `SM75_U16x8_LDSM_T` for `V` in the second GEMM.
- `partition_fragment_A/B/C` gives each thread its register fragment.
- `gemm_smem` loads `Q`/`K` from smem with `ldmatrix` and calls `cute::gemm`.
- `gemm_A_in_regs` keeps `P` in registers and only streams `V` from smem.
- The `C` fragment of GEMM-I (`QK^T`) uses the same thread/value layout as the
  `A` fragment of GEMM-II (`P V`); `convert_layout_rowcol_Aregs` retiles it so
  the two GEMMs fuse without a round trip through shared memory.

`convert_layout_acc_rowcol` reshapes the accumulator from
`(MMA=4, MMA_M, MMA_N)` to `(nrow=(2, MMA_M), ncol=(2, MMA_N))`, which is the
layout used by the row-wise softmax reductions and the mask.

## 5. Masking and ragged tiles

`apply_mask` computes the global `(row, col)` of every value in the register
fragment from `lane_id`, warp id, `m_block`, and `n_block`. It sets `-inf` for:

- `col >= k_seqlen` (ragged KV tail),
- `col > row` when causal.

For the `-inf` case the softmax uses `max == -inf -> max_scaled = 0` and
`exp2(-inf) = 0`, so an empty row produces `sum = 0`, `O = 0`, `lse = +inf`.

`copy_mn` predicates whole 8-row copy tiles. That is not enough for a ragged
tail such as `S % 8 != 0`, so `copy_mn_elementwise` falls back to a scalar loop
that checks the actual row coordinate. Without this, the kernel wrote up to 7
rows past the logical end, which silently corrupted packed varlen buffers and
segfaulted on the last sequence.

## 6. GQA

`grid.y` indexes the *query* head. The kernel computes

```text
kv_head = q_head / (nheads_q / nheads_k)
```

and offsets `K/V` by `kv_head * k_head_stride`; `Q/O/lse` use the query head.
This avoids materializing repeated K/V and is the main reason decode-time GQA
is bandwidth-efficient.

## 7. Varlen

The Python/C++ API accepts packed `(total_tokens, nheads, D)` tensors and
`cu_seqlens_q/k`. The current implementation copies the prefix sums to the host
and launches one kernel per sequence with `batch=1`, adjusting pointers and
`q_seqlen/k_seqlen`. This is correct and padding-free, but it is not a fused
persistent varlen kernel. A production version would keep `cu_seqlens` on the
device and map CTAs to `(sequence, q-tile)` pairs, with a work queue for load
balance.

## 8. Code map

| Piece | File |
|---|---|
| CuTe traits / layouts | `include/kernel_traits.h` |
| Mask, predicated copy, GEMM wrappers, online softmax | `include/flash_fwd_kernel_impl.cuh` |
| Launch (grid/smem) | `include/flash_fwd_launch.h` |
| PyTorch API, fixed + varlen | `csrc/flash_fwd_kernel.cu` |
| Independent reference | `standalone/reference.cuh`, `tests/reference.py` |
| Tests | `tests/` |
| Benchmarks | `benchmarks/bench_fwd.py`, `benchmarks/plot.py` |
| Tuning/debug stories | `docs/tuning.md`, `docs/debugging.md` |
| Interview Q&A | `docs/interview-notes.md` |
