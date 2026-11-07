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
`cu_seqlens_q/k`. Varlen is **fused**, not a host loop:

1. A small device kernel (`build_tile_map_kernel`) runs one thread per
   sequence. It computes `ntiles = ceil(qlen / kBlockM)` and uses `atomicAdd`
   to reserve a range in two device arrays:
   `tile_to_seq[t] = b`, `tile_to_mblock[t] = m_block`. The actual tile count
   is written to a device scalar.
2. The attention kernel launches with the upper-bound grid
   `(ceil(total_q / kBlockM) + batch, nheads)`. A CTA whose `tile_id` is past
   the actual count exits immediately. This avoids a device-to-host read of
   `cu_seqlens`.
3. Each live CTA reads its `(b, m_block)`, gets
   `q_start/q_end/k_start/k_end` from `cu_seqlens`, builds a per-CTA params
   view, and calls the same `flash_attention_v2_body` as the fixed-length
   kernel.

All sequences' Q tiles are in one grid, so a short sequence's CTA finishing
early lets another sequence's CTA fill the SM. There is one launch and one
final tail instead of one tail per sequence.

### How a Q tile finds its KV tiles

A Q tile never searches for K/V. Its sequence is known from `tile_to_seq`; the
K/V pointers are offset by `k_start` and the K/V tensor shape is
`(klen, D)`. The inner loop is simply

```text
for nbi in 0 .. n_block_max-1:
    gK = local_tile(K, (kBlockN, D), (nbi, _))
    gV = local_tile(V, (kBlockN, D), (nbi, _))
```

so KV tile `nbi` is at `k_start + nbi * kBlockN`; it is bounded by the
sequence's `klen`, never another sequence's K/V. For causal self-attention,
`n_block_max = min(ceil(klen/kBlockN), ceil((m_block+1)*kBlockM/kBlockN))`,
so the Q tile only visits KV tiles up to its diagonal. The per-element mask
then handles the diagonal and the ragged `klen` tail.

Remaining work: the static map uses nondeterministic atomic ordering; for a
more balanced causal tail, sort long tiles first or use a persistent
work-queue scheduler. `block_table` / prefix-cache indexing is not implemented;
that is required for paged prefill, and decode needs a separate paged-KV
kernel.

## 8. Paged-KV decode

Decode uses Q with one token per sequence and K/V from a paged cache:

```text
q:          (B, 1, H, D)
k_cache:    (num_pages, page_size, Hk, D)
block_table:(B, max_pages)      # logical page -> physical page
cache_seqlens:(B,)              # valid KV length
```

The decode kernel launches `grid = (1, B*H)` and reuses
`flash_attention_v2_body` with `Is_causal=false` (the cache contains only past
tokens). The body's K/V tile loader computes the physical address directly:

```text
logical   = nbi * kBlockN
page_idx  = logical / page_block_size
page_off  = logical % page_block_size
physical  = block_table[b, page_idx]
k_tile    = k_cache + physical * page_stride + page_off * row_stride
```

So there is no gather into a contiguous KV buffer. The host API checks
`page_block_size % 64 == 0` so a 64/128-wide KV tile never crosses a page.

The first version uses one CTA per `(batch, head)` and the same 64-row Q tile as
prefill; Q row 0 is the only valid row. It reaches 0.34–0.72× official
`flash_attn_with_kvcache` because official also split-KVs the cache across CTAs
and uses a decode-specific schedule. Split-KV + combine is the next step.

## 9. Code map

| Piece | File |
|---|---|
| CuTe traits / layouts | `include/kernel_traits.h` |
| Mask, predicated copy, GEMM wrappers, online softmax | `include/flash_fwd_kernel_impl.cuh` |
| Launch (grid/smem) | `include/flash_fwd_launch.h` |
| PyTorch API, fixed + varlen + decode | `csrc/flash_fwd_kernel.cu` |
| Independent reference | `standalone/reference.cuh`, `tests/reference.py` |
| Tests | `tests/` |
| Benchmarks | `benchmarks/bench_fwd.py`, `benchmarks/plot.py` |
| Tuning/debug stories | `docs/tuning.md`, `docs/debugging.md` |
| Interview Q&A | `docs/interview-notes.md` |
