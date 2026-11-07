# Interview notes

## 3-minute project story

"I implemented and hardened a FlashAttention-2 forward kernel with CUTLASS
CuTe. The starting point was an MIT learning repo whose standalone 'reference'
never actually ran softmax or PV, and whose `exp2` path forgot the `log2(e)`
factor. I rebuilt the reference, moved the kernel into a shared header, added
ragged-tile predication for arbitrary sequence lengths, GQA, bf16, varlen, LSE,
and a stream-safe PyTorch extension. Then I built a 233-case pytest matrix and
a cudaEvent benchmark harness. On an RTX 4060 Laptop the fixed-length kernel is
at 0.93–1.06× official flash-attn 2.8.3. I also fused varlen into a single grid
with a device-side tile map, taking it from 0.78–0.89× to 0.92–0.96×. I then
added a paged-KV decode kernel that reads K/V directly through `block_table`,
plus split-KV + a log-sum-exp combine kernel. It is correct and currently
0.50–0.76× official; the remaining gap is the 64-row Q tile, single-stage
pipeline, and GQA duplicate K/V reads. The next step is a decode-specific small
Q tile and a multi-stage pipeline."

## Q&A

### Why is attention memory-bound, and what does FlashAttention change?

Standard attention materializes `S, P` of size `S x S`, so HBM traffic is
`Theta(S^2 + S D)`. FlashAttention tiles Q and streams K/V, keeping the output
accumulator in registers and the current tiles in SRAM. With SRAM size `M`, HBM
traffic becomes `Theta(S^2 D^2 / M)` (plus `Theta(S D)` for Q/K/V/O), while the
compute stays `Theta(S^2 D)`. The arithmetic intensity rises, so the kernel is
closer to the tensor-core roofline. It is also exact attention, not an
approximation.

### Derive the online softmax update.

For a new block, let `m_new = max(m_old, rowmax(S_blk))`. Then

```text
l_new = exp(m_old - m_new) * l_old + sum(exp(S_blk - m_new))
O_new = exp(m_old - m_new) * O_old + exp(S_blk - m_new) V_blk
```

The `exp(m_old - m_new)` factor rescales the old normalizer and the old output
so both refer to the new max. At the end, `O = O / l`. The kernel uses
`exp2((x - m) * scale * log2e)`; `lse = m * scale + log(l)`.

### Why `exp2` instead of `exp`?

`exp2f` maps directly to the hardware `MUFU.EX2` path and lets the compiler use
FFMA for `x * scale_log2 - m * scale_log2`. It avoids the extra range-reduction
work of `expf` and is the standard FlashAttention trick. The scale must include
`log2(e)`; getting that wrong silently changes the softmax base.

### FA1 vs FA2 loop order?

FA1 outer-loops over KV blocks and inner-loops over Q blocks, so each Q block's
`O` is rescaled/written many times. FA2 outer-loops over Q blocks and keeps the
`O` accumulator in registers for the whole KV loop, reducing non-matmul work
and HBM writes. The kernel here is the FA2 order.

### Explain the CuTe mapping.

`TiledMMA` = `SM80_16x8x16_F32F16F16F32_TN` with 4 warps in M; each warp owns a
16-row slice of the 64-row Q tile. `partition_fragment_A/B/C` give each thread
its register fragment. The GEMM-I accumulator layout `(MMA=4, MMA_M, MMA_N)` is
converted to `(nrow=(2, MMA_M), ncol=(2, MMA_N))` for row-wise softmax, then
`convert_layout_rowcol_Aregs` retiles it to the A-operand layout of GEMM-II so
`P` never leaves registers. This is the key fusion: the C layout of QK^T must
match the A layout of PV, which is why the TN atom is used.

### Why `ldmatrix` and swizzle?

`ldmatrix` loads 8x8 fp16 tiles from shared memory into the exact register
pattern the MMA expects, avoiding per-thread address math and bank conflicts.
The shared-memory layout is XOR-swizzled (`Swizzle<3,3,3>` for 64-wide K) so
the row-major gmem writes and the `ldmatrix` reads do not hit the same banks.
`V` uses the transposed `LDSM` atom because GEMM-II needs `V` in the B-operand
orientation.

### What does the pipeline look like, and what is missing?

Initial: load Q and K[0], `cp.async.commit_group`.
Per iteration: wait Q/K, sync; issue V[i]; compute QK; mask; wait V, sync;
issue K[i+1]; online softmax/rescale; convert P; PV GEMM.
So V overlaps QK and K[i+1] overlaps softmax/PV. Missing: a second K/V buffer
and `wait_group<1>` so the next iteration's QK can start before all copies
finish. That is the top optimization left.

### How does the causal mask index work?

The accumulator fragment is `(nrow=(2, MMA_M), ncol=(2, MMA_N))`. For a thread,
`row = m_block*kBlockM + warp_id*16 + lane/4 + mi*8 + mj*kNWarps*16`, and
`col = n_block*kBlockN + (lane%4)*2 + j + nj*8`. The kernel compares global
`col > row` and writes `-inf`. It also masks `col >= k_seqlen` for the ragged
tail. The `kBlockN`-wide loop bound for causal is
`ceil((m_block+1)*kBlockM / kBlockN)`, which skips KV blocks entirely to the
right of the diagonal. The diagonal block is the only one that needs the
element-wise mask.

### GQA?

`kv_head = q_head / (nheads_q / nheads_k)`. K/V pointers use the KV head, Q/O
use the query head. No repeated K/V materialization. At decode time the KV cache
is read once per KV head and shared by the query-head group, so the kernel is
bandwidth-bound on KV cache reads.

### Varlen?

Packed tensors plus `cu_seqlens` prefix sums. A fused implementation maps CTAs
to `(sequence, q-tile)`; this repository currently loops on the host and
launches per sequence, which is correct but pays launch overhead and a host copy
of `cu_seqlens`. The design doc explains how to make it persistent.

### What was the hardest bug?

The ragged 8-row tile predication. The tile-level predicate looks correct for
S multiple of 64, but the copy atom operates on 8 rows; for S=97 the last tile is
partially valid and writes 7 extra rows. In the standalone buffer that was
silent corruption; in varlen it overwrote the next sequence and eventually
segfaulted. Fixing it required an element-wise fallback that checks the real row
coordinate. I found it by isolating a single-sequence varlen call, then a
multi-sequence call, and using gdb after the segfault moved from the kernel to
the host `cu_seqlens` dereference. See `docs/debugging.md`.

### How do you know it is correct?

- An independent CUDA reference and a PyTorch materialized reference.
- 233 pytest cases over shape/dtype/causal/GQA/varlen/edge cases.
- Official-style error bound: our error <= 2x PyTorch's fp16 error + 1e-3.
- `compute-sanitizer` on ragged shapes.
- LSE checked against `logsumexp`.
- Exact matches for S=64 single-block and causal row 0 sanity checks.

### Performance numbers and bottleneck?

On RTX 4060 Laptop, fp16, B=2,H=8: about 25-29 TFLOPS non-causal, 15-26 TFLOPS
causal depending on D/S, roughly at parity with SDPA. A same-GPU comparison
against official flash-attn 2.8.3, bf16 with the Qwen3-0.6B shape
(B=1,H=16,Hk=8,D=128), gives 0.93–1.06× for fixed-length and 0.92–0.96× for
fused varlen (mixed lengths); many short sequences are 0.86–0.88×. The
remaining varlen gap is the static tile order / causal tail, not the old
per-sequence launch loop. Paged-KV decode now uses split-KV + combine and is
0.50–0.76× official `flash_attn_with_kvcache`. It still reads paged K/V
directly, but reuses the 64-row prefill Q tile and a single-stage pipeline;
official has a more mature decode schedule and split heuristic. Tuning `kBlockN`
gave a measured ~9% win for D=64 non-causal (128 vs 64) but regressed causal and
D=128, so the dispatch is per-case.

### Limitations to state proactively

Forward only; no backward/dropout/ALiBi/window; SM80/89 only; no TMA/WGMMA;
varlen is fused but uses a static tile map; decode has `block_table` and
split-KV/combine but still uses the 64-row prefill Q tile and a single-stage
pipeline; prefill does not yet support `block_table`/prefix cache; contiguous
inputs only; head dim multiple of 32 and <= 256.

## Resume bullets (fill measured numbers from your own machine)

- Built a FlashAttention-2 forward kernel with CUTLASS CuTe on sm_89, adding
  arbitrary-length/ragged predication, causal masking, GQA/MQA, bf16, varlen,
  and LSE; packaged it as a stream-safe PyTorch extension.
- Built a 233-case pytest matrix with an independent reference and an
  official-style error bound, plus a compute-sanitizer script; fixed upstream
  reference, `exp2` scaling, and ragged-tile OOB bugs.
- Built a cudaEvent/CSV/plot benchmark harness against PyTorch SDPA; the kernel
  reaches __ TFLOPS (about __x SDPA) on RTX 4060 Laptop and is within __% of
  SDPA across D=64/128 and S=512-2048.
- Measured `kBlockN` configurations and dispatched per dtype/head-dim/causal
  case; documented the remaining single-stage cp.async bottleneck and the
  double-buffered next step.
- Explicitly attributed the MIT upstream skeleton and separated upstream code
  from my changes in `CONTRIBUTIONS.md`.

## Honest framing

Say: "I started from an MIT-licensed tiny-flash-attention repo, kept the CuTe
skeleton, and rebuilt it into a tested, benchmarked, varlen/GQA-capable
extension. The upstream code is attributed; the correctness fixes, test harness,
benchmarks, and the GQA/varlen/LSE work are mine." Do not claim the original
kernel design as yours; the value is in the engineering, debugging, and
measurement.
