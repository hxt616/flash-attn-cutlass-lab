# 设计笔记

[English](design.md) | [简体中文](design_zh.md)

## 1. FlashAttention 在算什么

对单个 head,`Q, K, V` 形状为 `(S, D)`:

```text
S_ij = scale * (Q_i . K_j)
P     = softmax(S, dim=-1)
O     = P V
```

如果物化 `S` 和 `P`,HBM 流量是 `O(S^2)`。FlashAttention 把序列分块,当前 `Q`
块和输出累加器放在寄存器/SRAM,K/V 分块流式读取;softmax 的归一化项在线累加,
所以 `P` 永远不会写回 HBM。

## 2. Online softmax(内核实现的数学)

对第 `j` 个 KV block,令 `S_j = scale * Q_blk K_j^T`,`m_j, l_j, O_j` 分别是
running max、分母和未归一化输出。更新式:

```text
m_j = max(m_{j-1}, rowmax(S_j))
l_j = exp2((m_{j-1} - m_j) * scale_log2) * l_{j-1}
      + rowsum(exp2(S_j * scale_log2 - m_j * scale_log2))
O_j = exp2((m_{j-1} - m_j) * scale_log2) * O_{j-1}
      + exp2(S_j * scale_log2 - m_j * scale_log2) V_j
```

结束时 `O = O_last / l_last`,并且

```text
lse = m_last * scale + log(l_last)
```

其中 `scale_log2 = scale * log2(e)`,所以内核用 `exp2f`,并把 scale 折进 max
项。`exp2` 让编译器生成 FFMA,避免 `exp` 的 range-reduction 开销,也比单独的
乘法/加法更省指令。

第一个 KV block 是特殊的(`Is_first=true`):没有旧的累加器需要 rescale,
只需要计算 max、应用 `exp2` 并求和。

## 3. Kernel 结构

Grid:

```text
grid.x = ceil(S_q / kBlockM)       # 一个 CTA 一个 Q tile
grid.y = batch * nheads_q          # 一个 CTA 一个 head
block  = kNWarps * 32              # 4 warps = 128 threads
```

每个 CTA:

1. 把 Q tile 加载到 shared memory。
2. 加载第一个 K tile,启动 `cp.async`。
3. 对每个 KV block `nbi`:
   - 等待 `Q/K`,sync;
   - 在 QK GEMM 运行时异步加载 `V[nbi]`;
   - 用 SM80 MMA atom 计算 `S = Q K^T`;
   - 对寄存器 fragment 应用 causal/length mask;
   - 等待 V,sync;
   - 异步预取 `K[nbi+1]`;
   - online-softmax 更新 + rescale 输出累加器;
   - P 从 fp32 转 fp16/bf16,执行 `O += P V`。
4. 按 `l` 归一化,转换为输出 dtype,通过 shared memory 写回,可选写 `lse`。

当前是 **单级流水线**:V 与 QK GEMM 重叠,K[nbi+1] 与 softmax/PV 重叠,
但没有 K/V 双缓冲环形队列;这是目前最主要的延迟隐藏机会。

## 4. CuTe 对象以及它们为什么能配合

- `include/kernel_traits.h` 里的
  `Flash_fwd_kernel_traits<D, BlockM, BlockN, Warps, T>` 选择:
  - fp16 用 `SM80_16x8x16_F32F16F16F32_TN`(bf16 用对应 atom);
  - `TiledMMA` 的 `Tile<16 * Warps, 16, 16>`,4 个 warp 覆盖 64 行;
  - swizzled shared-memory layout(`kBlockKSmem=64` 时 `Swizzle<3,3,3>`);
  - smem→register 用 `SM75_U32x4_LDSM_N`,第二个 GEMM 的 V 用转置的
    `SM75_U16x8_LDSM_T`。
- `partition_fragment_A/B/C` 给每个线程分配寄存器 fragment。
- `gemm_smem` 用 `ldmatrix` 从 smem 加载 Q/K,并调用 `cute::gemm`。
- `gemm_A_in_regs` 让 P 留在寄存器,只从 smem 流式加载 V。
- GEMM-I(`QK^T`)的 C fragment 与 GEMM-II(`P V`)的 A fragment 使用相同的
  线程/值布局;`convert_layout_rowcol_Aregs` 重新 tile,使两个 GEMM 融合,
  P 不需要经过 shared memory。

`convert_layout_acc_rowcol` 把累加器从
`(MMA=4, MMA_M, MMA_N)` 转成 `(nrow=(2, MMA_M), ncol=(2, MMA_N))`,这是
row-wise softmax reduction 和 mask 使用的布局。

## 5. Mask 与 ragged tile

`apply_mask` 根据 `lane_id`、warp id、`m_block`、`n_block` 算出寄存器 fragment
中每个元素的全局 `(row, col)`,对以下情况写 `-inf`:

- `col >= k_seqlen`(KV 尾部不完整);
- causal 时 `col > row`。

对于 `-inf`,softmax 使用 `max == -inf -> max_scaled = 0`、
`exp2(-inf) = 0`,因此空行的 `sum = 0`、`O = 0`、`lse = +inf`。

`copy_mn` 的谓词以整个 8 行 copy tile 为单位。对 `S % 8 != 0` 的 ragged
尾部不够,所以 `copy_mn_elementwise` 退化为逐元素循环,检查真实 row 坐标。
否则 kernel 会写到逻辑序列末尾之后最多 7 行,在 packed varlen buffer 中
静默破坏下一条序列,并在最后一条序列上段错误。

## 6. GQA

`grid.y` 索引的是 **query head**。Kernel 计算

```text
kv_head = q_head / (nheads_q / nheads_k)
```

K/V 指针偏移 `kv_head * k_head_stride`;Q/O/lse 使用 query head。这样不需要
物化重复的 K/V,也是 decode 时 GQA 能节省带宽的主要原因。

## 7. Varlen

Python/C++ API 接收 packed `(total_tokens, nheads, D)` 和 `cu_seqlens_q/k`。
Varlen 是 **fused** 的,不是 host 循环:

1. 小 device kernel(`build_tile_map_kernel`)每条序列一个线程,计算
   `ntiles = ceil(qlen / kBlockM)`,用 `atomicAdd` 在两个 device 数组里预留
   区间:`tile_to_seq[t] = b`、`tile_to_mblock[t] = m_block`;实际 tile 数
   写入 device scalar。
2. Attention kernel 用上界 grid
   `(ceil(total_q / kBlockM) + batch, nheads)` 启动;`tile_id` 超过实际数量的
   CTA 立即退出。这样不需要把 `cu_seqlens` 读回 host。
3. 每个有效 CTA 读自己的 `(b, m_block)`,从 `cu_seqlens` 取
   `q_start/q_end/k_start/k_end`,构造 per-CTA params,调用与 fixed-length
   kernel 相同的 `flash_attention_v2_body`。

所有序列的 Q tile 在同一个 grid 里,短序列的 CTA 结束后,其他序列的 CTA 可以
补上空闲 SM。只有一次 launch、一次最终尾部,而不是每条序列一次尾部。

### 一个 Q tile 如何找到 KV tile

Q tile 不做搜索。它通过 `tile_to_seq` 知道自己的序列;K/V 指针偏移
`k_start`,K/V tensor 形状是 `(klen, D)`。内层循环是:

```text
for nbi in 0 .. n_block_max-1:
    gK = local_tile(K, (kBlockN, D), (nbi, _))
    gV = local_tile(V, (kBlockN, D), (nbi, _))
```

所以第 `nbi` 个 KV tile 的起点是 `k_start + nbi * kBlockN`,由该序列的
`klen` 限制,不会访问其他序列。causal self-attention 下:

```text
n_block_max = min(ceil(klen/kBlockN),
                  ceil((m_block+1)*kBlockM/kBlockN))
```

Q tile 只访问到对角线附近;逐元素 mask 负责对角线 block 和 ragged `klen`
尾部。

剩余工作:静态 tile map 使用不确定的 atomic 顺序,没有按 causal 工作量排序;
可以按长 tile 优先或使用 persistent work queue。`block_table`/prefix-cache
索引尚未实现,这是 paged prefill 需要的;decode 需要单独的 paged-KV kernel。

## 8. Paged-KV decode

Decode 每个序列只有 1 个 token,K/V 来自 paged cache:

```text
q:            (B, 1, H, D)
k_cache:      (num_pages, page_size, Hk, D)
block_table:  (B, max_pages)      # 逻辑页 -> 物理页
cache_seqlens:(B,)                # 有效 KV 长度
```

Decode 从 `flash_attention_decode_kernel` 开始,`grid = (1, B*H)`,复用
`flash_attention_v2_body`,`Is_causal=false`(cache 里只有过去的 token)。
K/V tile loader 直接计算物理地址:

```text
logical   = nbi * kBlockN
page_idx  = logical / page_block_size
page_off  = logical % page_block_size
physical  = block_table[b, page_idx]
k_tile    = k_cache + physical * page_stride + page_off * row_stride
```

因此不需要把 paged KV gather 成连续 buffer。Host API 检查
`page_block_size % 64 == 0`,保证 64/128 宽的 KV tile 不会跨页。

当 `num_splits > 1`(默认 auto heuristic)时,kernel 用
`grid = (num_splits, B*H)`。每个 CTA 处理一段连续 KV block
`[n_block_min, n_block_max)`,写 partial `O`(按该 split 的 `l` 归一化)和
partial `lse` 到 workspace,再由 combine kernel 合并:

```text
m = max(lse_s)
l = sum(exp(lse_s - m))
O = sum(exp(lse_s - m) * O_s) / l
lse = m + log(l)
```

Auto heuristic 在 `B*H` 较小时使用更多 split。Decode 目前达到官方
`flash_attn_with_kvcache` 的 0.50–0.76×;仍然复用 64 行 prefill Q tile
(只有 1 行有效),使用单级 `cp.async`,并且 GQA 组内的 query head 没有共享
K/V 读取。这些是下一步 decode 优化方向。

## 9. 代码地图

| 部分 | 文件 |
|---|---|
| CuTe traits / layouts | `include/kernel_traits.h` |
| Mask、predicated copy、GEMM wrapper、online softmax | `include/flash_fwd_kernel_impl.cuh` |
| Launch(grid/smem) | `include/flash_fwd_launch.h` |
| PyTorch API,fixed + varlen + decode | `csrc/flash_fwd_kernel.cu` |
| 独立 reference | `standalone/reference.cuh`、`tests/reference.py` |
| 测试 | `tests/` |
| Benchmark | `benchmarks/bench_fwd.py`、`benchmarks/plot.py` |
| 调优记录 | `docs/tuning.md`、`docs/tuning_zh.md` |
