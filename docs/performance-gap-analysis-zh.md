# 性能差异分析(学习笔记)

这份文档记录当前实现与 PyTorch SDPA / 官方 flash-attn 2.8.3 的性能差异、原因和
后续优化方向,方便项目结束后对照代码学习。代码位置以函数名为主,行号可能随
改动变化。

---

## 0. 先看结论

| 场景 | 当前表现 | 主要原因 |
|---|---|---|
| 固定长度 prefill | 官方 FA2 的 **0.93–1.06×** | 数学和 tiling 与 FA2 基本一致;形状足够大时是 compute-bound,差距主要来自单级 cp.async、固定 block 配置、缺少 warp specialization |
| fused varlen | 官方 FA2 的 **0.92–0.96×**(混合长度) | 已从 host 循环改为单 grid + device tile map;剩余差距来自静态 tile 顺序、causal 尾部、额外 CTA |
| 很多短序列 varlen | 官方 FA2 的 **0.86–0.88×** | 每条序列 q tile 少,GPU 占用不足;上界 grid 会多 launch 一些空 CTA;静态调度不能按工作量排序 |
| paged-KV decode | 官方 `flash_attn_with_kvcache` 的 **0.50–0.76×** | split-KV/combine 已实现;剩余差距是 Q tile 复用 prefill 的 64 行、单级流水线、GQA 重复读 K/V、官方更成熟的 decode 调度 |

一句话:

- **prefill 已经接近官方,说明 FA2 主体和 CuTe 映射是对的;**
- **varlen 的剩余差距是“调度/尾部”,不是数学;**
- **decode 的差距已经从“没有 split-KV”缩小到“Q tile/流水线/GQA 读取”这些第二层问题;**

---

## 1. 测试环境与方法

- GPU:RTX 4060 Laptop 8 GB(sm_89)
- 系统:WSL2 + Ubuntu 22.04,CUDA 12.4
- 我们的扩展:Python 3.10 venv(torch 2.4.1+cu121)
- 官方对比:Python 3.12 venv(torch 2.6.0+cu124,官方 flash-attn 2.8.3 editable)
- 计时:`torch.cuda.Event`,warmup 10,50 次循环,3 次重复取 median
- 形状:
  - prefill:fp16,`B=2, H=8`,D=64/128,S=512/1024/2048
  - 官方对比:bf16,Qwen3-0.6B 形状 `B=1, H=16, Hk=8, D=128`
  - decode:bf16,`H=16, Hk=8, D=128`,page size 256,B=1/8,context 512/2048/4096

### 噪声与视频干扰

Laptop GPU 会热降频,后台浏览器视频会占用 GPU video decode/display、功耗和散热
预算。实测关掉视频前后,decode 的长 context 场景从约 0.45–0.56× 提升到
0.53–0.58×;prefill 与官方的比值基本稳定。

因此:

- **绝对 TFLOPS/延迟**有 ±10% 波动,不要跨环境直接比较;
- **同一环境下的比值**更可信,但也要在 GPU 空闲时测;
- 复现前用 `nvidia-smi` 确认 GPU utilization 接近 0%。

---

## 2. Prefill:为什么能接近官方 FA2

### 2.1 相同点

我们的 `flash_attention_v2_body` 和官方 FA2 做的是同一件事:

- Q tile 固定,外层循环 Q block,内层流式读 K/V;
- `S = Q K^T` 用 `SM80_16x8x16` tensor-core MMA;
- online softmax 在寄存器里维护 `m/l/O`;
- P 不写回 HBM,`O` 累加在寄存器;
- causal 只处理对角线及左侧 KV block;
- `ldmatrix` + swizzled smem + `cp.async`。

当 S 足够大时,prefill 是 compute-bound,tensor-core 和 softmax 的占比决定性能。
我们的 tiling 与官方接近,所以固定长度能到 0.93–1.06×。

### 2.2 还差在哪里

1. **单级 `cp.async` 流水线**
   - 每个 KV block 仍有 `cp_async_wait<0>`,没有 K/V 双缓冲环形队列。
   - 官方 FA2 用多 stage 的 gmem→smem 流水线,能更好地隐藏 HBM 延迟。
   - 代码位置:`flash_attention_v2_body` 里的 `cp_async_fence` / `cp_async_wait`。

2. **固定 block 配置**
   - 我们只有少量 `kBlockM/kBlockN/kNWarps` 组合,官方按 D 和 causal 做了更细的
     dispatch/调优。
   - 当前 D=64 非 causal 用 `kBlockN=128`,其余用 64;这是实测结果,但不是全局最优。

3. **causal 尾部**
   - causal 下不同 Q tile 的 KV block 数从 1 到 `ceil(S/kBlockN)`,CTA 工作量差异大。
   - 静态 grid 的最后一波可能只剩少数长 CTA,SM 空等。
   - 这是 causal D=128 仍比 SDPA 慢一点的重要原因。

4. **没有 warp specialization / TMA / WGMMA**
   - sm89 上官方也主要走 FA2 路线,但仍有更成熟的 producer/consumer 和调度;
   - Hopper 的 TMA/WGMMA/FA3 不在本项目范围,也没有 sm90 硬件验证。

5. **D=64 与 D=128 的差异**
   - D=128 每个 Q/K tile 的计算量更大,tensor-core 利用率更高,所以更容易接近官方;
   - D=64 的 softmax/非 matmul 开销占比更高,D=64 长序列相对更容易落后。

---

## 3. Varlen:从 0.78–0.89× 到 0.92–0.96×

### 3.1 旧实现为什么慢

旧版 `flash_attention_varlen_cutlass` 是 host 循环:

```text
cu_seqlens GPU -> CPU (D2H 同步)
for b in sequences:
    launch 一个 kernel,grid = (ceil(qlen/64), H)
```

问题:

- 每条序列一次 launch,同一 stream 串行,序列间不能互相补位;
- 每个 launch 都有自己的尾部;
- 短序列 grid 很小,GPU 占用不足;
- D2H 同步打断异步执行;
- 序列越多、越短,差距越大。

### 3.2 现在的 fused varlen

- `build_tile_map_kernel`:每条序列一个线程,`atomicAdd` 分配 tile 区间,写
  `tile_to_seq/tile_to_mblock`;
- attention grid = `(max_tiles, H)`,所有序列的 Q tile 在同一个 grid;
- `max_tiles = ceil(total_q/64) + batch` 是上界,超过实际 tile 数的 CTA 立即退出;
- 不再有 per-sequence launch 和 `cu_seqlens` D2H。

这样硬件调度器可以在序列边界之间回填 CTA,只留下一次尾部。

### 3.3 为什么还差 4–8%

- **静态 tile 顺序**:`atomicAdd` 的顺序不确定,也没有按 causal 工作量排序;
- **额外 CTA**:上界 grid 可能包含一些空 CTA,短序列场景比例更高;
- **causal 尾部**:最后仍可能剩下长 CTA;
- **没有 persistent work queue**:无法动态把长任务先做、短任务后做。

短序列 `32 × 16` 只有 0.86–0.88×,主要就是“每条序列 tile 太少 + 上界 grid +
静态顺序”叠加。

---

## 4. Paged-KV decode:为什么是 0.50–0.76×

Decode 的语义:

```text
q:            (B, 1, H, D)
k_cache:      (num_pages, page_size, Hk, D)
block_table:  (B, max_pages)  逻辑页 -> 物理页
cache_seqlens:(B,)            有效 KV 长度
```

当前实现:

- `num_splits == 1` 时用 `flash_attention_decode_kernel`,grid = `(1, B*H)`;
- `num_splits > 1` 时用 `flash_attention_decode_split_kernel`,grid = `(num_splits, B*H)`;
  每个 CTA 处理一段 KV block,写 partial `O/lse`;
- `combine_decode_kernel` 用 log-sum-exp 合并 partials;
- K/V tile 通过 `block_table` 直接算物理地址,不 gather;
- 复用 `flash_attention_v2_body`,Q tile 仍是 64 行,只有第 0 行有效。

自动 split 规则(在 host 上,根据 `B*H`):

```text
target = (B*H < 64) ? 48 : 192
num_splits = clamp(ceil(target / (B*H)), 1, 8)
```

例如 `B=1,H=16` 用 3 split,`B=8,H=16` 用 2 split。

### 4.1 split-KV 已经带来的提升

| B | context | 改前(no split) | 改后(auto split) |
|---:|---:|---:|---:|
| 1 | 512 | 0.72× | 0.76× |
| 1 | 2048 | 0.39× | 0.61× |
| 1 | 4096 | 0.34× | 0.56× |
| 8 | 512 | 0.52× | 0.50× |
| 8 | 2048 | 0.53× | 0.55× |
| 8 | 4096 | 0.58× | 0.54× |

B=1 的长 context 提升最明显,因为原来只有 16 个 CTA;split 后变成
`3 × 16 = 48` 个 CTA,GPU 能填得更满。B=8 本来就有 128 个 CTA,split 的
收益小,combine 还有固定开销,所以 512/4096 出现小幅回退或波动。

### 4.2 剩余差距:并行度仍然不够“满”

- 当前 auto 只到 3/2 split,官方可能用更多 split 或更细的调度;
- split 后带宽约 76–206 GB/s,官方约 116–340 GB/s;
- 这不是 HBM 不够快,而是并发请求、Q tile 浪费和单级流水线共同限制。

### 4.3 Q tile 复用 prefill 的 64 行

- decode 只有 1 个 query,q tile 的 63 行是无效的;
- tensor-core 和 smem 仍有 64 行的开销;
- 注意:官方 splitkv dispatch 也用 `kBlockM=64`,所以**这不是和官方差距的唯一原因**;
  但“decode 专用小 Q tile(1/16)”仍是值得尝试的优化,需要和 split-KV 一起看。

### 4.4 单级 `cp.async` 流水线

- K/V tile 的加载和计算没有多 stage 双缓冲;
- decode 是 memory-bound,隐藏延迟比 prefill 更重要;
- 官方 decode 的流水线和 split-KV 一起作用。

### 4.5 paged 寻址的额外开销

- 每个 KV tile 需要算 `page_idx/page_off/physical` 并读 `block_table`;
- 这部分是几次整数运算 + 一次小的全局读,不是主要瓶颈,但会放大短 kernel 的开销。

### 4.6 GQA 的 KV 重复读取

- `H=16, Hk=8` 时,每个 KV head 被 2 个 Q head 使用;
- 当前每个 Q head 的 CTA 都会独立读同一份 K/V;
- L2 会帮助缓存,但指令/请求仍然重复;
- 进一步可以尝试“一个 CTA 处理同一 KV head 下的多个 Q head”,减少 K/V 读取。

### 4.7 combine 与短 kernel 的固定开销

- split 后多了一个 combine kernel 和 partial workspace 分配;
- decode 单次只有几十微秒,这些固定开销占比高;
- 这也是 B=8 短 context 时 split 收益不明显、甚至略回退的原因。

### 4.8 官方仍有更成熟的 decode 调度

- 官方的 split 数、combine、Q tile 和流水线是一起调的;
- 我们的 auto heuristic 只是简单按 `B*H` 选 split,没有看 context 长度;
- 下一步可以根据 `cache_seqlens` 或 profiler 结果做更细的 split 选择。

---

## 5. 代码位置与学习顺序

| 想学的概念 | 代码位置 |
|---|---|
| online softmax / rescale | `include/flash_fwd_kernel_impl.cuh` 的 `softmax_rescale_o` |
| causal mask 索引 | `apply_mask` |
| ragged tile 谓词 | `copy_mn` / `copy_mn_elementwise` |
| QK/PV 两个 GEMM | `gemm_smem` / `gemm_A_in_regs` |
| 单级 cp.async 流水线 | `flash_attention_v2_body` 里的 `cp_async_fence` / `cp_async_wait` |
| GQA head 映射 | fixed/varlen kernel 里的 `kvh = qh / (q_head/k_head)` |
| fused varlen 调度 | `build_tile_map_kernel` + `flash_attention_v2_varlen_kernel` |
| paged KV 地址计算 | `flash_attention_v2_body` 里的 `kv_tile_ptr` |
| decode 映射/局限 | `flash_attention_decode_kernel` |
| block 配置 dispatch | `run_flash_fwd_` / `run_flash_fwd_varlen_` / `run_flash_fwd_decode_` |
| 测试基准 | `tests/test_decode.py`、`benchmarks/bench_*.py` |

建议学习顺序:

1. 先读 `flash_attention_v2_body` 的 QK → mask → softmax → PV 主循环;
2. 再读 `kernel_traits.h` 的 `TiledMMA`/`SmemLayout`;
3. 然后读 `kv_tile_ptr`,理解连续 vs paged 的区别;
4. 再读 `build_tile_map_kernel` 和 varlen kernel,理解 CTA 分配;
5. 最后读 decode kernel 和 benchmark,理解为什么 decode 需要 split-KV。

---

## 6. 下一步优化及预期

### P0(已完成):decode split-KV + combine

- `flash_attention_decode_split_kernel` 把 KV 长度切成 `num_splits`,每个 CTA
  处理一段,写 partial `O/lse`;
- `combine_decode_kernel` 用 log-sum-exp 合并;
- auto heuristic 按 `B*H` 选 3/2 split,`num_splits` 可覆盖;
- 结果:decode 从 0.34–0.72× 提升到 **0.50–0.76×**,B=1 长 context 提升最明显。

### P0:decode 专用小 Q tile

- 尝试 `kBlockM=1/16` 的 decode body,减少无效 tensor-core/smem 开销;
- 需要和 split-KV 一起评估,因为官方 splitkv 也用 64 行 tile;
- 这是目前 decode 最值得做的下一步。

### P1:双缓冲 K/V 流水线

- `cp.async` 多 stage,隐藏 prefill/decode 的 gmem 延迟;
- 预期 prefill 固定长度和 decode 都有收益,但 prefill 已接近官方,收益可能有限。

### P1:persistent varlen / causal tile 排序

- 用 work queue 或“长 tile 先做”减少 causal 尾部;
- 目标:varlen 短序列 0.86–0.88 → 0.95+。

### P2:GQA KV 复用

- 同一 KV head 下的多个 Q head 共享 K/V 读取,减少重复请求。

### P2:ncu 验证

- 用 `sm__throughput`、`gpu__dram_throughput`、tensor pipe、occupancy、
  bank conflict、warp stall 指标验证上面的假设,不要只凭直觉。

---

## 7. 复现命令与注意事项

```bash
# 1. 关闭浏览器视频,确认 GPU 空闲
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv

# 2. prefill
make bench
make bench-causal

# 3. 官方对比(Python 3.12 环境)
PYTHONPATH=/home/hxt/flash-attn-cutlass-lab \
  /home/hxt/nano-vllm/.venv/bin/python benchmarks/bench_official.py

# 4. decode
PYTHONPATH=/home/hxt/flash-attn-cutlass-lab \
  /home/hxt/nano-vllm/.venv/bin/python benchmarks/bench_decode.py
```

注意:

- 不要在后台播放视频/直播/动画时测;
- 用 median,不要用单次;
- 绝对数字会随温度/功耗变化,优先看 ours/official 比值;
- 跑完 decode 后 GPU 温度会上升,下一轮前等温度回落。

---

## 8. 面试速记

**问:为什么固定长度能接近官方?**

答:FA2 主循环、tiling、online softmax、MMA/shared-memory 布局和官方同构,
prefill 足够大时是 compute-bound;我们的差距主要是单级 cp.async 和 block 配置,
不是数学。

**问:为什么 varlen 还有差距?**

答:旧版是 per-sequence launch + D2H;现在改成 device tile map + 单 grid,已经到
0.92–0.96×。剩余是静态 tile 顺序和 causal 尾部,下一步 persistent/排序。

**问:为什么 decode 还慢?**

答:decode 已经实现 split-KV + combine,从 0.34–0.72× 提升到 0.50–0.76×。
剩余差距是 Q tile 复用 prefill 的 64 行(只有 1 行有效)、单级 cp.async、
GQA 重复读 K/V,以及官方更成熟的 decode 调度;paged 寻址本身不是主要瓶颈。

**问:官方都有了,为什么还做这个?**

答:生产上用官方;这个项目是为了证明能实现、验证、定位和修改这类 kernel。
能讲清楚 online softmax、paged 寻址、split-KV 和性能差距,比会调 API 更有价值。

---

## 9. 已知边界

- 只做 forward;没有 backward/dropout/ALiBi/window。
- sm80/89;没有 Hopper TMA/WGMMA/FA3。
- prefill 暂不支持 `block_table`/prefix cache;decode 支持 `block_table`。
- decode 已支持 split-KV/combine,但还没有专用小 Q tile 和多 stage 流水线。
- varlen 是静态 fused grid,不是 persistent work queue。
- 只支持 contiguous 输入;head dim 必须是 32 的倍数且 <= 256。
