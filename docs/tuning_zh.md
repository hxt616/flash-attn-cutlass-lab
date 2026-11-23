# 调优记录

[English](tuning.md) | [简体中文](tuning_zh.md)

## 方法

- 用 standalone 二进制隔离 kernel(不包含 Python 分配):
  `./build/flash_attn_lab --B 2 --H 8 --Hk 8 --S ... --D ... --bn ...`。
- 用 `cudaEvent` 包住 warmup + 100–300 次迭代,报告 ms/iter 和 TFLOPS;
  4060 Laptop GPU 会热降频,可疑结果要重跑。
- 每改一个 block 配置都先确认正确性(`--no-bench` 会用独立 reference)。
- Python benchmark 包含 `torch::empty` 分配和 LSE 写回,适合和 SDPA 做端到端
  对比;要归因到 kernel 本身时用 standalone。

## Block-N 实验(standalone,fp16,B=2,H=8)

| D | S | causal | kBlockN | ms/iter | TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 64 | 2048 | 否 | 64 | 0.6758 | 25.42 |
| 64 | 2048 | 否 | 128 | 0.6198 | 27.72 |
| 64 | 2048 | 是 | 64 | 0.3359 | 25.59 |
| 64 | 2048 | 是 | 128 | 0.3664 | 23.45 |
| 32 | 2048 | 否 | 64 | 0.3108 | 27.63 |
| 32 | 2048 | 否 | 128 | 0.3701 | 23.21 |
| 128 | 2048 | 否 | 64 | 1.1733 | 29.28 |
| 128 | 2048 | 否 | 128 | 1.4089 | 24.39 |
| 128 | 1024 | 是 | 64 | 0.2025 | 21.23 |
| 128 | 1024 | 是 | 128 | 0.2233 | 19.25 |

结论:只在 `D=64` 非 causal 时使用 `kBlockN=128`,其余组合用 64。Dispatch 在
`csrc/flash_fwd_kernel.cu` 的 `run_flash_fwd_`。

原因:

- 更大的 `BlockN` 把循环次数和 softmax rescale 次数减半,但增加 shared memory
  (K/V tile)和寄存器压力。D=64 非 causal 收益为正;D=32 的小 K tile 已经能放
  更多 CTA;D=128 的 128 宽 K/V tile 会把 occupancy 压得太低。
- Causal kernel 每个 Q tile 只访问少量 KV block,尾部 block 还有部分被 mask,
  更大的 N tile 增加工作量但复用不足。

## 与 PyTorch SDPA 的端到端对比(fp16,B=2,H=8)

原始数据见 `benchmarks/results/bench_fwd_{noncausal,causal}.csv`。在这块 Laptop
GPU 上,kernel 与 SDPA 的 flash backend 基本同级;D=128 经常更好,D=64 非 causal
在 S=2048 时落后。Laptop 时钟行为导致约 ±10% 的运行间波动。

## 与官方 flash-attn 2.8.3 同机对比

环境:nano-vllm Python 3.12 venv(`torch 2.6.0+cu124`)、官方
`flash_attn 2.8.3` editable、RTX 4060 Laptop。形状按 Qwen3-0.6B:
`B=1, H=16, Hk=8, D=128, bf16`,`causal=True/False`。

| 模式 | S | causal | ours ms | official ms | ours/official |
|---|---:|---:|---:|---:|---:|
| fixed | 512 | 否 | 0.0977 | 0.0957 | 0.98 |
| fixed | 1024 | 否 | 0.3427 | 0.3616 | 1.06 |
| fixed | 2048 | 否 | 1.1930 | 1.1606 | 0.97 |
| varlen(fused,512/1024/2048) | 2048 | 否 | 1.6156 | 1.5529 | 0.96 |
| fixed | 512 | 是 | 0.0589 | 0.0545 | 0.93 |
| fixed | 1024 | 是 | 0.1864 | 0.1748 | 0.94 |
| fixed | 2048 | 是 | 0.6618 | 0.6230 | 0.94 |
| varlen(fused,512/1024/2048) | 2048 | 是 | 0.8941 | 0.8230 | 0.92 |
| varlen(fused,32 × 16) | 16 | 否 | 0.0522 | 0.0460 | 0.88 |
| varlen(fused,32 × 16) | 16 | 是 | 0.0521 | 0.0447 | 0.86 |

原始数据:`benchmarks/results/bench_vs_official.csv`;脚本:
`benchmarks/bench_official.py`。

解释:

- 固定长度基本持平:官方的 0.93–1.06×。在这个 sm89 GPU 上,实现已经能和
  生产级 FA2 kernel 竞争,比“学习项目”的预期更强。
- Fused varlen 把原来的 0.78–0.89× 差距大部分补上:混合长度现在是 0.92×
  (causal)/0.96×(非 causal),很多短序列是 0.86–0.88×。剩余差距来自静态
  tile 顺序造成的 causal 尾部,不再是 per-sequence launch。
- Kernel 仍然只做 forward;官方还提供 paged-KV decode
  (`flash_attn_with_kvcache`),本项目已实现其中的 split-KV/combine 版本。

## Paged-KV decode

Decode 通过 `block_table` 直接寻址 K/V tile,并使用 split-KV + log-sum-exp
combine kernel。Auto heuristic 在 `B*H=16` 时选 3 个 split,`B*H=128` 时选 2 个;
可以通过 `num_splits` 覆盖。

环境:同样的 bf16/Qwen3 形状(`H=16, Hk=8, D=128`),page size 256,
`torch 2.6.0+cu124`,官方 `flash_attn 2.8.3`。

| B | context | auto splits | ours ms | official ms | speed |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 3 | 0.0239 | 0.0181 | 0.76× |
| 1 | 2048 | 3 | 0.0581 | 0.0352 | 0.61× |
| 1 | 4096 | 3 | 0.1058 | 0.0589 | 0.56× |
| 8 | 512 | 2 | 0.1105 | 0.0553 | 0.50× |
| 8 | 2048 | 2 | 0.3765 | 0.2069 | 0.55× |
| 8 | 4096 | 2 | 0.7286 | 0.3946 | 0.54× |

原始数据:`benchmarks/results/bench_decode.csv`;脚本:
`benchmarks/bench_decode.py`。

剩余的 decode 工作:更小的 Q tile(1/16 行),让 tensor-core 工作和唯一有效的
query 行匹配;多 stage `cp.async` 流水线;以及在 GQA 组内共享 K/V 读取。
官方仍然有更成熟的 decode 调度。

## 瓶颈假设

1. **单级 cp.async。** 每个 KV 迭代开始的 `cp_async_wait<0>` 会排空流水线;
   K/V 双缓冲可以隐藏更多 gmem 延迟。固定长度已经持平,所以这是 varlen
   scheduler 之后第二优先的优化。
2. **Causal load imbalance。** 序列顶部的 Q tile 只处理 1 个 KV block,底部的
   处理全部;固定 CTA/Q tile 会让 grid 尾部不均衡。Persistent scheduler 或
   反转/swizzle block 顺序可以缓解。
3. **非 matmul softmax 开销。** O 累加器的 rescale 和 row reduction 是
   scalar/vector 工作,不使用 tensor core;S 和 D 较小时占比更高。
4. **Occupancy vs shared memory。** D=128、`BlockN=128` 需要 16KB Q + 32KB K +
   32KB V = 80KB/CTA,限制常驻 CTA 数;D=64 轻得多(BlockN=64 时 8+8+8KB)。
5. **融合双 GEMM 的寄存器压力。** `kBlockM=64` 让 O 累加器和 P fragment 在
   内层循环中一直存活;更大的 BlockM 会增加复用但可能 spill。
6. **Varlen 静态 tile 顺序/causal 尾部。** Fused varlen 去掉了 per-sequence
   launch 和 host copy,但 atomic tile map 顺序不确定,也没有优先处理长 causal
   tile。混合长度还差 4–7%,很多短序列还差 10–16%。

## 下一步

- **Persistent varlen scheduler / causal 排序**:按长 causal tile 优先,或用
  persistent work queue,缩小剩余 4–7%(混合)和 10–16%(短序列)差距。
  prefix-cache prefill 还需要 `block_table` 支持。
- **K/V 双缓冲**:`cp.async.commit_group` / `wait_group<1>` + 环形 shared
  memory buffer。固定长度已经持平,收益可能比“从零写”的 kernel 小。
- **Persistent causal scheduler**:压平尾部。
- 按 D 调优 `(BlockM, BlockN, Warps)`,用编译期 dispatch + 目标 GPU autotune。
- Hopper:TMA load + WGMMA + warp-specialized producer/consumer。
- 如果要用在 nano-vllm 的 prefix-cache prefill,需要加
  `block_table`/prefix-cache 索引。
