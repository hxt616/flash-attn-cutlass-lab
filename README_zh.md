# flash-attn-cutlass-lab

[English](README.md) | [简体中文](README_zh.md)

一个 **AI 辅助完成**的 FlashAttention-2 前向/解码 kernel 重构项目,使用
[CUTLASS CuTe](https://github.com/NVIDIA/cutlass) 实现,并配套可复现的正确性
与性能测试。项目基于优秀的
[tiny-flash-attention](https://github.com/66RING/tiny-flash-attention) 学习仓库:
上游提供 CuTe 骨架和教程,本仓库增加了正确性修复、任意长度/varlen、GQA、
paged-KV decode、PyTorch 扩展、pytest 矩阵、benchmark 和文档。

> 署名:见 [`NOTICE`](NOTICE)。上游是 MIT;CUTLASS 和 FlashAttention 的算法/
> reference 模式是 BSD-3。来自上游的文件保留其版权声明。本项目不声称对上游骨架
> 拥有原创著作权。

## 开发说明

本项目在开发过程中使用了 AI 编程助手。AI 助手参与了 kernel 实现、host 包装、
测试、benchmark 和文档的部分工作。作者负责项目方向和技术范围,并对最终验证和
报告结果负责。仓库已在下方描述的环境中完成构建、测试和 benchmark;所有数字都可
由仓库内脚本复现。

这份声明是有意为之:本项目是基于 MIT 上游骨架的学习与工程实践,不声称每一行
代码都是完全手工从零编写。

## 特性

- 使用 CuTe 实现 FlashAttention-2 前向(`SM80_16x8x16` MMA、`ldmatrix`、
  `cp.async`)。
- 支持 `fp16`/`bf16`;head dim 支持 32、64、128。
- 支持任意序列长度,包括不是 64 行 tile 倍数的情况(用逐元素谓词处理 ragged
  8 行 copy tile)。
- 支持 causal / non-causal mask;causal 路径会跳过完全被 mask 的 KV block。
- 支持 GQA / MQA(`nheads_k < nheads`),KV head 映射正确。
- 支持无 padding 的 varlen(`cu_seqlens`)。Varlen 使用单个 fused kernel:
  device 端小 kernel 构建 `tile -> (sequence, m_block)` 映射,所有序列的 Q tile
  共享一个 grid;没有 per-sequence launch,也没有 `cu_seqlens` 的 device-to-host
  拷贝。
- Paged-KV decode:`flash_attn_with_kvcache` 风格 API,支持 `block_table`、
  `cache_seqlens`、GQA 和 page size 64/128/256;K/V tile 直接从 paged cache
  读取(不做 gather),并使用 split-KV + log-sum-exp combine kernel。
- PyTorch 扩展在当前 CUDA stream 上启动,返回 `(out, lse)`,并校验 dtype/
  shape/contiguity。
- 233 个 pytest 用例(shape、causal、GQA、varlen、decode/paged KV、bf16、
  边界情况、API 契约,以及官方风格的 error-vs-PyTorch 准则)。
- 可复现的 benchmark harness,输出 CSV/PNG,并与 SDPA/官方 flash-attn 对比。

## 目录结构

```text
include/
  flash_fwd_kernel_impl.cuh   # 与框架无关的 CuTe kernel
  flash_fwd_launch.h          # grid/smem launch
  flash_params.h              # 共享 launch 参数
  kernel_traits.h, utils.h    # 来自上游的 CuTe 辅助
csrc/
  flash_fwd_kernel.cu         # PyTorch host 包装(fixed + varlen + decode)
  flash_api.cpp               # pybind11 模块
  static_switch.h
standalone/
  main.cu                     # 对拍 naive + cudaEvent benchmark
  reference.cuh               # 简单、易审计的 reference kernel
tests/                        # pytest 矩阵 + fp64/fp32 reference(含 decode)
benchmarks/                   # bench_fwd.py、bench_official.py、bench_decode.py、results/
docs/                         # 公开的设计/调优文档(中英文)
```

## 快速开始

```bash
# 1. 环境(uv 很快;普通 venv + pip 也可以)
bash scripts/setup_env.sh

# 2. standalone 二进制(不依赖 PyTorch)
make build

# 3. PyTorch 扩展(in-place)
make ext

# 4. 测试
make test

# 5. Benchmark(先 non-causal,再 causal)
make bench
make bench-causal
# 也可以 editable 安装:pip install -e . --no-build-isolation
```

Standalone 快速检查:

```bash
./build/flash_attn_lab --B 1 --H 2 --Hk 2 --S 97 --D 64 --causal
./build/flash_attn_lab --B 2 --H 8 --Hk 8 --S 2048 --D 64 --bn 128 --iters 200
./build/flash_attn_lab --B 1 --H 2 --Hk 2 --S 129 --D 128 --causal --no-bench
```

Python API(也可以通过 `flash_attn_lab.*` 使用):

```python
from flash_attn_lab import (
    flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache,
)

out, lse = flash_attn_func(q, k, v, causal=True)   # q: (B,S,H,D)
out, lse = flash_attn_varlen_func(q, k, v, cu_q, cu_k, causal=True)

# paged-KV decode:q (B,1,H,D),cache (num_pages,page_size,Hk,D)
out, lse = flash_attn_with_kvcache(
    q, k_cache, v_cache, block_table, cache_seqlens
)
```

`q` 形状是 `(batch, seqlen, nheads, headdim)`;`k/v` 是
`(batch, seqlen_k, nheads_k, headdim)`;输入必须是 contiguous 的 `fp16`/`bf16`。
`lse` 是 `float32`,形状 `(batch, nheads, seqlen_q)`。Varlen 输入是
`(total_tokens, nheads, headdim)`,`lse` 是 `(total_tokens, nheads)`。Decode 的
`block_table` 是 `(batch, max_pages)` int32,`cache_seqlens` 是 `(batch,)` int32。

## 正确性

Reference 包括两部分:独立、易审计的 CUDA kernel(`standalone/reference.cuh`)
和 PyTorch 物化 reference(`tests/reference.py`)。pytest 矩阵覆盖:

- `dtype ∈ {fp16, bf16}`,`D ∈ {32, 64, 128}`
- `S ∈ {1, 2, 17, 63, 64, 65, 97, 127, 128, 129, 200, 257, 512, 1024}`
- causal / non-causal、自定义 scale
- GQA 比例 `Hk ∈ {H, H/2, H/4, 1}`
- varlen 的零长度和 ragged 序列
- API 拒绝非 contiguous / 错误 dtype / 非法 head dim
- 官方风格检查:`our_error <= 2 * pytorch_error + 1e-3`

```text
233 passed
```

运行 `make sanitize`,会在小 ragged shape 上执行 `compute-sanitizer`
(`memcheck`、`racecheck`、`initcheck`、`synccheck`)。

## 性能(RTX 4060 Laptop 8 GB,WSL2,CUDA 12.4)

方法:fp16,`B=2, H=8`,warmup 10 次,50 次计时迭代,重复 3 次取 median;
PyTorch SDPA 作为基线。Laptop GPU 会热降频,因此 ±10% 视为运行间噪声。原始数据:
[`benchmarks/results/`](benchmarks/results/)。

**Non-causal(TFLOPS)**

| D | S | 本 kernel | SDPA |
|---:|---:|---:|---:|
| 64 | 512 | 19.6 | 21.8 |
| 64 | 1024 | 22.8 | 23.7 |
| 64 | 2048 | 23.5 | 29.7 |
| 128 | 512 | 24.6 | 25.1 |
| 128 | 1024 | 28.1 | 26.4 |
| 128 | 2048 | 28.7 | 29.1 |

**Causal(TFLOPS)**

| D | S | 本 kernel | SDPA |
|---:|---:|---:|---:|
| 64 | 512 | 14.8 | 14.1 |
| 64 | 1024 | 19.4 | 18.2 |
| 64 | 2048 | 22.4 | 22.1 |
| 128 | 512 | 18.4 | 19.7 |
| 128 | 1024 | 23.3 | 24.5 |
| 128 | 2048 | 26.1 | 27.4 |

本 kernel 与 SDPA 的 flash backend 在这块 Laptop GPU 上基本同级;D=128 经常更好,
因为额外的 K 维计算摊薄了 softmax 的非 matmul 开销。`kBlockN` 实验和瓶颈分析见
[`docs/tuning.md`](docs/tuning.md) / [`docs/tuning_zh.md`](docs/tuning_zh.md);
prefill/varlen/decode 的设计见 [`docs/design.md`](docs/design.md) /
[`docs/design_zh.md`](docs/design_zh.md)。详细的个人学习笔记(调试、面试、
性能差距)不放在公开仓库中。

**对比官方 flash-attn 2.8.3**(同机、bf16、Qwen3-0.6B 形状:
`B=1, H=16, Hk=8, D=128`;ratio = 本 kernel TFLOPS / 官方 TFLOPS)

| 模式 | S | causal | ratio |
|---|---:|---:|---:|
| fixed | 512 | 否 | 0.98 |
| fixed | 1024 | 否 | 1.06 |
| fixed | 2048 | 否 | 0.97 |
| varlen(fused) | 512/1024/2048 | 否 | 0.96 |
| fixed | 512 | 是 | 0.93 |
| fixed | 1024 | 是 | 0.94 |
| fixed | 2048 | 是 | 0.94 |
| varlen(fused) | 512/1024/2048 | 是 | 0.92 |

与官方的最大输出差异约 `1e-3`(bf16)。固定长度和 fused varlen 目前基本持平
(0.92–1.06×)。很多短序列(`32 × 16`)是 0.88(非 causal)/ 0.86(causal);
剩余差距来自 causal 尾部和静态 tile 顺序。官方基线在 nano-vllm 的 Python 3.12
环境(`torch 2.6.0+cu124`,官方 `flash_attn 2.8.3`)中运行。原始数据:
`benchmarks/results/bench_vs_official.csv`。

**Paged-KV decode**(bf16,`H=16, Hk=8, D=128`,page size 256,split-KV 自动)

| B | context | auto splits | 本 kernel ms | 官方 ms | speed vs official |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 3 | 0.0239 | 0.0181 | 0.76× |
| 1 | 2048 | 3 | 0.0581 | 0.0352 | 0.61× |
| 1 | 4096 | 3 | 0.1058 | 0.0589 | 0.56× |
| 8 | 512 | 2 | 0.1105 | 0.0553 | 0.50× |
| 8 | 2048 | 2 | 0.3765 | 0.2069 | 0.55× |
| 8 | 4096 | 2 | 0.7286 | 0.3946 | 0.54× |

正确性与官方差异约 `1e-3`。Split-KV + combine 已实现:每个 CTA 处理一段连续
KV range,写 partial `O/lse`,再由 combine kernel 合并。Decode 现在达到官方的
0.50–0.76×;剩余差距来自 64 行 Q tile(只有 1 行有效)、单级 `cp.async`、GQA
重复读 K/V,以及官方更成熟的 decode 调度。原始数据:
`benchmarks/results/bench_decode.csv`。

> 口径说明:prefill 的 ratio 是**吞吐比**(本 kernel TFLOPS / 官方 TFLOPS),
> 等价于本 kernel 延迟 / 官方延迟 = 0.94–1.08×;decode 的 speed 是
> **官方时间 / 本 kernel 时间**,等价于本 kernel 延迟 / 官方延迟 = 1.32–2.0×。

## 相对上游的改动

见 [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md)。要点:

- 把 kernel 重构成与框架无关的 header,并增加 stream-safe、带输入校验的
  PyTorch 扩展。
- 修复 standalone reference(之前没有执行 softmax/PV)和 `exp2` scale(缺少
  `log2(e)`)。
- 增加 ragged-tile 谓词、任意序列长度、GQA、bf16、fused varlen(device 端
  tile map)和 LSE。
- 增加 pytest 矩阵、sanitizer 脚本、benchmark harness 和文档。
- 按 dtype/head dim/causal 实测调优 `kBlockN`。

## 已知限制

- 只做 forward;没有 backward、dropout、ALiBi/RoPE/window。
- 只支持 SM80/SM89 路径;没有 Hopper TMA/wgmma 或 warp specialization。
- Varlen 使用静态 fused grid + device 端 atomic tile map;persistent work-queue
  调度还能进一步减少 causal 尾部。
- Decode 已使用 split-KV + combine(auto heuristic,`num_splits` 可覆盖),
  但仍复用 prefill 的 64 行 Q tile 和单级流水线;这是 decode 只有 0.50–0.76×
  的原因。
- Prefill 暂不支持 `block_table`/prefix-cache 索引,只支持无 prefix-cache 的
  prefill;decode 支持 `block_table`。
- 单级 `cp.async` 流水线(K/V 加载和 QK GEMM 有部分重叠,但还没有多 stage
  ring buffer)。
- 只支持 contiguous 输入;head dim 必须是 32 的倍数且 ≤ 256。
- 官方对比使用单独的 Python 3.12 环境,因为项目开发环境是 Python 3.10;
  kernel 源码相同。见 `benchmarks/bench_official.py`。

## License

本仓库使用 MIT;保留上游 MIT 版权。CUTLASS 和 FlashAttention 派生部分使用
BSD-3。见 [`LICENSE`](LICENSE) 和 [`NOTICE`](NOTICE)。
