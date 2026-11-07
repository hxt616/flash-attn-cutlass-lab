#!/usr/bin/env python3
"""Paged-KV decode: our CuTe kernel vs official flash_attn_with_kvcache.

Run with the nano-vllm Python 3.12 environment:
  PYTHONPATH=/home/hxt/flash-attn-cutlass-lab \
    /home/hxt/nano-vllm/.venv/bin/python benchmarks/bench_decode.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch

try:
    from flash_attn import flash_attn_with_kvcache
    HAS_OFFICIAL = True
except Exception:
    flash_attn_with_kvcache = None
    HAS_OFFICIAL = False

import flash_attn_lab


def time_ms(fn, warmup, iters, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def make_inputs(B, H, Hk, D, context, page_size, dtype):
    max_pages = (context + page_size - 1) // page_size
    num_pages = B * max_pages
    k_cache = torch.randn((num_pages, page_size, Hk, D), dtype=dtype, device="cuda") * 0.5
    v_cache = torch.randn((num_pages, page_size, Hk, D), dtype=dtype, device="cuda") * 0.5
    block_table = torch.empty((B, max_pages), dtype=torch.int32, device="cuda")
    for b in range(B):
        block_table[b] = torch.randperm(num_pages, device="cuda")[:max_pages]
    cache_seqlens = torch.full((B,), context, dtype=torch.int32, device="cuda")
    q = torch.randn((B, 1, H, D), dtype=dtype, device="cuda") * 0.5
    return q, k_cache, v_cache, block_table, cache_seqlens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--contexts", default="512,2048,4096")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--headsk", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="benchmarks/results/bench_decode.csv")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    batches = [int(x) for x in args.batches.split(",") if x]
    contexts = [int(x) for x in args.contexts.split(",") if x]
    rows = []

    for B in batches:
        for context in contexts:
            q, k_cache, v_cache, block_table, cache_seqlens = make_inputs(
                B, args.heads, args.headsk, args.dim, context,
                args.page_size, dtype,
            )
            scale = 1.0 / math.sqrt(args.dim)

            out_ours, _ = flash_attn_lab.flash_attn_with_kvcache(
                q, k_cache, v_cache, block_table, cache_seqlens, scale
            )
            if HAS_OFFICIAL:
                out_off = flash_attn_with_kvcache(
                    q, k_cache, v_cache,
                    cache_seqlens=cache_seqlens,
                    block_table=block_table,
                    softmax_scale=scale,
                    causal=True,
                )
                err = (out_ours.float() - out_off.float()).abs().max().item()
            else:
                err = float("nan")

            t_ours = time_ms(
                lambda: flash_attn_lab.flash_attn_with_kvcache(
                    q, k_cache, v_cache, block_table, cache_seqlens, scale
                ),
                args.warmup, args.iters, args.repeats,
            )
            if HAS_OFFICIAL:
                t_off = time_ms(
                    lambda: flash_attn_with_kvcache(
                        q, k_cache, v_cache,
                        cache_seqlens=cache_seqlens,
                        block_table=block_table,
                        softmax_scale=scale,
                        causal=True,
                    ),
                    args.warmup, args.iters, args.repeats,
                )
            else:
                t_off = float("nan")
            bytes_read = B * context * args.headsk * args.dim * 2 * 2
            row = dict(
                B=B, context=context, page_size=args.page_size,
                H=args.heads, Hk=args.headsk, D=args.dim,
                dtype=args.dtype,
                ours_ms=t_ours, official_ms=t_off,
                ours_tok_s=B / (t_ours * 1e-3),
                official_tok_s=B / (t_off * 1e-3),
                ours_gbps=bytes_read / (t_ours * 1e-3) / 1e9,
                official_gbps=bytes_read / (t_off * 1e-3) / 1e9,
                ratio=t_off / t_ours,
                max_diff=err,
            )
            rows.append(row)
            print(f"B={B:3d} ctx={context:5d} | ours {t_ours:.4f} ms "
                  f"({row['ours_tok_s']:.1f} tok/s, {row['ours_gbps']:.1f} GB/s) | "
                  f"official {t_off:.4f} ms ({row['official_tok_s']:.1f} tok/s, "
                  f"{row['official_gbps']:.1f} GB/s) | speed {row['ratio']:.2f}x | "
                  f"diff {err:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
