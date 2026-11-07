#!/usr/bin/env python3
"""Reproducible forward-pass benchmark for the CuTe FlashAttention-2 kernel.

Timing uses torch.cuda.Event around a warmup + N-iteration loop, repeated a few
times; the reported number is the median of the per-repeat averages.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flash_attn_lab  # noqa: E402


def attention_flops(B, H, S, D, causal):
    if causal:
        valid = S * (S + 1) / 2
        return 4.0 * B * H * valid * D
    return 4.0 * B * H * S * S * D


def time_ms(fn: Callable[[], object], warmup: int, iters: int, repeats: int = 5):
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
    return statistics.median(samples), min(samples), max(samples)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, hk, s, d = x.shape
    return x[:, :, None, :, :].expand(b, hk, n_rep, s, d).reshape(b, hk * n_rep, s, d)


def sdpa_fn(q, k, v, causal, scale):
    n_rep = q.shape[2] // k.shape[2]
    if n_rep > 1:
        k = repeat_kv(k.transpose(1, 2), n_rep).transpose(1, 2)
        v = repeat_kv(v.transpose(1, 2), n_rep).transpose(1, 2)
    return lambda: F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )


def official_fn(q, k, v, causal, scale):
    try:
        from flash_attn import flash_attn_func
    except Exception:
        return None
    return lambda: flash_attn_func(
        q, k, v, causal=causal, softmax_scale=scale
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--dims", default="64,128")
    parser.add_argument("--seqlens", default="512,1024,2048")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--headsk", type=int, default=8)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--baselines", default="sdpa,official")
    parser.add_argument("--out", default="benchmarks/results/bench_fwd.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    dims = [int(x) for x in args.dims.split(",") if x]
    seqlens = [int(x) for x in args.seqlens.split(",") if x]
    baselines = [x for x in args.baselines.split(",") if x]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for D in dims:
        for S in seqlens:
            B, H, Hk = args.batch, args.heads, args.headsk
            q = torch.randn((B, S, H, D), dtype=dtype, device="cuda") * 0.5
            k = torch.randn((B, S, Hk, D), dtype=dtype, device="cuda") * 0.5
            v = torch.randn((B, S, Hk, D), dtype=dtype, device="cuda") * 0.5
            scale = 1.0 / math.sqrt(D)
            flops = attention_flops(B, H, S, D, args.causal)

            impls = {
                "cutlass_lab": lambda: flash_attn_lab.flash_attn_func(
                    q, k, v, args.causal, scale
                )
            }
            if "sdpa" in baselines:
                impls["sdpa"] = sdpa_fn(q, k, v, args.causal, scale)
            if "official" in baselines:
                fn = official_fn(q, k, v, args.causal, scale)
                if fn is not None:
                    impls["official"] = fn

            for name, fn in impls.items():
                median, lo, hi = time_ms(
                    fn, args.warmup, args.iters, args.repeats
                )
                tflops = flops / (median * 1e-3) / 1e12
                row = dict(
                    impl=name,
                    dtype=args.dtype,
                    B=B,
                    H=H,
                    Hk=Hk,
                    S=S,
                    D=D,
                    causal=int(args.causal),
                    ms_median=median,
                    ms_min=lo,
                    ms_max=hi,
                    tflops=tflops,
                )
                rows.append(row)
                print(
                    f"{name:12s} dtype={args.dtype} D={D:3d} S={S:5d} "
                    f"causal={int(args.causal)} | {median:.4f} ms | "
                    f"{tflops:6.2f} TFLOPS"
                )

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
