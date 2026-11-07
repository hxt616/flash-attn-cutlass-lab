#!/usr/bin/env python3
"""Same-GPU comparison: our CuTe kernel vs official flash-attn 2.8.3.

Run with the nano-vllm Python 3.12 environment:
  PYTHONPATH=/home/hxt/flash-attn-cutlass-lab \
    /home/hxt/nano-vllm/.venv/bin/python benchmarks/bench_official.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func

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


def flops(B, H, S, D, causal):
    valid = S * (S + 1) / 2 if causal else S * S
    return 4.0 * B * H * valid * D


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def bench_fixed(B, H, Hk, S, D, dtype, causal):
    torch.manual_seed(0)
    q = torch.randn((B, S, H, D), dtype=dtype, device="cuda") * 0.5
    k = torch.randn((B, S, Hk, D), dtype=dtype, device="cuda") * 0.5
    v = torch.randn((B, S, Hk, D), dtype=dtype, device="cuda") * 0.5
    scale = 1.0 / math.sqrt(D)

    out_ours, _ = flash_attn_lab.flash_attn_func(q, k, v, causal, scale)
    out_off = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
    err = max_diff(out_ours, out_off)

    t_ours = time_ms(lambda: flash_attn_lab.flash_attn_func(q, k, v, causal, scale), 10, 50, 3)
    t_off = time_ms(lambda: flash_attn_func(q, k, v, causal=causal, softmax_scale=scale), 10, 50, 3)
    f = flops(B, H, S, D, causal)
    return {
        "mode": "fixed",
        "lengths": str(S),
        "B": B, "S": S, "H": H, "Hk": Hk, "D": D,
        "causal": int(causal), "dtype": str(dtype).split(".")[-1],
        "ours_ms": t_ours, "official_ms": t_off,
        "ours_tflops": f / (t_ours * 1e-3) / 1e12,
        "official_tflops": f / (t_off * 1e-3) / 1e12,
        "max_diff": err,
    }


def bench_varlen(lengths, H, Hk, D, dtype, causal):
    torch.manual_seed(1)
    total = sum(lengths)
    q = torch.randn((total, H, D), dtype=dtype, device="cuda") * 0.5
    k = torch.randn((total, Hk, D), dtype=dtype, device="cuda") * 0.5
    v = torch.randn((total, Hk, D), dtype=dtype, device="cuda") * 0.5
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist()),
                      dtype=torch.int32, device="cuda")
    max_s = max(lengths)
    scale = 1.0 / math.sqrt(D)

    out_ours, _ = flash_attn_lab.flash_attn_varlen_func(q, k, v, cu, cu, causal, scale)
    out_off = flash_attn_varlen_func(
        q, k, v, cu, cu, max_s, max_s, causal=causal, softmax_scale=scale
    )
    err = max_diff(out_ours, out_off)

    t_ours = time_ms(
        lambda: flash_attn_lab.flash_attn_varlen_func(q, k, v, cu, cu, causal, scale),
        10, 50, 3,
    )
    t_off = time_ms(
        lambda: flash_attn_varlen_func(
            q, k, v, cu, cu, max_s, max_s, causal=causal, softmax_scale=scale
        ),
        10, 50, 3,
    )
    f = sum(flops(1, H, s, D, causal) for s in lengths)
    return {
        "mode": "varlen",
        "lengths": (f"{len(lengths)}x{lengths[0]}"
                    if len(set(lengths)) == 1 else "mixed"),
        "B": len(lengths), "S": max_s, "H": H, "Hk": Hk, "D": D,
        "causal": int(causal), "dtype": str(dtype).split(".")[-1],
        "ours_ms": t_ours, "official_ms": t_off,
        "ours_tflops": f / (t_ours * 1e-3) / 1e12,
        "official_tflops": f / (t_off * 1e-3) / 1e12,
        "max_diff": err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--out", default="benchmarks/results/bench_vs_official.csv")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    rows = []
    for causal in (False, True):
        for S in (512, 1024, 2048):
            rows.append(bench_fixed(1, 16, 8, S, 128, dtype, causal))
        rows.append(bench_varlen([512, 1024, 2048], 16, 8, 128, dtype, causal))
        rows.append(bench_varlen([16] * 32, 16, 8, 128, dtype, causal))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'mode':7s} {'S':>5s} {'causal':>6s} {'ours ms':>8s} {'off ms':>8s} "
          f"{'ours TF':>8s} {'off TF':>8s} {'ratio':>6s} {'maxdiff':>9s}")
    for r in rows:
        ratio = r["ours_tflops"] / r["official_tflops"]
        print(f"{r['mode']:7s} {r['S']:5d} {r['causal']:6d} {r['ours_ms']:8.4f} "
              f"{r['official_ms']:8.4f} {r['ours_tflops']:8.2f} "
              f"{r['official_tflops']:8.2f} {ratio:6.2f} {r['max_diff']:9.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
