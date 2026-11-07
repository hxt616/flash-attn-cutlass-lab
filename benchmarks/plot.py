#!/usr/bin/env python3
"""Plot the CSV produced by bench_fwd.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="benchmarks/results/bench_fwd_noncausal.csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")

    impls = sorted(df["impl"].unique())
    dims = sorted(df["D"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for impl in impls:
        for D in dims:
            sub = df[(df["impl"] == impl) & (df["D"] == D)].sort_values("S")
            if sub.empty:
                continue
            style = "-o" if impl == "cutlass_lab" else "--s"
            ax.plot(sub["S"], sub["tflops"], style, label=f"{impl} D={D}")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("TFLOPS")
    ax.set_title(Path(args.csv).stem)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
