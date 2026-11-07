from __future__ import annotations

import math

import torch


def make_qkv(B, Sq, Sk, H, Hk, D, dtype, device, std=0.5):
    q = torch.empty((B, Sq, H, D), dtype=dtype, device=device).normal_(0, std)
    k = torch.empty((B, Sk, Hk, D), dtype=dtype, device=device).normal_(0, std)
    v = torch.empty((B, Sk, Hk, D), dtype=dtype, device=device).normal_(0, std)
    return q, k, v


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def mean_diff(a, b):
    return (a.float() - b.float()).abs().mean().item()


def tolerance(dtype):
    return 2e-2 if dtype == torch.float16 else 4e-2


def ref_dtype(B, H, Sq, Sk):
    # Keep fp64 for small problems (better diagnostics), fp32 for big ones.
    return torch.float64 if B * H * Sq * Sk <= 4_000_000 else torch.float32


def default_scale(D):
    return 1.0 / math.sqrt(D)
