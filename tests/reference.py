"""Reference attention implementations used by the correctness tests."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, Hk, S, D) -> (B, Hk * n_rep, S, D) without extra copies if possible."""
    if n_rep == 1:
        return x
    b, hk, s, d = x.shape
    return x[:, :, None, :, :].expand(b, hk, n_rep, s, d).reshape(b, hk * n_rep, s, d)


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialized attention reference.

    q: (B, Sq, H, D), k/v: (B, Sk, Hk, D). Returns (out, lse) with out in
    (B, Sq, H, D) and lse in (B, H, Sq).
    """
    out_dtype = q.dtype
    qh = q.transpose(1, 2).to(dtype)
    kh = k.transpose(1, 2).to(dtype)
    vh = v.transpose(1, 2).to(dtype)

    if kh.shape[1] != qh.shape[1]:
        kh = repeat_kv(kh, qh.shape[1] // kh.shape[1])
        vh = repeat_kv(vh, qh.shape[1] // vh.shape[1])

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    scores = torch.matmul(qh, kh.transpose(-1, -2)) * softmax_scale
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        rows = torch.arange(sq, device=scores.device)[:, None]
        cols = torch.arange(sk, device=scores.device)[None, :]
        scores = scores.masked_fill(cols > rows, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh)
    return out.transpose(1, 2).to(out_dtype), lse


def attention_ref_padded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool,
    softmax_scale: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Varlen reference built by looping over sequences."""
    outs = []
    flat = torch.empty(q.shape[0], q.shape[1], device=q.device, dtype=torch.float32)
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    off = 0
    for i in range(len(cq) - 1):
        qs, qe = cq[i], cq[i + 1]
        ks, ke = ck[i], ck[i + 1]
        qi = q[qs:qe].unsqueeze(0)
        ki = k[ks:ke].unsqueeze(0)
        vi = v[ks:ke].unsqueeze(0)
        o, l = attention_ref(qi, ki, vi, causal, softmax_scale)
        o = o.squeeze(0)
        outs.append(o)
        n = o.shape[0]
        if n:
            # l is (1, H, n) -> (n, H) to match the CUDA API.
            flat[off : off + n] = l.squeeze(0).transpose(0, 1)
        off += n
    out = torch.cat(outs, dim=0)
    return out, flat
