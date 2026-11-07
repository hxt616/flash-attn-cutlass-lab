"""Python-facing wrapper around the CUDA/CuTe FlashAttention-2 extension."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

import attention_cutlass as _ext

__version__ = "0.1.0"


def _default_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(float(head_dim))


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed-length self/cross attention.

    Args:
        q: (batch, seqlen_q, nheads, headdim) fp16/bf16, contiguous.
        k: (batch, seqlen_k, nheads_k, headdim).
        v: (batch, seqlen_k, nheads_k, headdim).
        causal: apply the causal mask.
        softmax_scale: defaults to 1 / sqrt(headdim).

    Returns:
        (out, lse) with out in the input dtype and lse float32
        (batch, nheads, seqlen_q).
    """
    if softmax_scale is None:
        softmax_scale = _default_scale(q.shape[-1])
    return _ext.flash_attention_v2_cutlass(
        q, k, v, bool(causal), float(softmax_scale)
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Variable-length attention without padding.

    q/k/v are packed as (total_tokens, nheads, headdim); cu_seqlens_* are
    int32 prefix sums on the GPU. Implemented as one fused launch: a device
    tile map maps each CTA to (sequence, q_block).
    """
    if softmax_scale is None:
        softmax_scale = _default_scale(q.shape[-1])
    return _ext.flash_attention_varlen_cutlass(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        bool(causal),
        float(softmax_scale),
    )


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: Optional[float] = None,
    num_splits: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paged-KV decode attention.

    q: (batch, 1, nheads, headdim)
    k_cache/v_cache: (num_pages, page_size, nheads_k, headdim)
    block_table: (batch, max_pages) int32 logical -> physical page
    cache_seqlens: (batch,) int32 valid KV length
    Returns (out, lse) with out (batch, 1, nheads, headdim), lse (batch, nheads, 1).
    """
    if softmax_scale is None:
        softmax_scale = _default_scale(q.shape[-1])
    return _ext.flash_attention_decode_cutlass(
        q,
        k_cache,
        v_cache,
        block_table,
        cache_seqlens,
        float(softmax_scale),
        int(num_splits),
    )
