"""Compare our numerical error against PyTorch's own fp16 attention error.

This follows the official flash-attention test philosophy: FlashAttention is
allowed to be at most ~2x worse than a PyTorch implementation of the same
precision, measured against an fp64 reference.
"""

import pytest
import torch
import torch.nn.functional as F

import flash_attn_lab
from reference import attention_ref, repeat_kv
from helpers import default_scale, make_qkv, max_diff, ref_dtype


def _sdpa(q, k, v, causal, scale):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    ).transpose(1, 2)


@pytest.mark.cuda
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("S", [128, 200, 512, 1024])
@pytest.mark.parametrize("causal", [False, True])
def test_error_vs_pytorch(device, D, S, causal):
    B, H = 1, 4
    q, k, v = make_qkv(B, S, S, H, H, D, torch.float16, device)
    scale = default_scale(D)
    out, _ = flash_attn_lab.flash_attn_func(q, k, v, causal, scale)
    out_pt = _sdpa(q, k, v, causal, scale)
    ref, _ = attention_ref(q, k, v, causal, scale, ref_dtype(B, H, S, S))
    err_our = max_diff(out, ref)
    err_pt = max_diff(out_pt, ref)
    assert err_our <= 2.0 * err_pt + 1e-3, (
        f"our error {err_our:.6f} > 2x torch error {err_pt:.6f} "
        f"(D={D}, S={S}, causal={causal})"
    )


@pytest.mark.cuda
def test_error_vs_pytorch_gqa(device):
    B, H, Hk, S, D = 2, 8, 2, 512, 64
    q, k, v = make_qkv(B, S, S, H, Hk, D, torch.float16, device)
    scale = default_scale(D)
    out, _ = flash_attn_lab.flash_attn_func(q, k, v, True, scale)
    kr = repeat_kv(k.transpose(1, 2), H // Hk).transpose(1, 2)
    vr = repeat_kv(v.transpose(1, 2), H // Hk).transpose(1, 2)
    out_pt = _sdpa(q, kr, vr, True, scale)
    ref, _ = attention_ref(q, k, v, True, scale, torch.float64)
    assert max_diff(out, ref) <= 2.0 * max_diff(out_pt, ref) + 1e-3
