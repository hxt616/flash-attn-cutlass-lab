import pytest
import torch

import flash_attn_lab
from reference import attention_ref, attention_ref_padded, repeat_kv
from helpers import default_scale, make_qkv, max_diff, ref_dtype, tolerance


@pytest.mark.cuda
@pytest.mark.parametrize("Hk", [8, 4, 2, 1])
@pytest.mark.parametrize("causal", [False, True])
def test_gqa(device, Hk, causal):
    B, H, S, D = 2, 8, 128, 64
    q, k, v = make_qkv(B, S, S, H, Hk, D, torch.float16, device)
    scale = default_scale(D)
    out, lse = flash_attn_lab.flash_attn_func(q, k, v, causal, scale)
    ref, lse_ref = attention_ref(q, k, v, causal, scale, torch.float64)
    assert max_diff(out, ref) <= 2e-2
    assert max_diff(lse, lse_ref) <= 2e-2


@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
def test_varlen(device, causal):
    torch.manual_seed(7)
    lengths = [1, 17, 64, 97, 200, 3]
    H, Hk, D = 4, 2, 64
    total_q = sum(lengths)
    total_k = total_q
    q = torch.empty((total_q, H, D), dtype=torch.float16, device=device).normal_(0, 0.5)
    k = torch.empty((total_k, Hk, D), dtype=torch.float16, device=device).normal_(0, 0.5)
    v = torch.empty((total_k, Hk, D), dtype=torch.float16, device=device).normal_(0, 0.5)

    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist()),
                      dtype=torch.int32, device=device)
    scale = default_scale(D)
    out, lse = flash_attn_lab.flash_attn_varlen_func(
        q, k, v, cu, cu, causal, scale
    )
    ref, lse_ref = attention_ref_padded(q, k, v, cu, cu, causal, scale)
    assert max_diff(out, ref) <= 2e-2
    assert max_diff(lse, lse_ref) <= 2e-2


@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_many_short_sequences(device, causal):
    """Many short sequences exercise the fused-varlen CTA packing."""
    lengths = [16] * 32
    H, Hk, D = 16, 4, 128
    total_q = sum(lengths)
    total_k = total_q
    q = torch.empty((total_q, H, D), dtype=torch.bfloat16, device=device).normal_(0, 0.5)
    k = torch.empty((total_k, Hk, D), dtype=torch.bfloat16, device=device).normal_(0, 0.5)
    v = torch.empty((total_k, Hk, D), dtype=torch.bfloat16, device=device).normal_(0, 0.5)

    cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    scale = default_scale(D)
    out, lse = flash_attn_lab.flash_attn_varlen_func(
        q, k, v, cu, cu, causal, scale
    )
    ref, lse_ref = attention_ref_padded(q, k, v, cu, cu, causal, scale)
    assert max_diff(out, ref) <= tolerance(torch.bfloat16)
    assert max_diff(lse, lse_ref) <= 2e-2


@pytest.mark.cuda
def test_varlen_zero_length(device):
    lengths = [3, 0, 5]
    H, Hk, D = 2, 2, 64
    total_q = sum(lengths)
    q = torch.randn((total_q, H, D), dtype=torch.float16, device=device) * 0.5
    k = torch.randn((total_q, Hk, D), dtype=torch.float16, device=device) * 0.5
    v = torch.randn((total_q, Hk, D), dtype=torch.float16, device=device) * 0.5
    cu = torch.tensor([0, 3, 3, 8], dtype=torch.int32, device=device)
    out, lse = flash_attn_lab.flash_attn_varlen_func(
        q, k, v, cu, cu, True, default_scale(D)
    )
    ref, _ = attention_ref_padded(q, k, v, cu, cu, True, default_scale(D))
    assert max_diff(out, ref) <= 2e-2

    # The zero-length sequence must not produce NaNs in the output/lse.
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse).all()
