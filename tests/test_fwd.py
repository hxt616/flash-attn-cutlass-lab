import pytest
import torch

import flash_attn_lab
from reference import attention_ref
from helpers import default_scale, make_qkv, max_diff, mean_diff, ref_dtype, tolerance

DIMS = [32, 64, 128]
SEQLENS = [1, 17, 64, 65, 97, 128, 129, 200, 257, 511, 512, 1024, 1025]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", DIMS)
@pytest.mark.parametrize("S", SEQLENS)
@pytest.mark.parametrize("causal", [False, True])
def test_fwd_shapes(device, dtype, D, S, causal):
    B, H = 1, 1
    q, k, v = make_qkv(B, S, S, H, H, D, dtype, device)
    scale = default_scale(D)
    out, lse = flash_attn_lab.flash_attn_func(q, k, v, causal, scale)
    ref, lse_ref = attention_ref(q, k, v, causal, scale, ref_dtype(B, H, S, S))
    atol = tolerance(dtype)
    assert max_diff(out, ref) <= atol, (
        f"out mismatch dtype={dtype} D={D} S={S} causal={causal}: "
        f"max={max_diff(out, ref):.6f} mean={mean_diff(out, ref):.6f}"
    )
    assert max_diff(lse, lse_ref) <= 2e-2


@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("D", [64, 128])
def test_fwd_multihead(device, causal, D):
    B, H, S = 2, 8, 512
    q, k, v = make_qkv(B, S, S, H, H, D, torch.float16, device)
    scale = default_scale(D)
    out, lse = flash_attn_lab.flash_attn_func(q, k, v, causal, scale)
    ref, lse_ref = attention_ref(q, k, v, causal, scale, torch.float32)
    assert max_diff(out, ref) <= 2e-2
    assert max_diff(lse, lse_ref) <= 2e-2


@pytest.mark.cuda
def test_fwd_custom_scale(device):
    B, H, S, D = 1, 4, 200, 64
    q, k, v = make_qkv(B, S, S, H, H, D, torch.float16, device)
    for scale in [1.0, 0.3, 0.05]:
        out, _ = flash_attn_lab.flash_attn_func(q, k, v, True, scale)
        ref, _ = attention_ref(q, k, v, True, scale, torch.float64)
        assert max_diff(out, ref) <= 2e-2


@pytest.mark.cuda
def test_lse_matches_logsumexp(device):
    B, H, S, D = 1, 2, 300, 64
    q, k, v = make_qkv(B, S, S, H, H, D, torch.float16, device)
    scale = default_scale(D)
    _, lse = flash_attn_lab.flash_attn_func(q, k, v, False, scale)
    _, lse_ref = attention_ref(q, k, v, False, scale, torch.float64)
    assert max_diff(lse, lse_ref) <= 2e-2
