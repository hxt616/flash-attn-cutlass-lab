import pytest
import torch

import flash_attn_lab
from reference import attention_ref
from helpers import default_scale, make_qkv, max_diff, tolerance


@pytest.mark.cuda
@pytest.mark.parametrize("S", [1, 2, 63, 65, 127, 129])
@pytest.mark.parametrize("causal", [False, True])
def test_tiny_sequences(device, S, causal):
    B, H, D = 1, 2, 64
    q, k, v = make_qkv(B, S, S, H, H, D, torch.float16, device)
    out, lse = flash_attn_lab.flash_attn_func(
        q, k, v, causal, default_scale(D)
    )
    ref, lse_ref = attention_ref(
        q, k, v, causal, default_scale(D), torch.float64
    )
    assert max_diff(out, ref) <= 2e-2
    assert max_diff(lse, lse_ref) <= 2e-2


@pytest.mark.cuda
def test_bf16(device):
    B, H, S, D = 1, 4, 300, 64
    q, k, v = make_qkv(B, S, S, H, H, D, torch.bfloat16, device)
    out, _ = flash_attn_lab.flash_attn_func(q, k, v, True, default_scale(D))
    ref, _ = attention_ref(q, k, v, True, default_scale(D), torch.float64)
    assert max_diff(out, ref) <= tolerance(torch.bfloat16)


@pytest.mark.cuda
def test_rejects_non_contiguous(device):
    q, k, v = make_qkv(1, 64, 64, 2, 2, 64, torch.float16, device)
    q_bad = q.transpose(1, 2)
    with pytest.raises(RuntimeError):
        flash_attn_lab.flash_attn_func(q_bad, k, v)


@pytest.mark.cuda
def test_rejects_wrong_dtype(device):
    q, k, v = make_qkv(1, 64, 64, 2, 2, 64, torch.float32, device)
    with pytest.raises(RuntimeError):
        flash_attn_lab.flash_attn_func(q, k, v)


@pytest.mark.cuda
@pytest.mark.parametrize("D", [48, 96, 160, 512])
def test_head_dim_contract(device, D):
    q, k, v = make_qkv(1, 64, 64, 2, 2, D, torch.float16, device)
    if D % 32 == 0 and D <= 256:
        out, _ = flash_attn_lab.flash_attn_func(q, k, v)
        assert out.shape == q.shape
    else:
        with pytest.raises(RuntimeError):
            flash_attn_lab.flash_attn_func(q, k, v)


@pytest.mark.cuda
def test_rejects_shape_mismatch(device):
    q, k, _ = make_qkv(1, 64, 64, 4, 4, 64, torch.float16, device)
    v = torch.empty_like(k)
    k_bad = k[:, :-1]
    with pytest.raises(RuntimeError):
        flash_attn_lab.flash_attn_func(q, k_bad, v)


@pytest.mark.cuda
def test_rejects_bad_gqa_ratio(device):
    q, k, v = make_qkv(1, 64, 64, 4, 3, 64, torch.float16, device)
    with pytest.raises(RuntimeError):
        flash_attn_lab.flash_attn_func(q, k, v)
