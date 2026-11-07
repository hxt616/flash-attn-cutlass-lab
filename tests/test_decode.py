import pytest
import torch

import flash_attn_lab
from reference import attention_ref
from helpers import default_scale, max_diff, tolerance


def make_paged_inputs(B, H, Hk, D, cache_lens, page_size, dtype, device):
    max_len = max(cache_lens)
    max_pages = (max_len + page_size - 1) // page_size
    num_pages = B * max_pages
    k_cache = torch.randn((num_pages, page_size, Hk, D), dtype=dtype, device=device) * 0.5
    v_cache = torch.randn((num_pages, page_size, Hk, D), dtype=dtype, device=device) * 0.5
    block_table = torch.empty((B, max_pages), dtype=torch.int32, device=device)
    for b in range(B):
        block_table[b] = torch.randperm(num_pages, device=device)[:max_pages]
    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device=device)
    q = torch.randn((B, 1, H, D), dtype=dtype, device=device) * 0.5
    return q, k_cache, v_cache, block_table, cache_seqlens


def gather_kv(k_cache, v_cache, block_table, length, page_size):
    n_pages = (length + page_size - 1) // page_size
    pages = block_table[:n_pages].long()
    k = k_cache[pages].reshape(-1, *k_cache.shape[2:])[:length]
    v = v_cache[pages].reshape(-1, *v_cache.shape[2:])[:length]
    return k, v


@pytest.mark.cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,Hk", [(8, 8), (8, 2), (8, 1)])
def test_decode_gqa_and_pages(device, D, H, Hk):
    B = 3
    page_size = 256
    cache_lens = [1, 255, 257, 512, 1024, 2048][:B]
    dtype = torch.bfloat16
    q, k_cache, v_cache, block_table, cache_seqlens = make_paged_inputs(
        B, H, Hk, D, cache_lens, page_size, dtype, device
    )
    out, lse = flash_attn_lab.flash_attn_with_kvcache(
        q, k_cache, v_cache, block_table, cache_seqlens, default_scale(D)
    )

    refs = []
    lses = []
    for b in range(B):
        k, v = gather_kv(k_cache, v_cache, block_table[b], cache_lens[b], page_size)
        o, l = attention_ref(
            q[b : b + 1], k.unsqueeze(0), v.unsqueeze(0), False,
            default_scale(D), torch.float64,
        )
        refs.append(o)
        lses.append(l.squeeze(0))  # (H, 1)
    ref = torch.cat(refs, dim=0)
    lse_ref = torch.stack(lses, dim=0)  # (B, H, 1)
    assert max_diff(out, ref) <= tolerance(dtype)
    assert max_diff(lse, lse_ref) <= 2e-2


@pytest.mark.cuda
@pytest.mark.parametrize("page_size", [64, 128, 256])
def test_decode_page_sizes(device, page_size):
    B, H, Hk, D = 2, 8, 4, 64
    cache_lens = [64, 300]
    q, k_cache, v_cache, block_table, cache_seqlens = make_paged_inputs(
        B, H, Hk, D, cache_lens, page_size, torch.float16, device
    )
    out, _ = flash_attn_lab.flash_attn_with_kvcache(
        q, k_cache, v_cache, block_table, cache_seqlens, default_scale(D)
    )
    refs = []
    for b in range(B):
        k, v = gather_kv(k_cache, v_cache, block_table[b], cache_lens[b], page_size)
        o, _ = attention_ref(
            q[b : b + 1], k.unsqueeze(0), v.unsqueeze(0), False,
            default_scale(D), torch.float64,
        )
        refs.append(o)
    assert max_diff(out, torch.cat(refs, 0)) <= 2e-2


@pytest.mark.cuda
def test_decode_rejects_bad_page_size(device):
    B, H, Hk, D = 1, 4, 4, 64
    q, k_cache, v_cache, block_table, cache_seqlens = make_paged_inputs(
        B, H, Hk, D, [32], 48, torch.float16, device
    )
    with pytest.raises(RuntimeError):
        flash_attn_lab.flash_attn_with_kvcache(
            q, k_cache, v_cache, block_table, cache_seqlens
        )
