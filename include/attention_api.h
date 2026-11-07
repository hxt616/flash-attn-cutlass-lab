#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flash_attention_v2_cutlass(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal = false,
    double softmax_scale = 1.0);

std::vector<torch::Tensor> flash_attention_varlen_cutlass(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k,
    bool is_causal = false, double softmax_scale = 1.0);

std::vector<torch::Tensor> flash_attention_decode_cutlass(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor cache_seqlens,
    double softmax_scale = 1.0);
