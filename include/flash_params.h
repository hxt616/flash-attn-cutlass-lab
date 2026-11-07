#pragma once

#include <cstdint>

// Shared, framework-agnostic launch parameters for the FlashAttention-2
// forward kernel. Strides are expressed in elements (not bytes).
struct Flash_fwd_params {
  const void* q_ptr = nullptr;
  const void* k_ptr = nullptr;
  const void* v_ptr = nullptr;
  void* out_ptr = nullptr;
  void* lse_ptr = nullptr;  // optional: float [batch, q_head, q_seqlen]

  int batch = 0;
  int q_seqlen = 0;
  int k_seqlen = 0;
  int dim = 0;

  int q_head = 0;
  int k_head = 0;  // grouped-query attention: q_head must be a multiple of k_head

  int64_t q_batch_stride = 0;
  int64_t q_head_stride = 0;
  int64_t q_row_stride = 0;

  int64_t k_batch_stride = 0;
  int64_t k_head_stride = 0;
  int64_t k_row_stride = 0;

  int64_t v_batch_stride = 0;
  int64_t v_head_stride = 0;
  int64_t v_row_stride = 0;

  int64_t o_batch_stride = 0;
  int64_t o_head_stride = 0;
  int64_t o_row_stride = 0;

  int64_t lse_batch_stride = 0;
  int64_t lse_head_stride = 0;
  int64_t lse_row_stride = 0;

  float softmax_scale = 1.0f;
  bool is_causal = false;
  bool is_bf16 = false;
};
