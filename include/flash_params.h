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

  // Fused varlen metadata. All pointers are device pointers.
  // cu_seqlens_*: int32 prefix sums, length batch + 1.
  // tile_to_seq / tile_to_mblock: one entry per Q tile; the tile's sequence
  // and its m_block inside that sequence.
  const int* cu_seqlens_q = nullptr;
  const int* cu_seqlens_k = nullptr;
  int* tile_to_seq = nullptr;
  int* tile_to_mblock = nullptr;
  int* total_q_tiles_dev = nullptr;
  int total_q_tiles = 0;  // host upper bound used as grid.x

  // Paged-KV decode metadata. block_table maps a logical KV page to a
  // physical page in k_cache/v_cache. K/V are laid out as
  // (num_pages, page_block_size, k_head, dim); k_batch_stride is the page
  // stride, k_row_stride is the in-page row stride.
  const int* block_table = nullptr;
  int block_table_batch_stride = 0;
  int page_block_size = 0;
  const int* cache_seqlens = nullptr;

  float softmax_scale = 1.0f;
  bool is_causal = false;
  bool is_bf16 = false;
};
