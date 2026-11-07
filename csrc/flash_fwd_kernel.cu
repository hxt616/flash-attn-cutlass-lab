#include <torch/extension.h>

#include <limits>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "attention_api.h"
#include "flash_params.h"
#include "flash_fwd_launch.h"
#include "static_switch.h"

namespace {

void check_common(const torch::Tensor& q, const torch::Tensor& k,
                  const torch::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "q/k/v must be CUDA tensors");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "q/k/v must be contiguous");
  const auto dt = q.scalar_type();
  TORCH_CHECK(dt == at::kHalf || dt == at::kBFloat16,
              "only fp16 and bf16 are supported");
  TORCH_CHECK(k.scalar_type() == dt && v.scalar_type() == dt,
              "q/k/v must have the same dtype");
  TORCH_CHECK(q.size(-1) == k.size(-1) && q.size(-1) == v.size(-1),
              "q/k/v head dim must match");
  TORCH_CHECK(q.size(-1) <= 256, "head dim must be <= 256");
  TORCH_CHECK(q.size(-1) % 32 == 0, "head dim must be a multiple of 32");
}

template <typename T, int Headdim>
void run_flash_fwd_(Flash_fwd_params& params, cudaStream_t stream) {
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    // Measured on RTX 4060 laptop: kBlockN=128 is ~9% faster for fp16
    // non-causal D=64, while causal and D=32/128 prefer kBlockN=64.
    if constexpr (Headdim == 64) {
      if constexpr (!Is_causal) {
        flash_lab::launch_flash_fwd<
            Flash_fwd_kernel_traits<Headdim, 64, 128, 4, T>, Is_causal>(
            params, stream);
        return;
      }
    }
    flash_lab::launch_flash_fwd<
        Flash_fwd_kernel_traits<Headdim, 64, 64, 4, T>, Is_causal>(params,
                                                                   stream);
  });
}

void run_flash_attn_cutlass(Flash_fwd_params& params, cudaStream_t stream) {
  TORCH_CHECK(params.dim <= 256, "head dim must be <= 256");
  FP16_SWITCH(!params.is_bf16, [&] {
    FWD_HEADDIM_SWITCH(params.dim, [&] {
      run_flash_fwd_<elem_type, kHeadDim>(params, stream);
    });
  });
}

template <typename T, int Headdim>
void run_flash_fwd_varlen_(Flash_fwd_params& params, int total_q_tiles,
                           cudaStream_t stream) {
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    if constexpr (Headdim == 64) {
      if constexpr (!Is_causal) {
        flash_lab::launch_flash_fwd_varlen<
            Flash_fwd_kernel_traits<Headdim, 64, 128, 4, T>, Is_causal>(
            params, total_q_tiles, stream);
        return;
      }
    }
    flash_lab::launch_flash_fwd_varlen<
        Flash_fwd_kernel_traits<Headdim, 64, 64, 4, T>, Is_causal>(
        params, total_q_tiles, stream);
  });
}

void run_flash_attn_varlen_cutlass(Flash_fwd_params& params,
                                   int total_q_tiles, cudaStream_t stream) {
  TORCH_CHECK(params.dim <= 256, "head dim must be <= 256");
  FP16_SWITCH(!params.is_bf16, [&] {
    FWD_HEADDIM_SWITCH(params.dim, [&] {
      run_flash_fwd_varlen_<elem_type, kHeadDim>(params, total_q_tiles,
                                                 stream);
    });
  });
}

template <typename T, int Headdim>
void run_flash_fwd_decode_(Flash_fwd_params& params, cudaStream_t stream) {
  using Traits = Flash_fwd_kernel_traits<Headdim, 64, 64, 4, T>;
  if (params.num_splits > 1) {
    flash_lab::launch_flash_fwd_decode_split<Traits>(params, stream);
    flash_lab::launch_combine_decode<T>(
        reinterpret_cast<const T*>(params.partial_o_ptr),
        params.partial_lse_ptr, reinterpret_cast<T*>(params.out_ptr),
        reinterpret_cast<float*>(params.lse_ptr),
        params.batch * params.q_head, params.num_splits, params.dim, stream);
  } else {
    flash_lab::launch_flash_fwd_decode<Traits>(params, stream);
  }
}

void run_flash_attn_decode_cutlass(Flash_fwd_params& params,
                                   cudaStream_t stream) {
  TORCH_CHECK(params.dim <= 256, "head dim must be <= 256");
  FP16_SWITCH(!params.is_bf16, [&] {
    FWD_HEADDIM_SWITCH(params.dim, [&] {
      run_flash_fwd_decode_<elem_type, kHeadDim>(params, stream);
    });
  });
}

}  // namespace

std::vector<torch::Tensor> flash_attention_v2_cutlass(torch::Tensor q,
                                                      torch::Tensor k,
                                                      torch::Tensor v,
                                                      bool is_causal,
                                                      double softmax_scale) {
  check_common(q, k, v);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "q/k/v must be (batch, seqlen, head, dim)");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "batch size must match");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "seqlen must match for self-attention");
  TORCH_CHECK(q.size(2) % k.size(2) == 0,
              "q head count must be a multiple of kv head count");

  const at::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int B = int(q.size(0));
  const int S = int(q.size(1));
  const int H = int(q.size(2));
  const int Hk = int(k.size(2));
  const int D = int(q.size(3));

  auto out = torch::empty_like(q);
  auto lse = torch::empty({B, H, S}, q.options().dtype(torch::kFloat32));

  Flash_fwd_params p;
  p.q_ptr = q.data_ptr();
  p.k_ptr = k.data_ptr();
  p.v_ptr = v.data_ptr();
  p.out_ptr = out.data_ptr();
  p.lse_ptr = lse.data_ptr();
  p.batch = B;
  p.q_seqlen = S;
  p.k_seqlen = S;
  p.dim = D;
  p.q_head = H;
  p.k_head = Hk;
  p.q_batch_stride = q.stride(0);
  p.q_head_stride = q.stride(2);
  p.q_row_stride = q.stride(1);
  p.k_batch_stride = k.stride(0);
  p.k_head_stride = k.stride(2);
  p.k_row_stride = k.stride(1);
  p.v_batch_stride = v.stride(0);
  p.v_head_stride = v.stride(2);
  p.v_row_stride = v.stride(1);
  p.o_batch_stride = out.stride(0);
  p.o_head_stride = out.stride(2);
  p.o_row_stride = out.stride(1);
  p.lse_batch_stride = lse.stride(0);
  p.lse_head_stride = lse.stride(1);
  p.lse_row_stride = lse.stride(2);
  p.softmax_scale = float(softmax_scale);
  p.is_causal = is_causal;
  p.is_bf16 = q.scalar_type() == at::kBFloat16;

  run_flash_attn_cutlass(p, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

std::vector<torch::Tensor> flash_attention_varlen_cutlass(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k, bool is_causal,
    double softmax_scale) {
  check_common(q, k, v);
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
              "varlen q/k/v must be (total_seqlen, head, dim)");
  TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(),
              "cu_seqlens must be CUDA tensors");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt &&
                  cu_seqlens_k.scalar_type() == at::kInt,
              "cu_seqlens must be int32");
  TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
              "cu_seqlens must be contiguous");
  const int batch = int(cu_seqlens_q.numel()) - 1;
  TORCH_CHECK(batch >= 0 && cu_seqlens_k.numel() == cu_seqlens_q.numel(),
              "cu_seqlens_q/k must have the same length");
  TORCH_CHECK(q.size(1) % k.size(1) == 0,
              "q head count must be a multiple of kv head count");

  const at::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int H = int(q.size(1));
  const int Hk = int(k.size(1));
  const int D = int(q.size(2));

  auto out = torch::empty_like(q);
  auto lse = torch::empty({q.size(0), H}, q.options().dtype(torch::kFloat32));

  TORCH_CHECK(!is_causal || q.size(0) == k.size(0),
              "causal varlen currently requires equal total lengths");

  // Build the tile -> (sequence, m_block) map on the device. No cu_seqlens
  // device-to-host copy is needed. The host only knows an upper bound on the
  // number of Q tiles; the attention kernel exits extra CTAs.
  constexpr int kBlockM = 64;
  const int max_tiles =
      (int(q.size(0)) + kBlockM - 1) / kBlockM + batch;
  if (max_tiles <= 0) {
    return {out, lse};
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
  auto tile_seq_t = torch::empty({max_tiles}, opts);
  auto tile_m_t = torch::empty({max_tiles}, opts);
  auto total_tiles_t = torch::zeros({1}, opts);

  flash_lab::launch_build_tile_map(
      cu_seqlens_q.data_ptr<int>(), cu_seqlens_k.data_ptr<int>(), batch,
      kBlockM, tile_seq_t.data_ptr<int>(), tile_m_t.data_ptr<int>(),
      total_tiles_t.data_ptr<int>(), max_tiles, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  Flash_fwd_params p;
  p.q_ptr = q.data_ptr();
  p.k_ptr = k.data_ptr();
  p.v_ptr = v.data_ptr();
  p.out_ptr = out.data_ptr();
  p.lse_ptr = lse.data_ptr();
  p.batch = batch;
  p.q_seqlen = int(q.size(0));
  p.k_seqlen = int(k.size(0));
  p.dim = D;
  p.q_head = H;
  p.k_head = Hk;
  p.q_batch_stride = 0;
  p.q_head_stride = D;
  p.q_row_stride = int64_t(H) * D;
  p.k_batch_stride = 0;
  p.k_head_stride = D;
  p.k_row_stride = int64_t(Hk) * D;
  p.v_batch_stride = 0;
  p.v_head_stride = D;
  p.v_row_stride = int64_t(Hk) * D;
  p.o_batch_stride = 0;
  p.o_head_stride = D;
  p.o_row_stride = int64_t(H) * D;
  p.lse_batch_stride = 0;
  p.lse_head_stride = 1;
  p.lse_row_stride = H;
  p.softmax_scale = float(softmax_scale);
  p.is_causal = is_causal;
  p.is_bf16 = q.scalar_type() == at::kBFloat16;
  p.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  p.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
  p.tile_to_seq = tile_seq_t.data_ptr<int>();
  p.tile_to_mblock = tile_m_t.data_ptr<int>();
  p.total_q_tiles_dev = total_tiles_t.data_ptr<int>();
  p.total_q_tiles = max_tiles;

  run_flash_attn_varlen_cutlass(p, max_tiles, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

std::vector<torch::Tensor> flash_attention_decode_cutlass(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor cache_seqlens,
    double softmax_scale, int num_splits) {
  check_common(q, k_cache, v_cache);
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 1,
              "decode q must be (batch, 1, head, dim)");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
              "k_cache/v_cache must be (num_pages, page_size, head, dim)");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(),
              "k_cache/v_cache must have the same shape");
  TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == at::kInt &&
                  block_table.dim() == 2 && block_table.is_contiguous(),
              "block_table must be a contiguous int32 (batch, max_pages)");
  TORCH_CHECK(cache_seqlens.is_cuda() &&
                  cache_seqlens.scalar_type() == at::kInt &&
                  cache_seqlens.dim() == 1 && cache_seqlens.is_contiguous(),
              "cache_seqlens must be a contiguous int32 (batch)");
  TORCH_CHECK(block_table.size(0) == q.size(0) &&
                  cache_seqlens.size(0) == q.size(0),
              "block_table/cache_seqlens batch must match q");
  TORCH_CHECK(q.size(2) % k_cache.size(2) == 0,
              "q head count must be a multiple of kv head count");
  TORCH_CHECK(k_cache.size(1) % 64 == 0,
              "page_block_size must be a multiple of 64");

  const at::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int B = int(q.size(0));
  const int H = int(q.size(2));
  const int Hk = int(k_cache.size(2));
  const int D = int(q.size(3));
  const int page_size = int(k_cache.size(1));
  const int BH = B * H;

  if (num_splits <= 0) {
    // Small batches need more KV splits to fill the GPU; large batches already
    // have enough (batch * head) CTAs, so use fewer splits to avoid combine
    // overhead. Capped at 8.
    const int target = BH < 64 ? 48 : 192;
    num_splits = (target + BH - 1) / BH;
    num_splits = num_splits < 1 ? 1 : num_splits;
    num_splits = num_splits > 8 ? 8 : num_splits;
  }
  TORCH_CHECK(num_splits >= 1 && num_splits <= 32,
              "num_splits must be in [1, 32]");

  auto out = torch::empty_like(q);
  auto lse = torch::empty({B, H, 1}, q.options().dtype(torch::kFloat32));
  torch::Tensor partial_o;
  torch::Tensor partial_lse;
  if (num_splits > 1) {
    partial_o = torch::empty({BH * num_splits, D}, q.options());
    partial_lse = torch::full(
        {BH * num_splits}, -std::numeric_limits<float>::infinity(),
        q.options().dtype(torch::kFloat32));
  }

  Flash_fwd_params p;
  p.q_ptr = q.data_ptr();
  p.k_ptr = k_cache.data_ptr();
  p.v_ptr = v_cache.data_ptr();
  p.out_ptr = out.data_ptr();
  p.lse_ptr = lse.data_ptr();
  p.batch = B;
  p.q_seqlen = 1;
  p.k_seqlen = 0;
  p.dim = D;
  p.q_head = H;
  p.k_head = Hk;
  p.q_batch_stride = q.stride(0);
  p.q_head_stride = q.stride(2);
  p.q_row_stride = q.stride(1);
  p.k_batch_stride = k_cache.stride(0);
  p.k_head_stride = k_cache.stride(2);
  p.k_row_stride = k_cache.stride(1);
  p.v_batch_stride = v_cache.stride(0);
  p.v_head_stride = v_cache.stride(2);
  p.v_row_stride = v_cache.stride(1);
  p.o_batch_stride = out.stride(0);
  p.o_head_stride = out.stride(2);
  p.o_row_stride = out.stride(1);
  p.lse_batch_stride = lse.stride(0);
  p.lse_head_stride = lse.stride(1);
  p.lse_row_stride = lse.stride(2);
  p.softmax_scale = float(softmax_scale);
  p.is_causal = false;
  p.is_bf16 = q.scalar_type() == at::kBFloat16;
  p.block_table = block_table.data_ptr<int>();
  p.block_table_batch_stride = int(block_table.stride(0));
  p.page_block_size = page_size;
  p.cache_seqlens = cache_seqlens.data_ptr<int>();
  p.num_splits = num_splits;
  if (num_splits > 1) {
    p.partial_o_ptr = partial_o.data_ptr();
    p.partial_lse_ptr = partial_lse.data_ptr<float>();
  }

  run_flash_attn_decode_cutlass(p, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}