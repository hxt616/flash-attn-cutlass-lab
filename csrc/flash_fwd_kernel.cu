#include <torch/extension.h>

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

  // cu_seqlens lives on the GPU; the current varlen implementation launches
  // one kernel per sequence, so the host needs the prefix sums. This host copy
  // is the price of the simple (non-persistent) varlen implementation.
  auto cu_q_host = cu_seqlens_q.to(torch::kCPU);
  auto cu_k_host = cu_seqlens_k.to(torch::kCPU);
  auto cu_q = cu_q_host.data_ptr<int>();
  auto cu_k = cu_k_host.data_ptr<int>();

  for (int b = 0; b < batch; ++b) {
    const int q_start = cu_q[b], q_end = cu_q[b + 1];
    const int k_start = cu_k[b], k_end = cu_k[b + 1];
    const int qlen = q_end - q_start;
    const int klen = k_end - k_start;
    TORCH_CHECK(!is_causal || qlen == klen,
                "causal varlen currently requires seqlen_q == seqlen_k");
    if (qlen == 0) continue;

    Flash_fwd_params p;
    p.q_ptr = static_cast<char*>(q.data_ptr()) +
              int64_t(q_start) * H * D * q.element_size();
    p.k_ptr = static_cast<char*>(k.data_ptr()) +
              int64_t(k_start) * Hk * D * k.element_size();
    p.v_ptr = static_cast<char*>(v.data_ptr()) +
              int64_t(k_start) * Hk * D * v.element_size();
    p.out_ptr = static_cast<char*>(out.data_ptr()) +
                int64_t(q_start) * H * D * out.element_size();
    p.lse_ptr = static_cast<char*>(lse.data_ptr()) +
                int64_t(q_start) * H * lse.element_size();
    p.batch = 1;
    p.q_seqlen = qlen;
    p.k_seqlen = klen;
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

    run_flash_attn_cutlass(p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out, lse};
}