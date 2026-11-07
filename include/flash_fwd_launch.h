#pragma once

#include <cuda_runtime.h>

#include "flash_fwd_kernel_impl.cuh"

namespace flash_lab {

template <typename Kernel_traits, bool Is_causal>
inline void launch_flash_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

  const int num_m_block =
      (params.q_seqlen + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.batch * params.q_head, 1);
  dim3 block(Kernel_traits::kNThreads);

  const int smem_size = static_cast<int>(
      sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  auto kernel = &flash_attention_v2_kernel<Kernel_traits, Is_causal,
                                           Flash_fwd_params>;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
  }
  kernel<<<grid, block, smem_size, stream>>>(params);
}

inline void launch_build_tile_map(const int* cu_q, const int* cu_k,
                                  const int batch, const int block_m,
                                  int* tile_seq, int* tile_m,
                                  int* total_tiles, const int max_tiles,
                                  cudaStream_t stream) {
  if (batch <= 0) return;
  const int threads = 128;
  const int blocks = (batch + threads - 1) / threads;
  build_tile_map_kernel<<<blocks, threads, 0, stream>>>(
      cu_q, cu_k, batch, block_m, tile_seq, tile_m, total_tiles, max_tiles);
}

template <typename Kernel_traits, bool Is_causal>
inline void launch_flash_fwd_varlen(Flash_fwd_params& params,
                                    const int total_q_tiles,
                                    cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

  dim3 grid(total_q_tiles, params.q_head, 1);
  dim3 block(Kernel_traits::kNThreads);

  const int smem_size = static_cast<int>(
      sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  auto kernel = &flash_attention_v2_varlen_kernel<Kernel_traits, Is_causal,
                                                  Flash_fwd_params>;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
  }
  kernel<<<grid, block, smem_size, stream>>>(params);
}

template <typename Kernel_traits>
inline void launch_flash_fwd_decode(Flash_fwd_params& params,
                                    cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

  dim3 grid(1, params.batch * params.q_head, 1);
  dim3 block(Kernel_traits::kNThreads);

  const int smem_size = static_cast<int>(
      sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  auto kernel = &flash_attention_decode_kernel<Kernel_traits,
                                               Flash_fwd_params>;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
  }
  kernel<<<grid, block, smem_size, stream>>>(params);
}

template <typename Kernel_traits>
inline void launch_flash_fwd_decode_split(Flash_fwd_params& params,
                                          cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

  dim3 grid(params.num_splits, params.batch * params.q_head, 1);
  dim3 block(Kernel_traits::kNThreads);

  const int smem_size = static_cast<int>(
      sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  auto kernel = &flash_attention_decode_split_kernel<
      Kernel_traits, Flash_fwd_params>;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
  }
  kernel<<<grid, block, smem_size, stream>>>(params);
}

template <typename Element>
inline void launch_combine_decode(const Element* partial_o,
                                  const float* partial_lse, Element* out,
                                  float* out_lse, const int BH,
                                  const int num_splits, const int D,
                                  cudaStream_t stream) {
  combine_decode_kernel<Element><<<BH, 256, 0, stream>>>(
      partial_o, partial_lse, out, out_lse, BH, num_splits, D);
}

}  // namespace flash_lab
