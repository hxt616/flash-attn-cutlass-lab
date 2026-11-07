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

}  // namespace flash_lab
