#pragma once

// FlashAttention-2 forward kernel expressed with CuTe.
//
// This file is intentionally framework-agnostic (no torch headers) so the same
// kernel can be compiled into the standalone correctness/benchmark binary and
// into the PyTorch extension.

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"

#include "flash_params.h"
#include "kernel_traits.h"
#include "utils.h"

namespace flash_lab {

using namespace cute;

// ---------------------------------------------------------------------------
// Causal / out-of-bounds masking on the (row, col) view of the QK accumulator.
// tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N)).
// ---------------------------------------------------------------------------
template <bool Is_causal, int kBlockM, int kBlockN, int kNWarps,
          typename Engine, typename Layout>
inline __device__ void apply_mask(Tensor<Engine, Layout>& tensor,
                                  const int m_block, const int n_block,
                                  const int k_seqlen) {
  static_assert(Layout::rank == 2, "Only support 2D Tensor");
  const int lane_id = threadIdx.x % 32;
  const int col_idx_offset = kBlockN * n_block + (lane_id % 4) * 2;
  const int row_idx_offset =
      kBlockM * m_block + (threadIdx.x / 32) * 16 + lane_id / 4;
  const int group_stride = kNWarps * 16;

#pragma unroll
  for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
    const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
    for (int j = 0; j < size<1, 0>(tensor); ++j) {
      const int col_idx = col_idx_base + j;
      const bool col_oob = col_idx >= k_seqlen;
#pragma unroll
      for (int mi = 0; mi < size<0, 0>(tensor); ++mi) {
#pragma unroll
        for (int mj = 0; mj < size<0, 1>(tensor); ++mj) {
          const int row_idx =
              row_idx_offset + mi * 8 + mj * group_stride;
          const bool causal_oob = Is_causal && (col_idx > row_idx);
          if (col_oob || causal_oob) {
            tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// gmem -> smem copy with an M/N-bound predicate. D is always a multiple of the
// vector width in this project, so only the M/N dimension needs predication.
// ---------------------------------------------------------------------------
template <bool Is_even_MN, bool Clear_OOB_MN,
          typename TiledCopy,
          typename Engine0, typename Layout0,
          typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
inline __device__ void copy_mn(TiledCopy tiled_copy,
                               Tensor<Engine0, Layout0> const& S,
                               Tensor<Engine1, Layout1>& D,
                               Tensor<Engine2, Layout2> const& identity_MN,
                               const int max_MN) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(identity_MN) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
      }
    } else if (Clear_OOB_MN) {
      cute::clear(D(_, m, _));
    }
  }
}

// Element-wise (scalar) fallback for a partial 8-row copy tile. This is only
// used for the last, ragged tile; full tiles stay on the vectorized path.
template <bool Clear_OOB, typename TensorS, typename TensorD, typename TensorId>
inline __device__ void copy_mn_elementwise(TensorS const& S, TensorD& D,
                                           TensorId const& identity,
                                           const int max_MN) {
  using DVal = typename TensorD::value_type;
#pragma unroll
  for (int i = 0; i < size(S); ++i) {
    if (get<0>(identity(i)) < max_MN) {
      D(i) = S(i);
    } else if (Clear_OOB) {
      D(i) = DVal(0);
    }
  }
}

// ---------------------------------------------------------------------------
// Two-GEMM inner-loop helpers.
// ---------------------------------------------------------------------------
template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy, typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB,
                                      Tensor3 const& tCsB, TiledMma tiled_mma,
                                      TiledCopy smem_tiled_copy_B,
                                      ThrCopy smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename Tensor4, typename TiledMma, typename TiledCopyA,
          typename TiledCopyB, typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB,
                                 Tensor3 const& tCsA, Tensor4 const& tCsB,
                                 TiledMma tiled_mma,
                                 TiledCopyA smem_tiled_copy_A,
                                 TiledCopyB smem_tiled_copy_B,
                                 ThrCopyA smem_thr_copy_A,
                                 ThrCopyB smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
  cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

template <int N>
CUTE_HOST_DEVICE void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Convert an fp32 accumulator fragment to the output element type.
template <typename To_type, typename Fragment>
inline __device__ auto convert_type_f32(Fragment const& acc_fp32) {
  Tensor acc = make_tensor<To_type>(shape(acc_fp32));
#pragma unroll
  for (int i = 0; i < size(acc_fp32); ++i) {
    acc(i) = static_cast<To_type>(acc_fp32(i));
  }
  return acc;
}

// exp2(x * scale_log2 - max * scale_log2), with the -inf guard.
template <bool Scale_max = true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0>& tensor,
                                        Tensor<Engine1, Layout1> const& max,
                                        const float scale_log2) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const float max_scaled =
        max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale_log2 : 1.f);
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale_log2 - max_scaled);
    }
  }
}

template <typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)),
                     make_layout(get<0>(get<0>(l)), get<2>(l)));
}

template <typename MMA_traits, typename Layout>
inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout) {
  using X = Underscore;
  static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
  static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
  constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
  static_assert(mma_shape_K == 8 || mma_shape_K == 16);
  constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
  auto l = logical_divide(rowcol_layout,
                          Shape<X, Shape<X, Int<MMA_N_divisor>>>{});
  return make_layout(
      make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)),
                  get<0>(get<1>(get<1>(l)))),
      get<1>(get<0>(l)), get<1>(get<1>(get<1>(l))));
}

// Online softmax + output rescaling. scale_log2 = softmax_scale * log2(e).
template <bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0& scores, Tensor1& scores_max,
                                         Tensor1& scores_sum, Tensor2& acc_o,
                                         float softmax_scale_log2) {
  if (Is_first) {
    reduce_max</*zero_init=*/true>(scores, scores_max);
    scale_apply_exp2(scores, scores_max, softmax_scale_log2);
    reduce_sum(scores, scores_sum);
  } else {
    Tensor scores_max_prev = make_fragment_like(scores_max);
    cute::copy(scores_max, scores_max_prev);
    reduce_max</*zero_init=*/false>(scores, scores_max);
    Tensor acc_o_rowcol =
        make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
    for (int mi = 0; mi < size(scores_max); ++mi) {
      const float scores_max_cur = scores_max(mi);
      const float scores_scale =
          exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
      scores_sum(mi) *= scores_scale;
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale;
      }
    }
    scale_apply_exp2(scores, scores_max, softmax_scale_log2);
    Tensor scores_sum_cur = make_fragment_like(scores_sum);
    reduce_sum(scores, scores_sum_cur);
#pragma unroll
    for (int mi = 0; mi < size(scores_sum); ++mi) {
      scores_sum(mi) += scores_sum_cur(mi);
    }
  }
}

// ---------------------------------------------------------------------------
// Shared storage.
// ---------------------------------------------------------------------------
template <class ElementType, class SmemLayoutQ, class SmemLayoutK,
          class SmemLayoutV>
struct SharedStorage {
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

// ---------------------------------------------------------------------------
// Per-(sequence, head, q-tile) body. Shared by the fixed-length kernel and the
// fused varlen kernel.
// ---------------------------------------------------------------------------
template <typename Kernel_traits, bool Is_causal, typename Params>
__device__ void flash_attention_v2_body(const Params& params,
                                        const int m_block,
                                        const int bidb,
                                        const int qh,
                                        const int kvh) {
  using namespace cute;

  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using TiledMMA = typename Kernel_traits::TiledMma;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
  using SmemLayoutVtNoSwizzle =
      typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  extern __shared__ char smem_[];
  using SharedStorageT =
      SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
  SharedStorageT& shared_storage =
      *reinterpret_cast<SharedStorageT*>(smem_);

  const Element* q_ptr = reinterpret_cast<const Element*>(params.q_ptr) +
                         bidb * params.q_batch_stride +
                         qh * params.q_head_stride;
  const Element* k_ptr = reinterpret_cast<const Element*>(params.k_ptr) +
                         bidb * params.k_batch_stride +
                         kvh * params.k_head_stride;
  const Element* v_ptr = reinterpret_cast<const Element*>(params.v_ptr) +
                         bidb * params.v_batch_stride +
                         kvh * params.v_head_stride;

  Tensor Q = make_tensor(make_gmem_ptr(q_ptr),
                         make_shape(params.q_seqlen, Int<kHeadDim>{}),
                         make_stride(params.q_row_stride, Int<1>{}));

  const auto tile_m = make_tile(Int<kBlockM>{}, Int<kHeadDim>{});

  // A KV tile is either contiguous (block_table == nullptr) or one page of a
  // paged KV cache. In the paged case kBlockN must divide page_block_size so a
  // tile never crosses a page boundary.
  auto kv_tile_ptr = [&](const Element* base, int nbi, int64_t row_stride,
                         int64_t page_stride) -> const Element* {
    if (params.block_table == nullptr) {
      return base + int64_t(nbi) * kBlockN * row_stride;
    }
    const int logical = nbi * kBlockN;
    const int page_idx = logical / params.page_block_size;
    const int page_off = logical - page_idx * params.page_block_size;
    const int physical = params.block_table[page_idx];
    return base + int64_t(physical) * page_stride +
           int64_t(page_off) * row_stride;
  };

  Tensor gQ = local_tile(Q, tile_m, make_coord(m_block, _));
  Tensor gK = make_tensor(
      make_gmem_ptr(kv_tile_ptr(k_ptr, 0, params.k_row_stride,
                                params.k_batch_stride)),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_stride(params.k_row_stride, Int<1>{}));
  Tensor gV = make_tensor(
      make_gmem_ptr(kv_tile_ptr(v_ptr, 0, params.v_row_stride,
                                params.v_batch_stride)),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_stride(params.v_row_stride, Int<1>{}));

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()),
                          SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()),
                          SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()),
                          SmemLayoutV{});
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()),
                           SmemLayoutVt{});
  Tensor sVtNoSwizzle =
      make_tensor(make_smem_ptr(shared_storage.smem_v.data()),
                  SmemLayoutVtNoSwizzle{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(threadIdx.x);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  // Identity tensors are used only for M/N-bound predication.
  Tensor cQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
  Tensor cKV = make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(threadIdx.x);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{},
                        tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  const int m_remaining = params.q_seqlen - m_block * kBlockM;
  if (m_remaining >= kBlockM) {
    copy_mn</*Is_even_MN=*/true, /*Clear=*/false>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, 0);
  } else if (m_remaining % 8 == 0) {
    copy_mn</*Is_even_MN=*/false, /*Clear=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, m_remaining);
  } else {
    copy_mn_elementwise</*Clear=*/true>(tQgQ, tQsQ, tQcQ, m_remaining);
  }

  const int k_remaining_0 = params.k_seqlen;
  if (k_remaining_0 >= kBlockN) {
    copy_mn</*Is_even_MN=*/true, /*Clear=*/false>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, 0);
  } else if (k_remaining_0 % 8 == 0) {
    copy_mn</*Is_even_MN=*/false, /*Clear=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, k_remaining_0);
  } else {
    copy_mn_elementwise</*Clear=*/true>(tKgK, tKsK, tKVcKV, k_remaining_0);
  }
  cute::cp_async_fence();

  Tensor rAccOut = partition_fragment_C(
      tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(rAccOut);

  const int n_block_max =
      Is_causal
          ? min(cute::ceil_div(params.k_seqlen, kBlockN),
                cute::ceil_div((m_block + 1) * kBlockM, kBlockN))
          : cute::ceil_div(params.k_seqlen, kBlockN);

  Tensor scores_max =
      make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});
  Tensor scores_sum = make_fragment_like(scores_max);

  const float scale_log2 = params.softmax_scale * float(M_LOG2E);

  for (int nbi = 0; nbi < n_block_max; ++nbi) {
    auto rAccScore = partition_fragment_C(
        tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
    clear(rAccScore);

    cp_async_wait<0>();
    __syncthreads();

    // Load V for this KV block while the QK GEMM runs.
    gV = make_tensor(
        make_gmem_ptr(kv_tile_ptr(v_ptr, nbi, params.v_row_stride,
                                  params.v_batch_stride)),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_stride(params.v_row_stride, Int<1>{}));
    tVgV = gmem_thr_copy_QKV.partition_S(gV);
    const int kv_remaining = params.k_seqlen - nbi * kBlockN;
    if (kv_remaining >= kBlockN) {
      copy_mn</*Is_even_MN=*/true, /*Clear=*/false>(
          gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, 0);
    } else if (kv_remaining % 8 == 0) {
      copy_mn</*Is_even_MN=*/false, /*Clear=*/true>(
          gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, kv_remaining);
    } else {
      copy_mn_elementwise</*Clear=*/true>(tVgV, tVsV, tKVcKV, kv_remaining);
    }
    cute::cp_async_fence();

    gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
              smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q,
              smem_thr_copy_K);

    Tensor scores =
        make_tensor(rAccScore.data(),
                    convert_layout_acc_rowcol(rAccScore.layout()));
    apply_mask<Is_causal, kBlockM, kBlockN, kNWarps>(
        scores, m_block, nbi, params.k_seqlen);

    cp_async_wait<0>();
    __syncthreads();

    if (nbi + 1 < n_block_max) {
      gK = make_tensor(
          make_gmem_ptr(kv_tile_ptr(k_ptr, nbi + 1, params.k_row_stride,
                                    params.k_batch_stride)),
          Shape<Int<kBlockN>, Int<kHeadDim>>{},
          make_stride(params.k_row_stride, Int<1>{}));
      tKgK = gmem_thr_copy_QKV.partition_S(gK);
      const int k_next_remaining = params.k_seqlen - (nbi + 1) * kBlockN;
      if (k_next_remaining >= kBlockN) {
        copy_mn</*Is_even_MN=*/true, /*Clear=*/false>(
            gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, 0);
      } else if (k_next_remaining % 8 == 0) {
        copy_mn</*Is_even_MN=*/false, /*Clear=*/true>(
            gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, k_next_remaining);
      } else {
        copy_mn_elementwise</*Clear=*/true>(tKgK, tKsK, tKVcKV,
                                            k_next_remaining);
      }
      cute::cp_async_fence();
    }

    if (nbi == 0) {
      softmax_rescale_o</*Is_first=*/true>(scores, scores_max, scores_sum,
                                          rAccOut, scale_log2);
    } else {
      softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum,
                                           rAccOut, scale_log2);
    }

    Tensor rP = convert_type_f32<Element>(rAccScore);
    Tensor tOrP =
        make_tensor(rP.data(),
                    convert_layout_rowcol_Aregs<TiledMMA>(scores.layout()));
    gemm_A_in_regs(rAccOut, tOrP, tOrVt, tOsVt, tiled_mma,
                   smem_tiled_copy_V, smem_thr_copy_V);
  }

  // Final normalization and optional LSE.
  Tensor acc_o_rowcol =
      make_tensor(rAccOut.data(), convert_layout_acc_rowcol(rAccOut.layout()));
  Tensor lse = make_fragment_like(scores_sum);
#pragma unroll
  for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    const float sum = scores_sum(mi);
    const bool empty = (sum == 0.f) || (sum != sum);
    const float inv_sum = empty ? 0.f : 1.f / sum;
    lse(mi) = empty ? INFINITY
                    : scores_max(mi) * params.softmax_scale + __logf(sum);
#pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
      acc_o_rowcol(mi, ni) *= inv_sum;
    }
  }

  Tensor rO = convert_type_f32<Element>(rAccOut);
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

  auto smem_tiled_copy_O = make_tiled_copy_C(
      typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(threadIdx.x);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor O = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element*>(params.out_ptr) +
                    bidb * params.o_batch_stride + qh * params.o_head_stride),
      make_shape(params.q_seqlen, Int<kHeadDim>{}),
      make_stride(params.o_row_stride, Int<1>{}));
  Tensor gO = local_tile(O, tile_m, make_coord(m_block, _));

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(threadIdx.x);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));
  Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
  Tensor tOcO = gmem_thr_copy_O.partition_S(cO);

  __syncthreads();
  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
  if (m_remaining >= kBlockM) {
    copy_mn</*Is_even_MN=*/true, /*Clear=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, 0);
  } else if (m_remaining % 8 == 0) {
    copy_mn</*Is_even_MN=*/false, /*Clear=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, m_remaining);
  } else {
    copy_mn_elementwise</*Clear=*/false>(tOrO, tOgO, tOcO, m_remaining);
  }

  if (params.lse_ptr != nullptr) {
    Tensor LSE = make_tensor(
        make_gmem_ptr(reinterpret_cast<float*>(params.lse_ptr) +
                      bidb * params.lse_batch_stride +
                      qh * params.lse_head_stride),
        make_shape(params.q_seqlen), make_stride(params.lse_row_stride));
    Tensor gLSE = local_tile(LSE, make_tile(Int<kBlockM>{}),
                             make_coord(m_block));
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    Tensor taccOcO_row =
        logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
      for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccOcO_row(mi));
        if (row < m_remaining) {
          gLSE(row) = lse(mi);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Build a device-side tile -> (sequence, m_block) map. One thread per sequence;
// atomicAdd assigns each sequence a contiguous range of tiles. The attention
// kernel launches with an upper-bound grid and exits extra CTAs, so the host
// never needs to read cu_seqlens back.
// ---------------------------------------------------------------------------
__global__ void build_tile_map_kernel(const int* cu_q, const int* cu_k,
                                      const int batch, const int block_m,
                                      int* tile_seq, int* tile_m,
                                      int* total_tiles, const int max_tiles) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;

  const int qlen = cu_q[b + 1] - cu_q[b];
  const int klen = cu_k[b + 1] - cu_k[b];
  (void)klen;
  const int ntiles = (qlen + block_m - 1) / block_m;
  const int base = atomicAdd(total_tiles, ntiles);
  if (base + ntiles > max_tiles) return;
  for (int t = 0; t < ntiles; ++t) {
    tile_seq[base + t] = b;
    tile_m[base + t] = t;
  }
}

// ---------------------------------------------------------------------------
// Fixed-length kernel: grid = (q tiles, batch * q_head).
// ---------------------------------------------------------------------------
template <typename Kernel_traits, bool Is_causal, typename Params>
__global__ void flash_attention_v2_kernel(const Params params) {
  const int m_block = blockIdx.x;
  const int bhid = blockIdx.y;
  const int bidb = bhid / params.q_head;
  const int qh = bhid - bidb * params.q_head;
  const int kvh = qh / (params.q_head / params.k_head);

  if (m_block * Kernel_traits::kBlockM >= params.q_seqlen) return;

  flash_attention_v2_body<Kernel_traits, Is_causal>(params, m_block, bidb, qh,
                                                    kvh);
}

// ---------------------------------------------------------------------------
// Fused varlen kernel: grid = (total_q_tiles, q_head). Every sequence's Q
// tiles live in the same grid, so the hardware scheduler can backfill idle
// SMs across sequence boundaries. cu_seqlens and the tile->sequence map stay
// on the device.
// ---------------------------------------------------------------------------
template <typename Kernel_traits, bool Is_causal, typename Params>
__global__ void flash_attention_v2_varlen_kernel(const Params params) {
  using Element = typename Kernel_traits::Element;

  const int tile_id = blockIdx.x;
  const int qh = blockIdx.y;
  if (tile_id >= *params.total_q_tiles_dev) return;

  const int b = params.tile_to_seq[tile_id];
  const int m_block = params.tile_to_mblock[tile_id];

  const int q_start = params.cu_seqlens_q[b];
  const int q_end = params.cu_seqlens_q[b + 1];
  const int k_start = params.cu_seqlens_k[b];
  const int k_end = params.cu_seqlens_k[b + 1];
  const int qlen = q_end - q_start;
  const int klen = k_end - k_start;
  if (qlen <= 0) return;

  // Build the per-CTA parameter view once in shared memory instead of copying
  // the whole Params struct per thread (which can spill to local memory).
  __shared__ Params s_params;
  if (threadIdx.x == 0) {
    s_params = params;
    s_params.q_ptr = reinterpret_cast<const Element*>(params.q_ptr) +
                     int64_t(q_start) * params.q_row_stride;
    s_params.k_ptr = reinterpret_cast<const Element*>(params.k_ptr) +
                     int64_t(k_start) * params.k_row_stride;
    s_params.v_ptr = reinterpret_cast<const Element*>(params.v_ptr) +
                     int64_t(k_start) * params.v_row_stride;
    s_params.out_ptr = reinterpret_cast<Element*>(params.out_ptr) +
                       int64_t(q_start) * params.o_row_stride;
    s_params.lse_ptr = reinterpret_cast<float*>(params.lse_ptr) +
                       int64_t(q_start) * params.lse_row_stride;
    s_params.q_seqlen = qlen;
    s_params.k_seqlen = klen;
    s_params.q_batch_stride = 0;
    s_params.k_batch_stride = 0;
    s_params.v_batch_stride = 0;
    s_params.o_batch_stride = 0;
    s_params.lse_batch_stride = 0;
  }
  __syncthreads();

  const int kvh = qh / (params.q_head / params.k_head);
  flash_attention_v2_body<Kernel_traits, Is_causal>(s_params, m_block, 0, qh,
                                                    kvh);
}

// ---------------------------------------------------------------------------
// Paged-KV decode kernel. Q has one token per sequence; K/V live in a paged
// cache addressed by block_table. There is no causal mask because the cache
// only contains past tokens. grid = (1, batch * q_head).
// ---------------------------------------------------------------------------
template <typename Kernel_traits, typename Params>
__global__ void flash_attention_decode_kernel(const Params params) {
  using Element = typename Kernel_traits::Element;

  const int bhid = blockIdx.y;
  const int bidb = bhid / params.q_head;
  const int qh = bhid - bidb * params.q_head;
  const int cache_len = params.cache_seqlens[bidb];
  if (cache_len <= 0) return;

  const int kvh = qh / (params.q_head / params.k_head);

  __shared__ Params s_params;
  if (threadIdx.x == 0) {
    s_params = params;
    s_params.q_ptr = reinterpret_cast<const Element*>(params.q_ptr) +
                     int64_t(bidb) * params.q_batch_stride +
                     int64_t(qh) * params.q_head_stride;
    s_params.k_ptr = reinterpret_cast<const Element*>(params.k_ptr) +
                     int64_t(kvh) * params.k_head_stride;
    s_params.v_ptr = reinterpret_cast<const Element*>(params.v_ptr) +
                     int64_t(kvh) * params.v_head_stride;
    s_params.out_ptr = reinterpret_cast<Element*>(params.out_ptr) +
                       int64_t(bidb) * params.o_batch_stride +
                       int64_t(qh) * params.o_head_stride;
    s_params.lse_ptr = reinterpret_cast<float*>(params.lse_ptr) +
                       int64_t(bidb) * params.lse_batch_stride +
                       int64_t(qh) * params.lse_head_stride;
    s_params.block_table =
        params.block_table + int64_t(bidb) * params.block_table_batch_stride;

    s_params.q_seqlen = 1;
    s_params.k_seqlen = cache_len;
    s_params.q_batch_stride = 0;
    s_params.o_batch_stride = 0;
    s_params.lse_batch_stride = 0;
  }
  __syncthreads();

  flash_attention_v2_body<Kernel_traits, /*Is_causal=*/false>(
      s_params, /*m_block=*/0, /*bidb=*/0, /*qh=*/0, /*kvh=*/0);
}

}  // namespace flash_lab
