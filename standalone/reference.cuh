#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include <cutlass/numeric_types.h>

// Simple, obviously-correct reference implementation used by the standalone
// test. One CUDA block per (batch, q_head, query row). Scores are materialized
// in shared memory, so this is O(S^2) memory but easy to audit.
template <typename Element>
__global__ void reference_attention_kernel(const Element* Q, const Element* K,
                                           const Element* V, float* O,
                                           int batch, int q_head, int k_head,
                                           int q_seqlen, int k_seqlen, int dim,
                                           bool causal, float softmax_scale) {
  const int q = blockIdx.x;
  const int bhid = blockIdx.y;
  const int b = bhid / q_head;
  const int qh = bhid - b * q_head;
  const int kv_ratio = q_head / k_head;
  const int kh = qh / kv_ratio;

  const Element* qp =
      Q + static_cast<int64_t>(b) * q_seqlen * q_head * dim + qh * dim;
  const Element* kp =
      K + static_cast<int64_t>(b) * k_seqlen * k_head * dim + kh * dim;
  const Element* vp =
      V + static_cast<int64_t>(b) * k_seqlen * k_head * dim + kh * dim;
  float* op =
      O + static_cast<int64_t>(b) * q_seqlen * q_head * dim + qh * dim;

  const int q_row_stride = q_head * dim;
  const int k_row_stride = k_head * dim;

  extern __shared__ float smem[];
  float* scores = smem;
  float* q_s = scores + k_seqlen;
  float* stats = q_s + dim;  // stats[0] = max, stats[1] = 1 / sum

  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int nwarps = blockDim.x / 32;

  for (int d = tid; d < dim; d += blockDim.x) {
    q_s[d] = static_cast<float>(qp[q * q_row_stride + d]);
  }
  __syncthreads();

  const int kend = causal ? min(q + 1, k_seqlen) : k_seqlen;

  for (int kk = warp; kk < kend; kk += nwarps) {
    float sum = 0.f;
    for (int d = lane; d < dim; d += 32) {
      sum += q_s[d] * static_cast<float>(kp[kk * k_row_stride + d]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, off);
    }
    if (lane == 0) {
      scores[kk] = sum * softmax_scale;
    }
  }
  __syncthreads();

  if (tid == 0) {
    float m = -INFINITY;
    for (int kk = 0; kk < kend; ++kk) {
      m = fmaxf(m, scores[kk]);
    }
    float l = 0.f;
    for (int kk = 0; kk < kend; ++kk) {
      const float e = (m == -INFINITY) ? 0.f : expf(scores[kk] - m);
      scores[kk] = e;
      l += e;
    }
    stats[0] = m;
    stats[1] = (l > 0.f && l == l) ? (1.f / l) : 0.f;
  }
  __syncthreads();

  const float inv_sum = stats[1];
  for (int d = tid; d < dim; d += blockDim.x) {
    float acc = 0.f;
    for (int kk = 0; kk < kend; ++kk) {
      acc += scores[kk] * static_cast<float>(vp[kk * k_row_stride + d]);
    }
    op[q * q_row_stride + d] = acc * inv_sum;
  }
}
