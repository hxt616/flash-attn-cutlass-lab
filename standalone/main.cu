#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cutlass/numeric_types.h>

#include "flash_params.h"
#include "flash_fwd_launch.h"

#include "reference.cuh"

using Element = cutlass::half_t;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,  \
                   __LINE__, cudaGetErrorString(err));                       \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

static double attention_flops(int batch, int head, int seqlen, int dim,
                              bool causal) {
  const double b = batch, h = head, s = seqlen, d = dim;
  if (causal) {
    const double valid = s * (s + 1.0) / 2.0;
    return 4.0 * b * h * valid * d;
  }
  return 4.0 * b * h * s * s * d;
}

struct Options {
  int batch = 2;
  int head = 8;
  int head_k = 8;
  int seqlen = 2048;
  int dim = 64;
  int block_n = 64;
  bool causal = false;
  float scale = 0.f;  // 0 => 1/sqrt(dim)
  int warmup = 20;
  int iters = 200;
  bool check = true;
  bool bench = true;
};

static Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--B") o.batch = std::atoi(next());
    else if (a == "--H") o.head = std::atoi(next());
    else if (a == "--Hk") o.head_k = std::atoi(next());
    else if (a == "--S") o.seqlen = std::atoi(next());
    else if (a == "--D") o.dim = std::atoi(next());
    else if (a == "--bn") o.block_n = std::atoi(next());
    else if (a == "--scale") o.scale = std::atof(next());
    else if (a == "--warmup") o.warmup = std::atoi(next());
    else if (a == "--iters") o.iters = std::atoi(next());
    else if (a == "--causal") o.causal = true;
    else if (a == "--no-check") o.check = false;
    else if (a == "--no-bench") o.bench = false;
    else if (a == "-h" || a == "--help") {
      std::printf(
          "usage: %s [--B n] [--H n] [--Hk n] [--S n] [--D n] [--bn n] "
          "[--causal] [--scale x] [--warmup n] [--iters n] [--no-check] "
          "[--no-bench]\n",
          argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      std::exit(2);
    }
  }
  if (o.scale == 0.f) o.scale = 1.0f / std::sqrt(float(o.dim));
  if (o.head % o.head_k != 0) {
    std::fprintf(stderr, "H must be a multiple of Hk\n");
    std::exit(2);
  }
  return o;
}

template <int kHeadDim, bool Is_causal, int kBlockN>
static int run_case(const Options& o) {
  using Traits = Flash_fwd_kernel_traits<kHeadDim, 64, kBlockN, 4, Element>;

  const int B = o.batch, H = o.head, Hk = o.head_k, S = o.seqlen, D = o.dim;
  const int64_t q_elems = static_cast<int64_t>(B) * S * H * D;
  const int64_t k_elems = static_cast<int64_t>(B) * S * Hk * D;
  const int64_t lse_elems = static_cast<int64_t>(B) * H * S;

  std::vector<Element> h_q(q_elems), h_k(k_elems), h_v(k_elems);
  std::vector<float> h_ref(q_elems);
  std::vector<Element> h_out(q_elems);
  std::vector<float> h_lse(lse_elems, 0.f);

  std::mt19937 rng(1234);
  std::normal_distribution<float> dist(0.f, 0.5f);
  for (auto& x : h_q) x = Element(dist(rng));
  for (auto& x : h_k) x = Element(dist(rng));
  for (auto& x : h_v) x = Element(dist(rng));

  Element *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  float *d_ref = nullptr, *d_lse = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, q_elems * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_k, k_elems * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_v, k_elems * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_o, q_elems * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_ref, q_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse, lse_elems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), k_elems * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), k_elems * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));

  // Reference.
  {
    const int smem = static_cast<int>((S + D + 2) * sizeof(float));
    if (smem >= 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(
          reference_attention_kernel<Element>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    dim3 grid(S, B * H, 1);
    reference_attention_kernel<Element><<<grid, 128, smem>>>(
        d_q, d_k, d_v, d_ref, B, H, Hk, S, S, D, Is_causal, o.scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  Flash_fwd_params p;
  p.q_ptr = d_q;
  p.k_ptr = d_k;
  p.v_ptr = d_v;
  p.out_ptr = d_o;
  p.lse_ptr = nullptr;  // standalone does not consume LSE yet
  p.batch = B;
  p.q_seqlen = S;
  p.k_seqlen = S;
  p.dim = D;
  p.q_head = H;
  p.k_head = Hk;
  p.q_batch_stride = static_cast<int64_t>(S) * H * D;
  p.q_head_stride = D;
  p.q_row_stride = static_cast<int64_t>(H) * D;
  p.k_batch_stride = static_cast<int64_t>(S) * Hk * D;
  p.k_head_stride = D;
  p.k_row_stride = static_cast<int64_t>(Hk) * D;
  p.v_batch_stride = p.k_batch_stride;
  p.v_head_stride = D;
  p.v_row_stride = p.k_row_stride;
  p.o_batch_stride = p.q_batch_stride;
  p.o_head_stride = D;
  p.o_row_stride = p.q_row_stride;
  p.lse_batch_stride = static_cast<int64_t>(H) * S;
  p.lse_head_stride = S;
  p.lse_row_stride = 1;
  p.softmax_scale = o.scale;
  p.is_causal = Is_causal;
  p.is_bf16 = false;

  flash_lab::launch_flash_fwd<Traits, Is_causal>(p, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_o, q_elems * sizeof(Element),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, q_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int rc = 0;
  if (o.check) {
    float max_abs = 0.f, mean_abs = 0.f, max_rel = 0.f;
    int64_t bad = -1;
    for (int64_t i = 0; i < q_elems; ++i) {
      const float diff = std::fabs(float(h_out[i]) - h_ref[i]);
      max_abs = std::fmax(max_abs, diff);
      mean_abs += diff;
      const float denom = std::fmax(1e-3f, std::fabs(h_ref[i]));
      max_rel = std::fmax(max_rel, diff / denom);
      if (bad < 0 && diff > 2e-2f) bad = i;
    }
    mean_abs /= float(q_elems);
    std::printf(
        "check: B=%d H=%d Hk=%d S=%d D=%d bn=%d causal=%d scale=%.6f | "
        "max_abs=%.6f mean_abs=%.6f max_rel=%.4f %s\n",
        B, H, Hk, S, D, o.block_n, int(Is_causal), o.scale, max_abs, mean_abs,
        max_rel, bad < 0 ? "PASS" : "FAIL");
    if (bad >= 0) {
      std::printf("first mismatch at %lld: out=%f ref=%f\n",
                  static_cast<long long>(bad), float(h_out[bad]), h_ref[bad]);
      rc = 1;
    }
  }

  if (o.bench) {
    for (int i = 0; i < o.warmup; ++i) {
      flash_lab::launch_flash_fwd<Traits, Is_causal>(p, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < o.iters; ++i) {
      flash_lab::launch_flash_fwd<Traits, Is_causal>(p, 0);
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const double per_iter = ms / o.iters;
    const double flops = attention_flops(B, H, S, D, Is_causal);
    const double tflops = flops / (per_iter * 1e-3) / 1e12;
    std::printf("bench: %.4f ms/iter, %.2f TFLOPS\n", per_iter, tflops);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));
  CUDA_CHECK(cudaFree(d_ref));
  CUDA_CHECK(cudaFree(d_lse));
  return rc;
}

template <int kHeadDim>
static int run_dispatch(const Options& o) {
  if (o.block_n == 128) {
    return o.causal ? run_case<kHeadDim, true, 128>(o)
                    : run_case<kHeadDim, false, 128>(o);
  }
  if (o.block_n != 64) {
    std::fprintf(stderr, "block_n must be 64 or 128\n");
    return 2;
  }
  return o.causal ? run_case<kHeadDim, true, 64>(o)
                  : run_case<kHeadDim, false, 64>(o);
}

int main(int argc, char** argv) {
  const Options o = parse_args(argc, argv);

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s (cc %d.%d, smem/SM %zu)\n", prop.name, prop.major,
              prop.minor, prop.sharedMemPerMultiprocessor);

  int rc = 0;
  if (o.dim == 32) rc = run_dispatch<32>(o);
  else if (o.dim == 64) rc = run_dispatch<64>(o);
  else if (o.dim == 128) rc = run_dispatch<128>(o);
  else {
    std::fprintf(stderr, "standalone supports D in {32, 64, 128}\n");
    return 2;
  }
  return rc;
}
