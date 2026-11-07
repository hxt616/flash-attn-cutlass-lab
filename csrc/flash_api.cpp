#include <torch/extension.h>
#include <torch/python.h>

#include "attention_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_attention_v2_cutlass", &flash_attention_v2_cutlass,
        "FlashAttention-2 forward (CUDA/CuTe), returns (out, lse)",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("is_causal") = false, py::arg("softmax_scale") = 1.0);
  m.def("flash_attention_varlen_cutlass", &flash_attention_varlen_cutlass,
        "FlashAttention-2 varlen forward (CUDA/CuTe), returns (out, lse)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"), py::arg("is_causal") = false,
        py::arg("softmax_scale") = 1.0);
  m.def("flash_attention_decode_cutlass", &flash_attention_decode_cutlass,
        "Paged-KV decode attention (CUDA/CuTe), returns (out, lse)",
        py::arg("q"), py::arg("k_cache"), py::arg("v_cache"),
        py::arg("block_table"), py::arg("cache_seqlens"),
        py::arg("softmax_scale") = 1.0);
}
