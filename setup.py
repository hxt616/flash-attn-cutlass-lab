import os
from pathlib import Path

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent

# The kernels use SM80+ CuTe atoms. Default to the local RTX 40 (sm_89);
# override with TORCH_CUDA_ARCH_LIST for other GPUs.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

ext_modules = [
    CUDAExtension(
        name="attention_cutlass",
        sources=[
            "csrc/flash_api.cpp",
            "csrc/flash_fwd_kernel.cu",
        ],
        include_dirs=[
            this_dir / "include",
            this_dir / "deps/cutlass/include",
            this_dir / "deps/cutlass/tools/util/include",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-lineinfo",
                "--expt-relaxed-constexpr",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
        },
    )
]

setup(
    name="flash_attn_cutlass_lab",
    version="0.1.0",
    description="FlashAttention-2 forward kernel implemented with CUTLASS CuTe",
    packages=["flash_attn_lab"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch", "packaging", "ninja"],
)
