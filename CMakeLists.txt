cmake_minimum_required(VERSION 3.24)
project(flash_attn_cutlass_lab LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_COLOR_DIAGNOSTICS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Allow overriding the CUTLASS checkout for offline builds.
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/cutlass" CACHE PATH "CUTLASS root")

# CMake pre-populates CMAKE_CUDA_ARCHITECTURES with 52, which makes the
# SM80-only CuTe atoms fall back to SM75 and fail at runtime. Default to the
# local RTX 40 (sm_89); override with -DCMAKE_CUDA_ARCHITECTURES=... if needed.
if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures" FORCE)
endif()

add_executable(flash_attn_lab
  standalone/main.cu
)

target_include_directories(flash_attn_lab PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/standalone
  ${CUTLASS_DIR}/include
  ${CUTLASS_DIR}/tools/util/include
)

target_compile_options(flash_attn_lab PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr>
)

set_target_properties(flash_attn_lab PROPERTIES
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  CUDA_SEPARABLE_COMPILATION OFF
)
