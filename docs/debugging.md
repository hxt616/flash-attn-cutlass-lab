# Debugging log

Each of these was reproduced and fixed while turning the upstream learning
skeleton into a testable project. They make good interview stories because each
one required narrowing the failure to a specific assumption.

## 1. CMake compiled for `sm_52`, runtime used SM75 atoms

Symptom: `cudaErrorLaunchFailure` / illegal instruction inside
`mma_traits.hpp`, and cuda-gdb showed `SM75_16x8x8_F32F16F16F32_TN` in a kernel
that should have used the SM80 atom.

Cause: `CMAKE_CUDA_ARCHITECTURES` is pre-populated by CMake with `52`. My first
`if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` did not override it, so nvcc emitted
`compute_52`. `kernel_traits.h` selects the SM75 atom when
`__CUDA_ARCH__ < 800`.

Fix: force the default to 89 when the cache value is missing or still 52, and
document the override. This is a classic "it compiles but is the wrong binary"
bug.

## 2. The standalone reference never computed attention

Symptom: the upstream `all_close` compared against uninitialized memory.

Cause: in `self_attention_run`, the `row_softmax` and `naive_pv` calls were
commented out; only `QK` and the causal mask ran, and `d_O` was never written.

Fix: independent reference kernel (`standalone/reference.cuh`) that computes
`QK -> mask -> max -> exp -> sum -> PV`, with an explicit
`cudaDeviceSynchronize` after it so the flash kernel failure cannot be confused
with a reference failure.

## 3. `exp2` without `log2(e)`

Symptom: a single-block attention result was close but not equal to the
reference; the error was consistent with using base 2 instead of base e.

Cause: upstream standalone called `exp2f(x * softmax_scale - max * softmax_scale)`.
The correct form is `exp2f(x * scale * log2e - max * scale * log2e)`.

Fix: `scale_log2 = softmax_scale * M_LOG2E` is computed once and threaded
through the first-block and rescale paths. LSE uses the natural-log form
`max * scale + log(sum)`.

## 4. Ragged 8-row copy tiles wrote past the sequence

Symptom: varlen segfaulted on the last sequence; fixed-length S=97 produced
wrong values in adjacent rows.

Cause: `copy_mn` predicates whole 8-row copy tiles. For `S=97`, the last tile
starts at row 96, is partially valid, and the tile-level predicate still copied
and stored 8 rows. In a packed varlen buffer those rows belong to the next
sequence or beyond the allocation.

Fix: `copy_mn_elementwise` checks the actual `(row, col)` coordinate for the
ragged tail and either copies or clears each element. Full tiles keep the
vectorized path.

## 5. `cu_seqlens` was dereferenced on the host

Symptom: host SIGSEGV inside the varlen wrapper before any kernel launched.

Cause: `cu_seqlens_q.data_ptr<int>()` returns a device pointer, and the host
loop read `cu_q[b]` directly.

Fix: copy the prefix sums to CPU (`cu_seqlens_q.to(torch::kCPU)`) before the
host loop. This is explicit about the cost of the current per-sequence varlen
implementation; a fused varlen kernel would keep them on device.

## 6. Host buffer type mismatch hid the first correctness result

Symptom: the kernel output looked like all near-zero garbage.

Cause: the standalone allocated `std::vector<float> h_out` but `cudaMemcpy`'d
`half` data into it, then read it as `float`.

Fix: `std::vector<Element> h_out` and `float(h_out[i])` when comparing.

## 7. Reference layout mismatch for multi-head

Symptom: H=1 passed, H=2 failed; the first head's first row sometimes matched.

Cause: the reference kernel assumed `(B, H, S, D)` contiguous, while the kernel
and host buffers used `(B, S, H, D)`. For H=1 the layouts coincide.

Fix: reference pointer math uses `q_head * D` as the row stride and
`seqlen * q_head * D` as the batch stride.

## 8. Device `printf` in the hot kernel

Symptom: adding stage prints inside the kernel produced a "Warp Out-of-range
Address" in `SM75_U32x4_LDSM_N`; removing the prints made the same code pass.

Cause: not fully root-caused, but the prints perturbed code generation /
register allocation around the fused MMA loop under nvcc 12.4 `-O3`. It was a
useful reminder to keep hot kernels free of device `printf` and to dump
intermediates through a debug buffer instead.

## Reproduce the fixed test surface

```bash
make build
./build/flash_attn_lab --B 1 --H 2 --Hk 2 --S 97  --D 64 --causal
./build/flash_attn_lab --B 1 --H 2 --Hk 2 --S 129 --D 128 --causal
make ext
make test
make sanitize
```
