# Dependencies

- `deps/cutlass/` is a **vendored** CUTLASS v3.4.1 header tree (BSD-3).
  The repository builds offline without submodules or external checkouts.
  See `deps/cutlass/VENDORED.md`.

No other external source dependency is required. The Python environment
(`torch`, `pytest`, etc.) is described by `requirements.txt` and
`scripts/setup_env.sh`; it is not vendored because CUDA-enabled PyTorch wheels
are platform-specific and large.
