# CUTLASS (vendored)

This directory contains the CUTLASS **v3.4.1** headers needed by the build:

```text
deps/cutlass/include/                  # cute + cutlass headers
deps/cutlass/tools/util/include/       # cutlass/util helpers
```

It is vendored so the repository is self-contained: no network access, git
submodule initialization, or external checkout is required. The build only uses
these two include directories (see `CMakeLists.txt` and `setup.py`).

- Source: https://github.com/NVIDIA/cutlass
- Version: v3.4.1 (per `CHANGELOG.md`)
- License: BSD-3-Clause; see `LICENSE.txt` here and `LICENSES/CUTLASS.txt`.

To update, replace these two directories with the corresponding directories
from a newer CUTLASS release, update `CHANGELOG.md`/this note, and re-run the
test suite. The CuTe API is not stable across major/minor releases; the kernel
was validated against v3.4.1.
