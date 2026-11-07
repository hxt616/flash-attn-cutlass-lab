#!/usr/bin/env bash
# Create a reproducible Python environment for the PyTorch extension and tests.
set -euo pipefail

PYTHON=${PYTHON:-python3.10}
TORCH_INDEX=${TORCH_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
PKG_INDEX=${PKG_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; install it or use a normal venv:" >&2
  echo "  ${PYTHON} -m venv .venv && .venv/bin/pip install torch ..." >&2
  exit 1
fi

uv venv .venv --python "${PYTHON}"
uv pip install --python .venv/bin/python "torch==2.4.1" --index-url "${TORCH_INDEX}"
uv pip install --python .venv/bin/python \
  numpy pytest matplotlib pandas ninja packaging setuptools wheel \
  --index-url "${PKG_INDEX}"

echo
echo "Environment ready. Next:"
echo "  make ext && make test && make bench && make bench-causal"
