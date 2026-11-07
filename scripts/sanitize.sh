#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-build/flash_attn_lab}
if [[ ! -x "${BIN}" ]]; then
  echo "build the standalone binary first: make build" >&2
  exit 1
fi

run_case() {
  local tool=$1; shift
  echo "=== compute-sanitizer --tool ${tool}: $* ==="
  compute-sanitizer --tool "${tool}" "${BIN}" "$@" --no-bench
}

for tool in memcheck racecheck initcheck synccheck; do
  run_case "${tool}" --B 1 --H 2 --Hk 2 --S 65 --D 64 --causal
  run_case "${tool}" --B 1 --H 2 --Hk 2 --S 97 --D 64
  run_case "${tool}" --B 1 --H 2 --Hk 2 --S 129 --D 128 --causal
done
