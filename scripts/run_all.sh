#!/usr/bin/env bash
set -euo pipefail
make build
make ext
make test
make bench
make bench-causal
