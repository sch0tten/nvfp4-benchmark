#!/usr/bin/env bash
# Single-stream throughput matrix with the correct CUDA-13 toolchain + env.
# Extra args pass through to run_throughput.py (e.g. --only <arm>, --out <dir>).
#   bash scripts/launch_throughput.sh
set -uo pipefail
cd "$(dirname "$0")/.."                       # repo root
: "${VENV:=.venv}"
source cuda_env.sh
[ -f .env ] && source .env
export HF_TOKEN HUGGING_FACE_HUB_TOKEN TOKENIZERS_PARALLELISM=false
exec "$VENV/bin/python" scripts/run_throughput.py --config configs/models.yaml \
    --out results/throughput --logs logs "$@"
