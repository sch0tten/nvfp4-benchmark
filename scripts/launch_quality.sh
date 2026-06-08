#!/usr/bin/env bash
# Quality matrix with the correct CUDA-13 toolchain + env loaded.
# Extra args pass through to run_quality.py (e.g. --only <arm>, --out <dir>).
#   bash scripts/launch_quality.sh
set -uo pipefail
cd "$(dirname "$0")/.."                       # repo root
: "${VENV:=.venv}"
source cuda_env.sh                            # CUDA_HOME, LIBRARY_PATH (needs scripts/setup_env.sh once)
[ -f .env ] && source .env                    # HF_TOKEN for gated Gemma (see .env.example)
export HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_ALLOW_CODE_EVAL=1 \
       TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
exec "$VENV/bin/python" scripts/run_quality.py --config configs/models.yaml \
    --out results/quality --logs logs --cache cache/lm_eval "$@"
