#!/usr/bin/env bash
# Rescue arm 12 (qwen-MoE-NVFP4). vLLM 0.22.1 cannot construct the NVFP4-quantized
# lm_head this NVIDIA checkpoint ships, so we dequantize lm_head to BF16 (every
# expert and attention layer stays bit-exact NVFP4), then run the same quality +
# throughput protocol as the other arms. Writes the standard result JSONs.
#   bash scripts/launch_arm12.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${VENV:=.venv}"
source cuda_env.sh
[ -f .env ] && source .env
export HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_ALLOW_CODE_EVAL=1 \
       TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY="$VENV/bin/python"
echo ">>> [1/3] dequantizing lm_head to BF16 (swizzle=False)"; "$PY" scripts/dequant_lmhead.py false
echo ">>> [2/3] quality (5 tasks)";                            "$PY" scripts/run_arm12.py
echo ">>> [3/3] single-stream throughput";                    "$PY" scripts/throughput_arm12.py
