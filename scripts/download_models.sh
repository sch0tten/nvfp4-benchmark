#!/usr/bin/env bash
# ============================================================================
# Reproducible, pinned downloader for the 16-arm NVFP4 benchmark matrix.
#
#   4 "seed" models x 4 quantization formats (BF16, FP8, INT4-AWQ, NVFP4).
#   Every artifact is pinned to an exact Hugging Face commit SHA, so this
#   downloads bit-for-bit what the paper measured — it does NOT rely on any
#   pre-existing cache.
#
# Requirements:
#   - `hf` CLI (pip install "huggingface_hub[hf_transfer]")
#   - HF_TOKEN with the google/gemma-4 license accepted (Gemma repos are gated;
#     the NVIDIA NVFP4 repos and Qwen repos are open).
#   - ~600-700 GB free disk.
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/download_models.sh
# ============================================================================
set -uo pipefail
: "${HF_TOKEN:?Set HF_TOKEN to a token with the google/gemma-4 license accepted}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1
command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' CLI not found. pip install 'huggingface_hub[hf_transfer]'"; exit 1; }

# repo <TAB> pinned_commit_sha   (verified 2026-06-07)
read -r -d '' MATRIX <<'EOF'
Qwen/Qwen3.6-27B	6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
Qwen/Qwen3.6-27B-FP8	e89b16ebf1988b3d6befa7de50abc2d76f26eb09
QuantTrio/Qwen3.6-27B-AWQ	9b507bdc9afafb87b7898700cc2a591aa6639461
unsloth/Qwen3.6-27B-NVFP4	890bdef7a42feba6d83b6e17a03315c694112f2a
google/gemma-4-31B-it	3548789868c5356dbf307c98e6f609007b82b3eb
RedHatAI/gemma-4-31B-it-FP8-block	f676bf1357a9d27a77932dd4bf19d619724e74f6
QuantTrio/gemma-4-31B-it-AWQ	200d06ee83eb49a03c6e3120dbf7b09191eb1539
nvidia/Gemma-4-31B-IT-NVFP4	e5ef03afa233c35cb000323ff098d4291e1dd07c
Qwen/Qwen3.6-35B-A3B	995ad96eacd98c81ed38be0c5b274b04031597b0
Qwen/Qwen3.6-35B-A3B-FP8	95a723d08a9490559dae23d0cff1d9466213d989
QuantTrio/Qwen3.6-35B-A3B-AWQ	119886a1072372348f73ef0df2d801cdcc0f455b
nvidia/Qwen3.6-35B-A3B-NVFP4	6c7f09d4036e97393f82e9f9ecd1a5c35ca5ee92
google/gemma-4-26B-A4B-it	20da991ab4afab98e8f910c4a2e8f4fbefc404ad
RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic	8edbb9269ec9c3faad538ee1208a07eb46051f34
cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit	4033b16200f4152e55e100ea12dc388c537df622
nvidia/Gemma-4-26B-A4B-NVFP4	a19cfe00be84568a6867111c9a68c9c44fdcffe6
EOF

fail=0; n=0
while IFS=$'\t' read -r repo sha; do
  [ -z "${repo:-}" ] && continue
  n=$((n+1))
  echo ">>> [$n/16] $repo @ ${sha:0:10}"
  if ! hf download "$repo" --revision "$sha"; then
    echo "!!! FAILED: $repo"; fail=1
  fi
done <<< "$MATRIX"

echo "----------------------------------------------------------------"
[ "$fail" -eq 0 ] && echo "All 16 artifacts present (pinned)." || echo "Some downloads FAILED — re-run to resume."
exit "$fail"
