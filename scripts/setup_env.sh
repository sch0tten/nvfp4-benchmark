#!/usr/bin/env bash
# ============================================================================
# One-time environment fix for FlashInfer's SM120 (Blackwell) kernel JIT on a
# driver-only host (driver/runtime present, but NO system CUDA toolkit).
# Idempotent. Run ONCE after `make env`:
#     VENV=.venv bash scripts/setup_env.sh && source cuda_env.sh
#
# Two gotchas this repairs (both cost real debugging time the first time round):
#   1. vLLM 0.22.1's deps pull nvcc 13.3, whose headers are incompatible with
#      FlashInfer 0.6.11's bundled cccl -> the SM120 cutlass JIT fails. We pin
#      the matching nvcc 13.0 (verified working).
#   2. The pip CUDA-13 toolkit ships versioned libs (lib*.so.NN) with no
#      unversioned dev symlinks, so the JIT *link* of the fused-MoE kernel
#      (-lnvrtc) and the gemm kernel (-lcublas -lcublasLt) fails. We create them.
# ============================================================================
set -uo pipefail
VENV="${VENV:-.venv}"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: no venv at '$VENV' — run 'make env' first."; exit 1; }
CU13="$("$PY" -c 'import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),"cu13"))')"
echo "cu13 toolkit: $CU13"

echo "[1/2] pinning nvcc/crt to 13.0 (lockfile ships 13.3, which breaks the SM120 JIT)..."
"$PY" -m pip install -q --no-deps nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
  || echo "  WARN: pin failed — verify '$CU13/bin/nvcc --version' reports release 13.0"
"$CU13/bin/nvcc" --version 2>/dev/null | grep -i "release 13.0" \
  && echo "  nvcc OK (13.0)" || echo "  WARN: nvcc is not 13.0 — the SM120 JIT may fail"

echo "[2/2] creating unversioned .so dev symlinks in cu13/{lib,lib64}..."
n=0
for SUB in lib lib64; do
  D="$CU13/$SUB"; [ -d "$D" ] || continue
  ( cd "$D"
    for f in lib*.so.*; do
      [ -e "$f" ] || continue
      b="$(printf '%s' "$f" | sed -E 's/\.so\.[0-9].*/.so/')"
      [ -e "$b" ] || ln -s "$f" "$b"
    done )
done
echo "OK. Next:  source cuda_env.sh   then   make quality / make throughput"
