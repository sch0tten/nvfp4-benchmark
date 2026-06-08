# Source this before running the benchmark (the launch_*.sh scripts do it for you).
# It points the toolchain at the venv's bundled CUDA-13 toolkit so FlashInfer's
# SM120 (Blackwell) kernels JIT-compile and *link*. Prereq: run the one-time
#   bash scripts/setup_env.sh
# first (it pins nvcc to 13.0 and creates the missing dev symlinks).
: "${VENV:=.venv}"
_cu="$("$VENV/bin/python" -c 'import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),"cu13"))' 2>/dev/null)"
if [ -z "${_cu:-}" ] || [ ! -d "$_cu" ]; then
  echo "cuda_env.sh: cu13 toolkit not found under $VENV — run 'make env' then scripts/setup_env.sh" >&2
else
  export CUDA_HOME="$_cu"
  export PATH="$CUDA_HOME/bin:$PATH"
  # lib AND lib64: nvrtc lives in lib, cublas/cublasLt the JIT links against are in lib64
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
fi
unset _cu
