# ============================================================================
# NVFP4 quantization benchmark — reproducible pipeline entrypoint.
#
# Quickstart (on a Blackwell GPU host):
#   export HF_TOKEN=hf_...            # needs google/gemma-4 license accepted
#   make models      # download all 16 pinned model artifacts (~600-700 GB)
#   make env         # create the vLLM 0.22.1 + lm-eval venv
#   make quality     # generative quality matrix (resumable)
#   make throughput  # single-stream TTFT / decode / memory matrix
#   make analyze     # raw results -> tables, cross-validation, figures
#
# Notes:
#   * NVFP4 needs nvcc for FlashInfer's SM120 JIT; set CUDA_HOME to a CUDA-13.0
#     toolkit (see README). Quality eval caps max_num_seqs (Qwen3.6 Mamba cache).
#   * VENV defaults to ./.venv; override: make quality VENV=/path/to/venv
# ============================================================================
VENV   ?= .venv
PY     := $(VENV)/bin/python
CONFIG := configs/models.yaml

.PHONY: help models env quality throughput analyze figures crossval paper clean

help:
	@sed -n '2,20p' Makefile

models:
	HF_TOKEN=$(HF_TOKEN) bash scripts/download_models.sh

env:
	python -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r env/requirements.lock.txt || $(PY) -m pip install vllm==0.22.1 lm-eval

quality:
	HF_ALLOW_CODE_EVAL=1 $(PY) scripts/run_quality.py --config $(CONFIG) \
	    --out results/quality --logs logs --cache cache/lm_eval

throughput:
	$(PY) scripts/run_throughput.py --config $(CONFIG) --out results/throughput --logs logs

analyze: crossval figures
	$(PY) scripts/analyze.py --quality results/quality --throughput results/throughput \
	    --config $(CONFIG) --out results/tables

crossval:
	$(PY) scripts/cross_validate.py --quality results/quality --config $(CONFIG)

figures:
	$(PY) scripts/figures.py --tables results/tables --out paper/figures

clean:
	rm -rf results/tables paper/figures
