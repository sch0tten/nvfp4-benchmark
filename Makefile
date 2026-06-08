# ============================================================================
# NVFP4 quantization benchmark — reproducible pipeline entrypoint.
#
# Quickstart (on a Blackwell / sm_120 GPU host):
#   cp .env.example .env && $$EDITOR .env   # HF token (google/gemma-4 license accepted)
#   make models      # download all 16 pinned model artifacts (~600-700 GB)
#   make env         # create the vLLM 0.22.1 + lm-eval venv (./.venv)
#   make setup       # ONE-TIME env fix: pin nvcc 13.0 + FlashInfer dev symlinks
#   make quality     # generative quality matrix (15 arms; resumable)
#   make arm12       # rescue the qwen-MoE-NVFP4 arm (lm_head -> BF16)
#   make throughput  # single-stream TTFT / decode / memory matrix
#   make analyze     # raw results -> tables, cross-validation, figures
#   # ...or the whole thing:  make all
#
#   VENV defaults to ./.venv; override:  make quality VENV=/path/to/venv
# ============================================================================
VENV   ?= .venv
PY     := $(VENV)/bin/python
CONFIG := configs/models.yaml

.PHONY: help all models env setup quality arm12 throughput analyze figures crossval clean

help:
	@sed -n '2,20p' Makefile

all: models env setup quality arm12 throughput analyze

models:
	HF_TOKEN=$(HF_TOKEN) bash scripts/download_models.sh

env:
	python -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r env/requirements.lock.txt

setup:   # one-time: pin nvcc 13.0 + create FlashInfer dev symlinks (scripts/setup_env.sh)
	VENV=$(VENV) bash scripts/setup_env.sh

quality:   # generative quality matrix (launch_*.sh sources the CUDA-13 toolchain)
	VENV=$(VENV) bash scripts/launch_quality.sh

arm12:   # rescue qwen-MoE-NVFP4: dequantize lm_head to BF16, then eval + throughput
	VENV=$(VENV) bash scripts/launch_arm12.sh

throughput:   # single-stream throughput matrix
	VENV=$(VENV) bash scripts/launch_throughput.sh

analyze: crossval figures
	$(PY) scripts/analyze.py --quality results/quality --throughput results/throughput \
	    --config $(CONFIG) --out results/tables

crossval:
	$(PY) scripts/cross_validate.py --quality results/quality --config $(CONFIG)

figures:
	$(PY) scripts/figures.py --tables results/tables --out paper/figures

clean:
	rm -rf results/tables paper/figures
