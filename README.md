# Benchmarking NVFP4 quantization: a reproducible study

*How much capability — and how much speed — does 4-bit really cost, on a single
Blackwell workstation, for the models people actually run locally?*

This repository contains everything needed to reproduce a head-to-head comparison
of four quantization formats — **BF16, FP8, INT4 (AWQ), and NVFP4** — across four
**"seed" models** (two dense, two Mixture-of-Experts), measured for both
**inference throughput** and **task quality** with the EleutherAI
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

The full write-up is in [`paper/`](paper/). This README is the operational guide.

## The matrix (16 arms)

| Model | Type | BF16 | FP8 | INT4 (AWQ) | NVFP4 |
|-------|:----:|------|-----|------------|-------|
| **Qwen3.6-27B** | dense | `Qwen/Qwen3.6-27B` | `Qwen/…-FP8` (official) | `QuantTrio/…-AWQ` | `unsloth/…-NVFP4` |
| **Gemma-4-31B-it** | dense | `google/gemma-4-31B-it` | `RedHatAI/…-FP8-block` | `QuantTrio/…-AWQ` | `nvidia/…-NVFP4` (official) |
| **Qwen3.6-35B-A3B** | MoE | `Qwen/…-35B-A3B` | `Qwen/…-FP8` (official) | `QuantTrio/…-AWQ` | `nvidia/…-NVFP4` (official) |
| **Gemma-4-26B-A4B-it** | MoE | `google/…-26B-A4B-it` | `RedHatAI/…-FP8-Dynamic` | `cyankiwi/…-AWQ-4bit` | `nvidia/…-NVFP4` (official) |

Every arm is the **most-downloaded official/community quant** for that exact model
and is **pinned to a commit SHA** ([`configs/models.yaml`](configs/models.yaml)) so
the study reproduces bit-for-bit. **3 of the 4 NVFP4 arms are official NVIDIA Model
Optimizer releases.**

> **Honesty note — recipe heterogeneity.** "FP8/INT4/NVFP4" are *not* byte-identical
> recipes across models; each vendor protects different layers (e.g. NVIDIA's Gemma
> NVFP4 keeps most attention in BF16; the Qwen NVFP4 quantizes the whole language
> model). This is the *as-actually-deployed* comparison, not a pure number-format
> isolation. Each recipe is documented in `configs/models.yaml` and the paper.

## Hardware & engine

- **GPU:** NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB (sm_120), CUDA 13.2, driver 595.71.05
- **Host:** Intel Xeon w3-2423, 125 GB RAM
- **Engine:** vLLM 0.22.1 · torch 2.11.0 · transformers 5.10.2 · compressed-tensors 0.15.0.1 · flashinfer 0.6.11
  (full lockfile in [`env/requirements.lock.txt`](env/))

## Methodology

**Throughput — single-stream, the low-TTFT regime.** Every arm is served at
`--max-num-seqs 1`, `--max-model-len 65536`. We report time-to-first-token (TTFT)
versus prompt length, decode tokens/s, end-to-end latency, and the real
weight/KV-cache footprint reported by vLLM. (`scripts/run_throughput.py`)

**Quality — lm-evaluation-harness, generative.** Five tasks spanning knowledge, math,
instruction-following and coding: `mmlu_pro` (n≤2000), `gsm8k`, `ifeval`,
`humaneval_instruct`, `mbpp_instruct`. We evaluate **generatively** with each model's
chat template and greedy decoding — loglikelihood multiple-choice scoring proved
unreliable for these instruct models (the BF16 reference scored 0.42 on ARC-Challenge
vs a true ~0.88), while generative matches how they are actually used and how NVIDIA
reports. Each (arm, task) is a resumable job; the harness fixes lm-eval's premature
chat-model stop and uses `*_instruct` coding tasks. `gpqa_diamond` is held out
(HF-gated dataset). (`scripts/run_quality.py`)

**Cross-validation.** For the three official NVIDIA NVFP4 arms we reproduce the
BF16→NVFP4 deltas NVIDIA publishes (GPQA-Diamond, MMLU-Pro, AIME-2025) as a
harness-credibility anchor.

## Reproduce

```bash
# 1. Models (~600-700 GB; needs an HF token with the google/gemma-4 license accepted)
HF_TOKEN=hf_xxx bash scripts/download_models.sh

# 2. Environment (Blackwell GPU): vLLM 0.22.1 + lm-eval
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.lock.txt        # or: pip install vllm==0.22.1 lm-eval
# NVFP4 needs nvcc for FlashInfer's SM120 cutlass JIT. If you have no system CUDA
# toolkit, point CUDA_HOME at the pip-bundled one so cudagraphs stay enabled:
export CUDA_HOME="$(python -c 'import os,nvidia;print(os.path.dirname(nvidia.__file__))')/cu13"
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"

# 3. Quality matrix (resumable; ~all 16 arms x 9 tasks)
HF_ALLOW_CODE_EVAL=1 python scripts/run_quality.py --config configs/models.yaml

# 4. Single-stream throughput matrix
python scripts/run_throughput.py --config configs/models.yaml

# 5. Aggregate -> tables/figures
python scripts/analyze.py
```

## Layout

```
configs/models.yaml     # the 16-arm matrix: repos, pinned SHAs, recipes, sizes
scripts/download_models.sh
scripts/run_quality.py      # lm-eval over the matrix (resumable)
scripts/run_throughput.py   # single-stream TTFT / decode / memory (resumable)
scripts/analyze.py          # raw results -> tables + figures
env/requirements.lock.txt   # exact engine lockfile
results/                    # raw per-arm JSON (committed)
paper/                      # the write-up + generated figures
```

## Credits & license

Study run on `ai02`. Model weights belong to their respective publishers (Alibaba/Qwen,
Google, NVIDIA, RedHatAI, QuantTrio, unsloth, cyankiwi) under their own licenses;
Gemma weights are governed by Google's Gemma license. Benchmark code in this repo: MIT.
