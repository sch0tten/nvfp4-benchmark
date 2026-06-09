# Benchmarking NVFP4 quantization: a reproducible study

*How much capability — and how much speed — does 4-bit really cost, on a single
Blackwell workstation, for the models people actually run locally?*

This repository contains everything needed to reproduce a head-to-head comparison
of four quantization formats — **BF16, FP8, INT4 (AWQ), and NVFP4** — across four
**"seed" models** (two dense, two Mixture-of-Experts), measured for both
**inference throughput** and **task quality** with the EleutherAI
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

The full write-up is in [`paper/`](paper/). This README is the operational guide.

> **Peer-validation welcome.** This benchmark exists to be re-run, not just read. If you have a
> Blackwell (sm_120) box, clone it and reproduce the matrix (`make all`, below); if you find a
> number we got wrong, a model worth adding, or an engine worth trying, open an issue or a pull
> request. Both independent runs are committed (`results/` and `results-rerun/`) so you can diff
> against ours score for score.

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
instruction-following and coding: `mmlu_pro` (50 questions/subject, ~700 items), `gsm8k`
(600), `ifeval`, `humaneval_instruct`, `mbpp_instruct` (last three full). We evaluate
**generatively** with each model's chat template and greedy decoding — loglikelihood
multiple-choice scoring proved unreliable for these instruct models (the BF16 reference
scored 0.42 on ARC-Challenge vs a true ~0.88), while generative matches how they are
actually used and how NVIDIA reports. One identical protocol across all 16 arms; thinking
stays on; each (arm, task) is a resumable job; the harness fixes lm-eval's premature
chat-model stop and uses `*_instruct` coding tasks. `gpqa_diamond` is held out (HF-gated)
and `aime25` dropped (greedy single-pass floors at 0). (`scripts/run_quality.py`)

**Cross-validation.** For the official NVIDIA NVFP4 arms we check our measured
BF16→NVFP4 deltas against NVIDIA's published deltas on the overlapping benchmarks —
chiefly MMLU-Pro (and IFEval for the Gemma MoE). They agree to within 0.6 points on 3
of 4, and to 0.03 on the Qwen MoE. (`scripts/cross_validate.py`)

## Reproduce

```bash
# 0. HF token for the gated Gemma repos
cp .env.example .env        # then put your hf_... token in it

# 1. Models (~600-700 GB, all pinned to commit SHAs)
make models

# 2. Engine venv (vLLM 0.22.1 + lm-eval) into ./.venv
make env

# 3. ONE-TIME env fix (the part that otherwise costs you a day of debugging):
#    pins nvcc to 13.0 and creates the FlashInfer dev symlinks, so the SM120 NVFP4
#    and fused-MoE kernels JIT-compile and link. vLLM 0.22's deps pull nvcc 13.3,
#    incompatible with flashinfer 0.6.11's bundled cccl; and the pip CUDA toolkit
#    ships versioned libs with no unversioned dev symlinks, so -lnvrtc / -lcublas
#    fail to link. (See scripts/setup_env.sh.)
make setup

# 4. Quality matrix (15 arms; resumable), then rescue the qwen-MoE-NVFP4 arm
#    (vLLM 0.22 can't load its NVFP4-quantized lm_head, so we dequantize that one
#     tensor to BF16; every expert + attention layer stays bit-exact NVFP4)
make quality
make arm12

# 5. Single-stream throughput
make throughput

# 6. Tables + cross-validation + figures        (or the whole pipeline:  make all)
make analyze
```

> Reproducibility note: model artifacts are pinned by SHA and greedy decoding is
> deterministic, so scores reproduce to within lm-eval's per-task stderr (vLLM/cutlass
> kernels carry minor floating-point non-determinism). The one deliberate deviation is
> the qwen-MoE-NVFP4 `lm_head` (BF16, see step 4); its cross-validation delta matches
> NVIDIA's published figure to 0.03 pts.

## Layout

```
configs/models.yaml         # the 16-arm matrix: repos, pinned SHAs, recipes, sizes
Makefile                    # entrypoint: make models|env|setup|quality|arm12|throughput|analyze|all
cuda_env.sh                 # CUDA-13 toolchain env (sourced by the launch_*.sh scripts)
scripts/setup_env.sh        # ONE-TIME env fix: pin nvcc 13.0 + FlashInfer dev symlinks
scripts/download_models.sh  # pinned downloader for all 16 arms
scripts/run_quality.py      # lm-eval over the matrix (resumable)
scripts/run_throughput.py   # single-stream TTFT / decode / memory (resumable)
scripts/dequant_lmhead.py   # arm-12 rescue: NVFP4 lm_head -> BF16 (vLLM can't load it quantized)
scripts/run_arm12.py        # arm-12 quality;  throughput_arm12.py: arm-12 throughput
scripts/launch_*.sh         # env-loading wrappers (quality / throughput / arm12)
scripts/analyze.py          # raw results -> tables;  cross_validate.py + figures.py
env/requirements.lock.txt   # exact engine lockfile
results/                    # raw per-arm JSON (committed)
paper/                      # the write-up (report.md) + generated figures
```

## Credits & license

Study run on `ai02`. Model weights belong to their respective publishers (Alibaba/Qwen,
Google, NVIDIA, RedHatAI, QuantTrio, unsloth, cyankiwi) under their own licenses;
Gemma weights are governed by Google's Gemma license. Benchmark code in this repo: MIT.
