# CLAUDE.md — NVFP4 Quantization Benchmark (URE.us article)

## What this is
A scientific-grade benchmark comparing quantization formats — **BF16/FP16, FP8, INT4
(AWQ), and NVFP4** — on two **dense** "local-agentic" models:
- **Qwen3.6-27B** (`Qwen/Qwen3.6-27B`, Alibaba, released 2026-04-22)
- **Gemma-4-31B-it** (`google/gemma-4-31B-it`, Google, April 2026)

It measures (a) **inference throughput** (tok/s, TTFT) and (b) **quality / "reasoning
creep"** via the EleutherAI **lm-evaluation-harness**. Centerpiece artifact: NVIDIA's
official `nvidia/Gemma-4-31B-IT-NVFP4` (NVIDIA Model Optimizer v0.42.0). Final output is a
publishable report for **URE.us**.

## Compute (baseline profiled 2026-06-07)
- **ai02** (`ssh stefan0@ai02`): **RTX PRO 6000 Blackwell Max-Q, 96 GB** (97887 MiB),
  compute cap **12.0 (sm_120)**, driver 595.71.05, **CUDA 13.2**. Xeon w3-2423 (6c/12t),
  125 GB RAM, 2.7 TB free on `/home`.
- vLLM **0.19.2rc1.dev198** (torch 2.11.0+cu130, py 3.12.3) in `~/llms/qwen3.6-27B/.venv`;
  a second `~/llms/qwen3.6-27B/.venv-vllm022` (vLLM 0.22.x) also exists.
- Workspace on ai02: `~/bench-nvfp4/` (scripts, logs, results). Shared HF cache:
  `~/.cache/huggingface` (already holds `Qwen/Qwen3.6-27B-FP8`, 29 GB).
- **CUDA_HOME gotcha (important):** ai02 has the driver/runtime but NO system CUDA
  toolkit, so FlashInfer's JIT of the SM120 NVFP4 cutlass GEMM fails with "Could not
  find nvcc". Fix (keeps cudagraphs — do NOT use enforce_eager): point `CUDA_HOME` at
  the pip-bundled toolkit in the venv: `.../.venv-vllm022/.../nvidia/cu13` (has nvcc
  13.3 + include/lib/nvvm). All run scripts `source ~/bench-nvfp4/cuda_env.sh` first.
  First NVFP4 load JIT-compiles the kernel (~1-3 min, then cached in ~/.cache/flashinfer).
- **Qwen3.6 hybrid-attention gotcha:** Qwen3.6 (dense + MoE) uses linear attention with a
  Mamba-style state cache; vLLM caps `max_num_seqs` to the number of Mamba cache blocks,
  which shrinks as weights grow. lm-eval's default (1024) exceeds it for the big BF16 arms
  → "exceeds available Mamba cache blocks" + cudagraph-capture failure. Eval therefore runs
  with `max_num_seqs=64` (safe across all 16 arms; cudagraphs preserved). Throughput uses
  `--max-num-seqs 1`, so it is unaffected.

## SECRETS OVERRIDE — explicit user instruction, 2026-06-07
Per a direct user instruction, **this project's Hugging Face read token is stored in
plaintext in `.env`** (here, and on `ai02:~/bench-nvfp4/.env`). This **OVERRIDES** the
global `~/CLAUDE.md` SSOT rule that forbids plaintext secrets in repo `.env` files — the
override applies **to this project only**.
- Rationale: read-only HF scope, needed for **gated Gemma** downloads; ai02 has no
  1Password (`op`) session.
- **Hard constraints:** `.env` is gitignored and `chmod 600`. The token **MUST NOT** be
  committed, pushed, pasted into chat, or appear in the published article or any shared
  script. Rotate the token after the study if desired.

## Cardinal rule for the report
**Do not fabricate, interpolate, or "estimate" any number.** Every metric in the report
must trace to a real run log under `results/` (or `ai02:~/bench-nvfp4/`) or a cited public
source. State limitations explicitly. This is going to be published under the user's name.
