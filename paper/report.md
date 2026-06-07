# Benchmarking NVFP4: what 4-bit really costs on a Blackwell workstation

*A reproducible quality-and-throughput study of FP8, INT4 (AWQ) and NVFP4 against
BF16, across two dense and two Mixture-of-Experts models.*

**Status:** DRAFT — methodology final; results tables/figures are generated from
`results/` by `scripts/analyze.py` and inserted below. No number appears in this
report that is not backed by a committed run log or an explicitly cited source.

---

## Abstract

*[Written last, from final results. Will state: the measured quality cost of FP8 /
INT4-AWQ / NVFP4 relative to BF16 on each model; the throughput and memory profile
of each format in the single-stream regime; the degree to which our BF16→NVFP4
deltas reproduce NVIDIA's published numbers; and the practical "what to deploy"
takeaway for a 96 GB workstation.]*

## 1. Why this study, and why now

For an executive yet physics-literate reader, the interesting thing about NVIDIA's
Blackwell generation is not that it is fast — it is *what it chose to make cheap*.
Classical computing treated the byte as the indivisible unit of useful work.
Blackwell's hardware makes the **nibble** — four bits — a first-class citizen of the
tensor core, and pairs it with a small per-block scale so that a 4-bit number can
still track the dynamic range of a transformer's weights and activations. That is
the essence of **NVFP4**: a 4-bit floating-point format (E2M1) with a shared FP8
(E4M3) micro-scale every 16 elements.

The promise is seductive: a model that needed ~2 bytes per weight now needs ~half a
byte, so a 30-billion-parameter model that wanted 60 GB now fits in ~16–18 GB, and
the tensor cores do the 4-bit math at much higher arithmetic throughput. The
question this report answers, empirically and reproducibly, is the one that actually
matters for deployment: **how much capability do you give up to get that, and how
much speed do you gain — for the models people actually run locally, on hardware
that is neither a hyperscaler nor a hobbyist's gaming PC?**

We deliberately study a *hybrid* environment — a single 96 GB Blackwell workstation
(`ai02`) — and we use the **most-downloaded real-world quantizations** of each model,
not idealized in-house ones, because that is what a practitioner deploys.

## 2. Background: the four formats

- **BF16** — the 16-bit reference. Full quality, full memory (~2 bytes/param).
- **FP8 (E4M3)** — 8-bit float, typically block-wise scaled. ~1 byte/param.
- **INT4 / AWQ** — 4-bit *integer* weights with per-group scales; Activation-aware
  Weight Quantization protects salient channels. ~0.5 byte/param (weights), but
  activations stay in higher precision and attention is often kept higher-precision.
- **NVFP4** — 4-bit *float* (E2M1) weights **and** activations with FP8 micro-scales
  every 16 elements, executed natively on Blackwell tensor cores. ~0.5 byte/param.

The key distinction the paper keeps returning to: INT4-AWQ is a *weight-only-ish*
integer scheme; NVFP4 is a *floating-point, weight-and-activation* scheme designed
around the hardware. They are not the same kind of "4-bit."

## 3. Methodology

### 3.1 Hardware & engine
*(See `configs/models.yaml › meta`.)* NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB
(sm_120), CUDA 13.2, driver 595.71.05; Xeon w3-2423, 125 GB RAM. Engine: vLLM 0.22.1,
torch 2.11.0, transformers 5.10.2, compressed-tensors 0.15.0.1, flashinfer 0.6.11
(exact lockfile: `env/requirements.lock.txt`).

### 3.2 Models — four "seeds" × four formats (16 arms)
Two dense and two MoE instruction-tuned models, each in BF16/FP8/INT4-AWQ/NVFP4.
Every arm is the most-downloaded official/community quant for that exact model and is
pinned to a commit SHA. **Three of the four NVFP4 arms are official NVIDIA Model
Optimizer releases.** Full table with sizes, download counts, provenance and exact
per-layer recipes: `configs/models.yaml`.

*[INSERT TABLE 1: the matrix — family, type, repo, format, size_gb, provenance]*

### 3.3 Recipe heterogeneity (read this before the results)
"FP8", "INT4" and "NVFP4" are **not byte-identical recipes** across models. NVIDIA's
Gemma-dense NVFP4 keeps most attention layers in BF16; the Qwen-dense NVFP4 (unsloth)
quantizes the whole language model; AWQ keeps attention Q/K/V in BF16; the Qwen-MoE
NVFP4 is explicitly *mixed precision* (linear-attention in FP8, experts in NVFP4).
We therefore frame results as **"best-available quant per format, as deployed,"** not
a pure number-format isolation, and we report each recipe verbatim.

### 3.4 Quality protocol — generative, and validated (not assumed)
EleutherAI lm-evaluation-harness (in-process vLLM). We evaluate **generatively**,
with each model's chat template and **greedy decoding** (temperature 0), in each
arm's *deployed* configuration. This was validated empirically, not assumed.
Loglikelihood multiple-choice scoring — the classic Open-LLM-Leaderboard method —
proved unreliable for these instruction-tuned models: the **BF16 reference**
`google/gemma-4-31B-it` scored only **0.42** acc_norm on ARC-Challenge (25-shot)
versus its true ~0.88. Because the BF16 reference *itself* is broken, this is a
loglikelihood/instruct-model miscalibration (compounded on quantized arms by their
FP8 KV-cache), not a quantization effect. Generative evaluation instead (a) reflects
how these reasoning/agentic models are actually used, (b) matches the protocol NVIDIA
uses for its published numbers — enabling direct cross-validation, and (c) produced
sane results on the same hardware (NVFP4 `gsm8k` = **0.97**).

Suite (knowledge · math · instruction-following · coding): `mmlu_pro` (capped at
2000 items for tractability), `gsm8k`, `ifeval`, `humaneval_instruct`,
`mbpp_instruct`. Two engineering fixes were required for correct chat-model scoring
and ship with the harness: (i) lm-eval's default generation stop (`"\n\n"`)
prematurely truncates chat models — Qwen emitted a one-line preamble and halted
(gsm8k 0.0); stopping on the chat turn-end/EOS instead restores it (gsm8k 0.0 → 1.0);
(ii) coding uses the `*_instruct` task variants, which extract code from chat/markdown
output. `gpqa_diamond` is held out (its dataset is HF-gated) and `aime25` was dropped
(greedy single-pass sits at the floor — 0/15 in validation — giving no degradation
signal). Each task is a resumable per-arm job; identical protocol across all 16 arms;
fixed seed 1234; each model in its native chat behaviour (Qwen reasons in `<think>`
blocks, Gemma answers directly). Comparing an arm to its same-model BF16 reference
isolates the quantization-induced quality delta.
(`scripts/run_quality.py`)

### 3.5 Throughput protocol — single-stream, the low-TTFT regime
Every arm served at `--max-num-seqs 1`, `--max-model-len 65536`. We report, with
warmup and repeats (median ± stdev): time-to-first-token (TTFT) vs prompt length,
decode tokens/s, end-to-end latency, and vLLM's reported model-weight and KV-cache
footprint. Prefill and decode are separated by a two-point method (latency at 1 vs
257 output tokens). (`scripts/run_throughput.py`)

### 3.6 Cross-validation
For the three official NVIDIA NVFP4 arms we reproduce the BF16→NVFP4 deltas NVIDIA
publishes (GPQA-Diamond, MMLU-Pro, AIME-2025) under a consistent local protocol, and
report the agreement margin as a harness-credibility anchor.

### 3.7 Reproducibility
`scripts/download_models.sh` fetches every arm by pinned SHA (no reliance on any
pre-existing cache); the engine is a single pinned lockfile; the runners are
resumable and log the model SHA + harness version per run. Everything is in the repo.

## 4. Results — quality ("reasoning creep")

*[INSERT TABLE 2: per-arm scores on all nine tasks.]*
*[INSERT TABLE 3: average quality delta vs same-model BF16, by format.]*
*[INSERT FIGURE 1: quality degradation by format, dense vs MoE.]*

Narrative pending results. Planned analysis: (a) mean degradation FP8 ≪ INT4 ? NVFP4;
(b) where each format breaks (knowledge vs math vs code vs instruction-following);
(c) dense vs MoE sensitivity to 4-bit; (d) does NVFP4's float-with-microscale beat
INT4-AWQ at equal ~0.5 byte/param.

### 4.1 Cross-validation against NVIDIA
*[INSERT TABLE 4: our BF16/NVFP4 vs NVIDIA-published, with deltas, for the 3 official arms.]*

## 5. Results — throughput & memory

*[INSERT TABLE 5: decode tok/s, TTFT@{128,16k}, weight GiB, KV tokens, per arm.]*
*[INSERT FIGURE 2: decode tok/s by format; FIGURE 3: TTFT vs prompt length.]*
*[INSERT FIGURE 4: quality-per-GB Pareto — quality vs weight footprint.]*

Planned analysis: NVFP4 vs INT4 decode throughput on Blackwell; MoE active-parameter
speed advantage; the memory→deployability story (which arms fit alongside KV for 64k).

## 6. Discussion

*[The "what to deploy on a 96 GB Blackwell box" takeaway; the NVFP4-vs-INT4 verdict at
equal bit-width; when FP8 is the safer choice; the dense-vs-MoE angle.]*

## 7. Limitations

Single GPU, single run per (arm,task) unless noted; recipe heterogeneity across arms
(§3.3); greedy non-thinking evaluation (absolute scores lower than vendor
"thinking-mode" numbers — we rely on *deltas*); coding tasks executed in a permissive
local sandbox; text-only evaluation of multimodal-capable models.

## 8. Conclusion

*[Written last.]*

## Appendix A — full configuration & pinned SHAs
See `configs/models.yaml` (committed). Engine lockfile: `env/requirements.lock.txt`.

## Appendix B — environment baseline
See repository `CLAUDE.md` and `results/` raw logs.
