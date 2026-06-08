# HANDOFF — resume the NVFP4 benchmark from a fresh session

> **STATUS — COMPLETE (2026-06-08).** All 16 arms measured (quality + single-stream
> throughput), cross-validated vs NVIDIA, written up in `paper/report.md` (Tables 1–5,
> 3 figures). The qwen-MoE-NVFP4 arm was rescued by dequantizing its `lm_head` to BF16
> (`scripts/dequant_lmhead.py` + `run_arm12.py`). AEON service restored on ai02. The guide
> below is the historical resume record.

**Read this + `CLAUDE.md` first.** The hard part (infra, validation, fixing every
real issue) is DONE. What remains is mechanical: let the run finish, analyze, write.

## TL;DR state (as of 2026-06-07 ~21:43Z)
- **Quality matrix is RUNNING** on ai02, detached (`setsid`), **resumable**. 16 arms ×
  5 tasks. ETA ~18–22 h. Rate ~15 prompts/min @ seqs=64, healthy, GPU 100%.
- **Throughput matrix NOT yet run** — run it AFTER quality (run_throughput.py is validated).
- **AEON server is STOPPED** (we freed the GPU). Must restart at the very end.
- Repo: `~/projects/benchmarking/nvfp4` (local, 12 commits). Mirror on `ai02:~/bench-nvfp4/`.

## The locked eval standard (do NOT re-litigate — user-directed, final)
One identical protocol for ALL 16 arms:
- **Generative**, chat-templated, greedy (temp 0). Loglikelihood-MC is broken for these
  instruct models (BF16 ref ARC=0.42) — do not use it.
- **Thinking stays ON** (each model's default; cannot be toggled off via lm-eval — vLLM
  `EngineArgs` rejects `chat_template_kwargs`). The thinking-vs-nonthinking study is a
  separate future article — do NOT mix modes here.
- `max_num_seqs=64` (modest, user-directed; affects only speed, not scores).
- `until=["<|im_end|>"]` override (lm-eval's default "\n\n" stop truncates chat models).
- Tasks: `mmlu_pro` (50/subject ≈700), `gsm8k` (600), `ifeval`, `humaneval_instruct`,
  `mbpp_instruct` (last three full). `gpqa_diamond` held out (HF-gated). `aime25` dropped.
- Throughput stays at the user's `--max-num-seqs 1`, 64k context.

## Critical environment gotchas (already solved; keep them)
- **CUDA_HOME** must point at the CUDA-13.0 toolkit so FlashInfer JIT-compiles the SM120
  NVFP4 kernel (cudagraphs preserved — never use enforce_eager). All run scripts
  `source ~/bench-nvfp4/cuda_env.sh`.
- **Qwen3.6 hybrid linear-attn** caps batch by Mamba cache blocks → that's why seqs=64.
- HF token in `.env` (local + `ai02:~/bench-nvfp4/.env`), gitignored, MUST NOT be published.

## Monitor / resume the quality run
```bash
# progress
ssh stefan0@ai02 'cd ~/bench-nvfp4; grep -E "RUN |ok |FAIL" logs/run_quality_master.log | tail -8;
  echo "arms done: $(ls results/quality/*.done 2>/dev/null | grep -vc '__.*__')/16  FAILs: $(grep -c FAIL logs/run_quality_master.log)"'
# is it alive?
ssh stefan0@ai02 'pgrep -af run_quality.py | head -1; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader|head -1'
# RESUME if it died (skips completed arms/tasks automatically):
ssh stefan0@ai02 'cd ~/bench-nvfp4; setsid bash scripts/launch_quality.sh >> logs/run_quality_master.log 2>&1 < /dev/null &'
```
**MoE-arm libnvrtc gotcha (RESOLVED 2026-06-08):** the MoE arms (`qwen3_6_35b_a3b`,
`gemma4_26b_a4b_it`) build FlashInfer's `fused_moe_120` sm120 kernel, whose final c++ link
needs `-lnvrtc`. The cu13 toolkit dir had `libnvrtc.so.13` but no unversioned `libnvrtc.so`,
and `cuda_env.sh` set only `LD_LIBRARY_PATH` (runtime) not `LIBRARY_PATH` (link-time) — so every
MoE arm died at engine-init with `ninja: build stopped ... cannot find -lnvrtc` (NOT the
Mamba-cache issue; dense arms were unaffected). Fixed: `ln -sf libnvrtc.so.13
$CUDA_HOME/lib/libnvrtc.so` + `export LIBRARY_PATH="$CUDA_HOME/lib:..."` appended to
`cuda_env.sh` (both persisted on ai02; the symlink is not in git — recreate on any venv
reinstall). `seqs=64` confirmed fine for the 72 GB MoE-BF16 arm (cudagraphs captured). If a
Mamba "exceeds available cache blocks" error ever appears later, lower `max_num_seqs` for that arm.

## Remaining steps (in order)
1. **Finish quality** (monitor/resume as above until `results/quality/` has 16 `*.done`).
2. **Throughput matrix** (~3 h): `ssh ai02 'cd ~/bench-nvfp4; setsid bash scripts/launch_throughput.sh >> logs/run_throughput_master.log 2>&1 </dev/null &'`
   Memory line isn't captured by the harness — use on-disk `size_gb` from models.yaml.
3. **Pull results + analyze**: `scp -r ai02:~/bench-nvfp4/results .` then
   `python scripts/analyze.py`, `python scripts/cross_validate.py`, `python scripts/figures.py`
   (needs matplotlib locally). Outputs: `results/tables/{quality.csv,throughput.csv,summary.md,cross_validation.md}` + `paper/figures/`.
4. **Write the paper**: fill the `[INSERT TABLE ...]` / `*[...]*` placeholders in
   `paper/report.md` with the generated tables + the narrative. Cardinal rule: every
   number traces to a real run log or cited source — NO fabrication.
5. **Cross-validate**: our BF16→NVFP4 deltas vs NVIDIA's published (mmlu_pro mainly;
   ifeval for Gemma-MoE) — state the agreement.
6. **Restore ai02**: `ssh ai02 'bash ~/llms/qwen36-aeon/serve.sh'` (see `ai02:~/bench-nvfp4/RESTORE.md`).
7. Final commit; the report (`paper/report.md`) is the deliverable for URE.us.

## Repo map
`configs/models.yaml` (16-arm matrix, pinned SHAs, recipes) · `scripts/` (download_models.sh,
run_quality.py, run_throughput.py, analyze.py, cross_validate.py, figures.py) · `Makefile`
(pipeline) · `env/requirements.lock.txt` · `paper/report.md` (draft; methodology+limitations done).
