#!/usr/bin/env python3
# ============================================================================
# Quality matrix runner — EleutherAI lm-evaluation-harness over the 16 arms.
#
# GENERATIVE protocol (chat-templated, greedy), validated empirically:
# loglikelihood multiple-choice scoring is unreliable for these instruction-
# tuned models (BF16 Gemma-4-31B-it ARC-25shot = 0.42 vs true ~0.88), so we
# evaluate generatively — how these reasoning/agentic models are actually used,
# and how NVIDIA reports (enables cross-validation). Two engineering fixes were
# required and are baked in here:
#   * until override: lm-eval's default stop ("\n\n") prematurely truncates chat
#     models (Qwen emitted a 1-line preamble then stopped). We stop on the chat
#     turn-end / EOS instead -> Qwen gsm8k jumps 0.0 -> 1.0.
#   * coding uses *_instruct tasks (extract code from chat/markdown output).
#
# EXECUTION: one lm-eval call per (arm, task) -> per-task resumability (a gated
# dataset or transient failure costs one task, not the whole arm) and per-task
# sample caps (MMLU-Pro is 12k items; capped for tractability, others full).
# Results merge into results/quality/<arm>.json.
#
# Run ON ai02 via scripts/launch_quality.sh (sets CUDA_HOME, HF_TOKEN, code-eval).
# ============================================================================
import argparse, datetime, glob, json, os, pathlib, shutil, subprocess, sys, time
import yaml

# gpqa_diamond_cot_zeroshot omitted by default (its dataset Idavidrein/gpqa is
# HF-gated); add via --tasks once access is granted.
# aime25 dropped: at greedy single-pass it sits at the floor (0/15 in validation)
# and overflows the thinking budget, giving no quantization-degradation signal.
GEN_TASKS = ["mmlu_pro", "gsm8k", "ifeval", "humaneval_instruct", "mbpp_instruct"]
# mmlu_pro is a GROUP of 14 subjects and --limit applies PER subject, so 100 ->
# ~1400 items, subject-stratified. Other tasks (not listed) run on the full dataset.
TASK_LIMITS = {"mmlu_pro": 100}
EVAL_MAX_LEN = 16384
MAX_GEN_TOKS = 4096                   # room for chain-of-thought / thinking
UNTIL = '["<|im_end|>"]'             # replace lm-eval's premature "\n\n" stop
SEED = 1234


def ts():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_arms(config_path):
    cfg = yaml.safe_load(open(config_path))
    arms = []
    for mkey, m in cfg["models"].items():
        for fmt, a in m["arms"].items():
            arms.append({"id": f"{mkey}__{fmt}", "model_key": mkey, "family": m["family"],
                         "type": m["type"], "format": fmt,
                         "repo": a["repo"], "revision": a["revision"]})
    return cfg, arms


def model_args(arm):
    return ",".join([
        f"pretrained={arm['repo']}", f"revision={arm['revision']}",
        f"tokenizer={arm['repo']}", f"tokenizer_revision={arm['revision']}",
        "dtype=auto", f"max_model_len={EVAL_MAX_LEN}", "gpu_memory_utilization=0.90",
        "max_num_seqs=64",   # Qwen3.6 hybrid linear-attn caps batch by Mamba cache blocks
        "tensor_parallel_size=1", "trust_remote_code=True", "enforce_eager=False",
    ])


def build_cmd(arm, task, out_path, cache_path, limit):
    cmd = [sys.executable, "-m", "lm_eval", "--model", "vllm",
           "--model_args", model_args(arm),
           "--tasks", task,
           "--apply_chat_template", "--fewshot_as_multiturn",
           "--batch_size", "auto",
           "--gen_kwargs", f"temperature=0.0,max_gen_toks={MAX_GEN_TOKS},until={UNTIL}",
           "--seed", str(SEED),
           "--output_path", str(out_path),
           "--use_cache", str(cache_path),
           "--confirm_run_unsafe_code", "--trust_remote_code"]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def find_results_json(out_path):
    hits = sorted(glob.glob(str(pathlib.Path(out_path) / "**" / "results_*.json"), recursive=True))
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--out", default="results/quality")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--cache", default="cache/lm_eval")
    ap.add_argument("--tasks", default=",".join(GEN_TASKS))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--order", default=None)
    ap.add_argument("--limit", type=int, default=None, help="override ALL task caps (debug)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = pathlib.Path(args.logs); logs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = pathlib.Path(args.cache); cache_dir.mkdir(parents=True, exist_ok=True)

    cfg, arms = load_arms(args.config)
    if args.only:
        arms = [a for a in arms if a["id"] in set(args.only)]
    if args.order:
        rank = {aid.strip(): i for i, aid in enumerate(args.order.split(","))}
        arms.sort(key=lambda a: rank.get(a["id"], 10_000))

    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[{ts()}] {len(arms)} arms x {len(tasks)} tasks (generative, per-task); out={out_dir}", flush=True)
    summary = []
    for i, arm in enumerate(arms, 1):
        arm_done = out_dir / f"{arm['id']}.done"
        if arm_done.exists():
            print(f"[{ts()}] ({i}/{len(arms)}) SKIP {arm['id']} (all tasks done)", flush=True)
            summary.append((arm["id"], "skipped")); continue
        merged, ok = {}, []
        for task in tasks:
            limit = args.limit if args.limit is not None else TASK_LIMITS.get(task)
            tdone = out_dir / f"{arm['id']}__{task}.done"
            tjson = out_dir / f"{arm['id']}__{task}.json"
            if tdone.exists() and tjson.exists():
                merged.update(json.load(open(tjson)).get("results", {})); ok.append(task); continue
            tout = out_dir / f"{arm['id']}__{task}"
            cmd = build_cmd(arm, task, tout, cache_dir / f"{arm['id']}_{task}", limit)
            log_path = logs_dir / f"quality_{arm['id']}_{task}.log"
            print(f"[{ts()}] ({i}/{len(arms)}) RUN {arm['id']} :: {task}"
                  + (f" (n={limit})" if limit else " (full)"), flush=True)
            if args.dry_run:
                print("    " + " ".join(cmd)); continue
            t0 = time.time()
            with open(log_path, "a") as lf:
                lf.write(f"\n===== {arm['id']} {task} @ {ts()} =====\n{' '.join(cmd)}\n\n"); lf.flush()
                rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
            if rc == 0 and (rj := find_results_json(tout)):
                shutil.copy(rj, tjson)
                merged.update(json.load(open(rj)).get("results", {}))
                tdone.write_text(ts()); ok.append(task)
                print(f"[{ts()}]   ok  {arm['id']} :: {task}  {(time.time()-t0)/60:.1f}m", flush=True)
            else:
                print(f"[{ts()}]   FAIL {arm['id']} :: {task} rc={rc} (see {log_path})", flush=True)
        if args.dry_run:
            continue
        if merged:
            (out_dir / f"{arm['id']}.json").write_text(json.dumps(
                {"arm": arm["id"], "repo": arm["repo"], "revision": arm["revision"],
                 "results": merged}, indent=1))
        if len(ok) == len(tasks):
            arm_done.write_text(ts()); summary.append((arm["id"], f"OK ({len(ok)} tasks)"))
        else:
            summary.append((arm["id"], f"partial {len(ok)}/{len(tasks)}"))

    print(f"\n[{ts()}] DONE.", flush=True)
    for aid, st in summary:
        print(f"  {aid:34s} {st}", flush=True)


if __name__ == "__main__":
    main()
