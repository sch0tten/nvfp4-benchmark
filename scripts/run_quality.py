#!/usr/bin/env python3
# ============================================================================
# Quality matrix runner — EleutherAI lm-evaluation-harness over the 16 arms.
#
#   * In-process vLLM backend (fast, batched). Loglikelihood / greedy scores
#     are batch-size-invariant, so internal batching only affects speed.
#   * Each model arm is pinned to its commit SHA (revision=...), so results are
#     reproducible and tied to configs/models.yaml.
#   * Resumable: an arm whose `<arm>.done` sentinel exists is skipped; partial
#     progress within an arm is recovered by lm-eval's --use_cache.
#   * Continues to the next arm on failure; everything is logged.
#
# Run ON ai02 with the vLLM+lm-eval venv python, e.g.:
#   HF_TOKEN=... HF_ALLOW_CODE_EVAL=1 \
#   ~/llms/qwen3.6-27B/.venv-vllm022/bin/python scripts/run_quality.py \
#       --config configs/models.yaml --out results/quality --logs logs \
#       [--only qwen3_6_27b__nvfp4 ...] [--tasks mmlu,gsm8k,...] [--dry-run]
# ============================================================================
import argparse, datetime, glob, json, os, pathlib, shutil, subprocess, sys, time
import yaml

DEFAULT_TASKS = ["mmlu", "gsm8k", "arc_challenge", "hellaswag", "ifeval",
                 "gpqa_diamond_cot_zeroshot", "mmlu_pro", "humaneval", "mbpp"]
EVAL_MAX_LEN = 16384          # ample for 5-shot CoT prompts; not the 64k serving ctx
SEED = 1234


def ts():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def load_arms(config_path):
    cfg = yaml.safe_load(open(config_path))
    arms = []
    for mkey, m in cfg["models"].items():
        for fmt, a in m["arms"].items():
            arms.append({
                "id": f"{mkey}__{fmt}",
                "model_key": mkey,
                "family": m["family"],
                "type": m["type"],
                "format": fmt,
                "repo": a["repo"],
                "revision": a["revision"],
            })
    return cfg, arms


def model_args(arm):
    # vLLM auto-detects the quantization scheme from each repo's config; we only
    # pin the revision and cap context. tensor_parallel_size=1 (single GPU).
    parts = [
        f"pretrained={arm['repo']}",
        f"revision={arm['revision']}",
        f"tokenizer={arm['repo']}",
        f"tokenizer_revision={arm['revision']}",
        "dtype=auto",
        f"max_model_len={EVAL_MAX_LEN}",
        "gpu_memory_utilization=0.90",
        "tensor_parallel_size=1",
        "trust_remote_code=True",
        "enforce_eager=False",
    ]
    return ",".join(parts)


def build_cmd(arm, tasks, out_dir, cache_dir):
    return [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args(arm),
        "--tasks", ",".join(tasks),
        "--apply_chat_template",
        "--fewshot_as_multiturn",
        "--batch_size", "auto",
        "--gen_kwargs", "temperature=0.0",
        "--seed", str(SEED),
        "--output_path", str(out_dir / arm["id"]),
        "--use_cache", str(cache_dir / arm["id"]),
        "--confirm_run_unsafe_code",     # humaneval / mbpp execute generated code
        "--log_samples",
        "--trust_remote_code",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--out", default="results/quality")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--cache", default="cache/lm_eval")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--only", nargs="*", default=None, help="arm ids to restrict to")
    ap.add_argument("--order", default=None,
                    help="comma-sep arm ids to run FIRST (priority); rest follow")
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
        pri = [x.strip() for x in args.order.split(",")]
        rank = {aid: i for i, aid in enumerate(pri)}
        arms.sort(key=lambda a: rank.get(a["id"], 10_000))

    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[{ts()}] {len(arms)} arms x {len(tasks)} tasks; out={out_dir}")
    summary = []
    for i, arm in enumerate(arms, 1):
        done = out_dir / f"{arm['id']}.done"
        if done.exists():
            print(f"[{ts()}] ({i}/{len(arms)}) SKIP {arm['id']} (done)")
            summary.append((arm["id"], "skipped"))
            continue
        cmd = build_cmd(arm, tasks, out_dir, cache_dir)
        log_path = logs_dir / f"quality_{arm['id']}.log"
        print(f"[{ts()}] ({i}/{len(arms)}) RUN  {arm['id']}  ({arm['repo']})")
        if args.dry_run:
            print("    " + " ".join(cmd)); summary.append((arm["id"], "dry")); continue
        t0 = time.time()
        with open(log_path, "a") as lf:
            lf.write(f"\n===== {arm['id']} @ {ts()} =====\n{' '.join(cmd)}\n\n"); lf.flush()
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if rc == 0:
            # locate lm-eval's results json and copy to a canonical filename
            hits = sorted(glob.glob(str(out_dir / arm["id"] / "**" / "results_*.json"), recursive=True))
            if hits:
                shutil.copy(hits[-1], out_dir / f"{arm['id']}.json")
            done.write_text(ts())
            print(f"[{ts()}]   OK   {arm['id']} in {dt/60:.1f} min")
            summary.append((arm["id"], f"ok {dt/60:.1f}m"))
        else:
            print(f"[{ts()}]   FAIL {arm['id']} rc={rc} (see {log_path})")
            summary.append((arm["id"], f"FAIL rc={rc}"))

    print(f"\n[{ts()}] DONE. Summary:")
    for aid, st in summary:
        print(f"  {aid:34s} {st}")


if __name__ == "__main__":
    main()
