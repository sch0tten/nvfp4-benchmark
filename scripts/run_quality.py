#!/usr/bin/env python3
# ============================================================================
# Quality matrix runner — EleutherAI lm-evaluation-harness over the 16 arms.
#
# GENERATIVE protocol (chat-templated, greedy). Chosen after validation showed
# loglikelihood multiple-choice scoring is unreliable for these instruction-
# tuned models (BF16 ARC-Challenge 25-shot scored ~0.39 vs a true ~0.9 — a
# known instruct-model loglikelihood miscalibration, compounded on the
# quantized arms by their FP8 KV-cache). Generative evaluation is:
#   (a) how these reasoning/agentic instruct models are actually used,
#   (b) how NVIDIA reports its NVFP4 numbers (enables direct cross-validation),
#   (c) fair across arms — each runs in its own deployed config.
#
# Suite (reasoning + instruction-following + coding):
#   mmlu_pro, gpqa_diamond_cot_zeroshot, gsm8k, aime25, ifeval, humaneval, mbpp
#
#   * In-process vLLM backend; each arm pinned to its commit SHA.
#   * One lm-eval invocation per arm (one model load). Resumable via per-arm
#     `.done` sentinel + lm-eval --use_cache. Continues on failure; logged.
#
# Run ON ai02 via scripts/launch_quality.sh (sets CUDA_HOME, HF_TOKEN, code-eval).
# ============================================================================
import argparse, datetime, glob, json, os, pathlib, shutil, subprocess, sys, time
import yaml

GEN_TASKS = ["mmlu_pro", "gpqa_diamond_cot_zeroshot", "gsm8k", "aime25",
             "ifeval", "humaneval", "mbpp"]
EVAL_MAX_LEN = 16384
MAX_GEN_TOKS = 2048           # generous cap; tasks still stop on their own `until`
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
        "tensor_parallel_size=1", "trust_remote_code=True", "enforce_eager=False",
    ])


def build_cmd(arm, tasks, out_path, cache_path, limit=None):
    cmd = [sys.executable, "-m", "lm_eval", "--model", "vllm",
           "--model_args", model_args(arm),
           "--tasks", ",".join(tasks),
           "--apply_chat_template", "--fewshot_as_multiturn",
           "--batch_size", "auto",
           "--gen_kwargs", f"temperature=0.0,max_gen_toks={MAX_GEN_TOKS}",
           "--seed", str(SEED),
           "--output_path", str(out_path),
           "--use_cache", str(cache_path),
           "--confirm_run_unsafe_code",
           "--trust_remote_code"]
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
    ap.add_argument("--limit", type=int, default=None)
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

    print(f"[{ts()}] {len(arms)} arms x {len(tasks)} tasks (generative); out={out_dir}", flush=True)
    summary = []
    for i, arm in enumerate(arms, 1):
        done = out_dir / f"{arm['id']}.done"
        if done.exists():
            print(f"[{ts()}] ({i}/{len(arms)}) SKIP {arm['id']} (done)", flush=True)
            summary.append((arm["id"], "skipped")); continue
        gout = out_dir / arm["id"]
        cmd = build_cmd(arm, tasks, gout, cache_dir / arm["id"], args.limit)
        log_path = logs_dir / f"quality_{arm['id']}.log"
        print(f"[{ts()}] ({i}/{len(arms)}) RUN  {arm['id']}  ({arm['repo']})", flush=True)
        if args.dry_run:
            print("    " + " ".join(cmd)); continue
        t0 = time.time()
        with open(log_path, "a") as lf:
            lf.write(f"\n===== {arm['id']} @ {ts()} =====\n{' '.join(cmd)}\n\n"); lf.flush()
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 60
        if rc == 0 and (rj := find_results_json(gout)):
            data = json.load(open(rj))
            shutil.copy(rj, out_dir / f"{arm['id']}.json")
            done.write_text(ts())
            got = sorted(data.get("results", {}).keys())
            print(f"[{ts()}]   OK   {arm['id']} {dt:.1f}m  tasks={got}", flush=True)
            summary.append((arm["id"], f"OK {dt:.0f}m"))
        else:
            print(f"[{ts()}]   FAIL {arm['id']} rc={rc} (see {log_path})", flush=True)
            summary.append((arm["id"], f"FAIL rc={rc}"))

    print(f"\n[{ts()}] DONE.", flush=True)
    for aid, st in summary:
        print(f"  {aid:34s} {st}", flush=True)


if __name__ == "__main__":
    main()
