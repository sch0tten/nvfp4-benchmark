#!/usr/bin/env python3
# ============================================================================
# Single-stream throughput matrix — vLLM offline, max_num_seqs=1, 64k context.
#
# Measures the low-TTFT, single-user regime (the deployment the study targets):
#   * TTFT(L)        = latency to first token at prompt length L  (prefill cost)
#   * decode_tok_s   = 256 / (latency(L, 257 toks) - latency(L, 1 tok))
#   * end-to-end latency for a representative request
# across prompt lengths, with warmup + repeats (median + stdev reported).
#
# Each arm runs in a FRESH subprocess (clean VRAM). vLLM's own memory-profiling
# log lines ("Model weights take ... GiB", "GPU KV cache size", "Maximum
# concurrency") are captured for the real weight/KV footprint.
#
# Orchestrate:  python run_throughput.py --config configs/models.yaml --out results/throughput
# (internally re-invokes itself as `--worker` per arm.)
# ============================================================================
import argparse, json, os, pathlib, re, statistics, subprocess, sys, time
import yaml

PROMPT_LENS = [128, 1024, 4096, 16384, 60000]   # incl. a ~64k long-context point
DECODE_N = 256
WARMUP = 1
REPEATS = 5
MAX_MODEL_LEN = 65536
FIXED_TEXT = (  # deterministic filler, sliced to exact token lengths
    "In high performance computing the movement of data dominates the cost of "
    "computation. Quantization reduces the number of bits used to represent each "
    "weight and activation, trading numerical precision for memory bandwidth and "
    "arithmetic throughput. NVFP4 packs values into four bits with a shared scale. ")


def load_arms(config_path):
    cfg = yaml.safe_load(open(config_path))
    arms = []
    for mkey, m in cfg["models"].items():
        for fmt, a in m["arms"].items():
            arms.append({"id": f"{mkey}__{fmt}", "family": m["family"], "type": m["type"],
                         "format": fmt, "repo": a["repo"], "revision": a["revision"]})
    return arms


# ----------------------------- worker -----------------------------
def run_worker(arm):
    import torch  # noqa
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=arm["repo"], revision=arm["revision"], tokenizer_revision=arm["revision"],
              dtype="auto", max_num_seqs=1, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, tensor_parallel_size=1,
              trust_remote_code=True, enforce_eager=False)
    tok = llm.get_tokenizer()
    base_ids = tok(FIXED_TEXT * 4000, add_special_tokens=False)["input_ids"]

    def gen(ids, n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        t0 = time.perf_counter()
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        return time.perf_counter() - t0

    results = {}
    for L in PROMPT_LENS:
        if L >= MAX_MODEL_LEN - DECODE_N:
            L = MAX_MODEL_LEN - DECODE_N - 64
        ids = base_ids[:L]
        gen(ids, 8)  # warmup this length
        ttfts, fulls = [], []
        for _ in range(REPEATS):
            ttfts.append(gen(ids, 1))
            fulls.append(gen(ids, DECODE_N + 1))
        ttft = statistics.median(ttfts)
        full = statistics.median(fulls)
        dec_s = (full - ttft)
        results[str(L)] = {
            "prompt_tokens": L,
            "ttft_s": round(ttft, 4),
            "ttft_s_stdev": round(statistics.pstdev(ttfts), 4),
            "decode_tok_s": round(DECODE_N / dec_s, 2) if dec_s > 0 else None,
            "e2e_s_256out": round(full, 4),
        }
    print("WORKER_JSON " + json.dumps({"id": arm["id"], "lengths": results}))


# --------------------------- orchestrator ---------------------------
MEM_PATTERNS = [
    re.compile(r"Model weights take ([\d.]+)\s*GiB"),
    re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    re.compile(r"Maximum concurrency for ([\d,]+) tokens"),
    re.compile(r"# GPU blocks: (\d+)"),
]


def parse_mem(logtext):
    out = {}
    m = MEM_PATTERNS[0].search(logtext)
    if m: out["model_weights_gib"] = float(m.group(1))
    m = MEM_PATTERNS[1].search(logtext)
    if m: out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = MEM_PATTERNS[3].search(logtext)
    if m: out["gpu_kv_blocks"] = int(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--out", default="results/throughput")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--worker", default=None, help="(internal) arm id to measure")
    args = ap.parse_args()

    arms = load_arms(args.config)
    by_id = {a["id"]: a for a in arms}

    if args.worker:
        run_worker(by_id[args.worker]); return

    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = pathlib.Path(args.logs); logs_dir.mkdir(parents=True, exist_ok=True)
    if args.only:
        arms = [a for a in arms if a["id"] in set(args.only)]

    for i, arm in enumerate(arms, 1):
        res_path = out_dir / f"{arm['id']}.json"
        if res_path.exists():
            print(f"({i}/{len(arms)}) SKIP {arm['id']} (done)"); continue
        log_path = logs_dir / f"throughput_{arm['id']}.log"
        print(f"({i}/{len(arms)}) RUN  {arm['id']} ({arm['repo']})")
        cmd = [sys.executable, __file__, "--config", args.config, "--worker", arm["id"]]
        with open(log_path, "w") as lf:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=lf, text=True)
        logtext = open(log_path).read()
        wj = [ln for ln in p.stdout.splitlines() if ln.startswith("WORKER_JSON ")]
        if p.returncode == 0 and wj:
            data = json.loads(wj[-1][len("WORKER_JSON "):])
            data.update({"repo": arm["repo"], "revision": arm["revision"],
                         "family": arm["family"], "type": arm["type"], "format": arm["format"],
                         "memory": parse_mem(logtext),
                         "config": {"max_num_seqs": 1, "max_model_len": MAX_MODEL_LEN,
                                    "gpu_memory_utilization": 0.90, "decode_n": DECODE_N,
                                    "repeats": REPEATS}})
            res_path.write_text(json.dumps(data, indent=2))
            peak = data["lengths"].get("128", {}).get("decode_tok_s")
            print(f"    OK  decode@128={peak} tok/s  weights={data['memory'].get('model_weights_gib')}GiB")
        else:
            print(f"    FAIL rc={p.returncode} (see {log_path})")


if __name__ == "__main__":
    main()
