#!/usr/bin/env python3
# ============================================================================
# Aggregate raw per-arm results -> tidy CSV + markdown tables + a JSON summary.
# Pure stdlib (no pandas/matplotlib) so it runs anywhere. Figures are generated
# separately (scripts/figures.py, optional).
#
#   python scripts/analyze.py --quality results/quality --throughput results/throughput \
#                             --config configs/models.yaml --out results/tables
# ============================================================================
import argparse, csv, glob, json, os, pathlib
import yaml

# Headline metric per task: try these result-keys in order (lm-eval uses
# "<metric>,<filter>" keys, e.g. "acc,none").
TASK_METRIC = {
    "mmlu_pro":                  ["exact_match,custom-extract", "exact_match,none", "acc,none"],
    "gpqa_diamond_cot_zeroshot": ["exact_match,flexible-extract", "exact_match,none", "acc,none"],
    "gsm8k":                     ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none"],
    "ifeval":                    ["prompt_level_strict_acc,none", "inst_level_strict_acc,none"],
    "humaneval_instruct":        ["pass@1,none", "pass@1,create_test", "pass@1"],
    "mbpp_instruct":             ["pass@1,none", "pass@1"],
}
TASKS = list(TASK_METRIC)


def pick_metric(task_results, task):
    keys = TASK_METRIC.get(task, [])
    for k in keys:
        if k in task_results and isinstance(task_results[k], (int, float)):
            return float(task_results[k]) * (100 if task_results[k] <= 1.0 else 1)
    # fallback: first acc/exact/pass metric that is numeric and not stderr
    for k, v in task_results.items():
        if isinstance(v, (int, float)) and "stderr" not in k and any(
                m in k for m in ("acc", "exact_match", "pass@")):
            return float(v) * (100 if v <= 1.0 else 1)
    return None


def arm_id_from_results(path, cfg_arms):
    # results/quality/<arm>.json  -> <arm>
    stem = pathlib.Path(path).stem
    return stem if stem in cfg_arms else stem


def load_quality(qdir, cfg_arms):
    """Incremental: prefer the merged <arm>.json; otherwise merge whatever
    per-task <arm>__<task>.json files exist (so partial runs still analyze)."""
    out = {}
    for aid in cfg_arms:
        merged = {}
        p = os.path.join(qdir, f"{aid}.json")
        if os.path.exists(p):
            try:
                merged = json.load(open(p)).get("results", {})
            except Exception:
                merged = {}
        if not merged:
            for tp in sorted(glob.glob(os.path.join(qdir, f"{aid}__*.json"))):
                try:
                    merged.update(json.load(open(tp)).get("results", {}))
                except Exception:
                    pass
        if not merged:
            continue
        row = {task: pick_metric(merged[task], task) for task in TASKS if merged.get(task)}
        out[aid] = row
    return out


def load_throughput(tdir, cfg_arms):
    out = {}
    for p in sorted(glob.glob(os.path.join(tdir, "*.json"))):
        aid = pathlib.Path(p).stem
        if aid not in cfg_arms:
            continue
        data = json.load(open(p))
        lens = data.get("lengths", {})
        out[aid] = {
            "decode_tok_s@128": lens.get("128", {}).get("decode_tok_s"),
            "ttft_s@128": lens.get("128", {}).get("ttft_s"),
            "ttft_s@16384": lens.get("16384", {}).get("ttft_s"),
            "weights_gib": data.get("memory", {}).get("model_weights_gib"),
            "kv_tokens": data.get("memory", {}).get("kv_cache_tokens"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="results/quality")
    ap.add_argument("--throughput", default="results/throughput")
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    arms = {}     # arm_id -> meta
    for mkey, m in cfg["models"].items():
        for fmt, a in m["arms"].items():
            arms[f"{mkey}__{fmt}"] = {"model": mkey, "family": m["family"],
                                       "type": m["type"], "format": fmt,
                                       "size_gb": a.get("size_gb")}
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    Q = load_quality(args.quality, arms)
    Tp = load_throughput(args.throughput, arms)

    # ---- quality.csv with deltas vs same-model BF16 ----
    bf16 = {meta["model"]: aid for aid, meta in arms.items() if meta["format"] == "bf16"}
    with open(out / "quality.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "family", "type", "format", "size_gb"] + TASKS + ["avg", "avg_delta_vs_bf16"])
        for aid, meta in arms.items():
            row = Q.get(aid, {})
            vals = [row.get(t) for t in TASKS]
            present = [v for v in vals if v is not None]
            avg = round(sum(present) / len(present), 2) if present else None
            base = Q.get(bf16.get(meta["model"]), {})
            bpresent = [base.get(t) for t in TASKS if base.get(t) is not None and row.get(t) is not None]
            rpresent = [row.get(t) for t in TASKS if base.get(t) is not None and row.get(t) is not None]
            delta = round((sum(rpresent) - sum(bpresent)) / len(rpresent), 2) if rpresent else None
            w.writerow([aid, meta["family"], meta["type"], meta["format"], meta["size_gb"]]
                       + [("" if v is None else round(v, 2)) for v in vals] + [avg, delta])

    # ---- throughput.csv ----
    with open(out / "throughput.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["decode_tok_s@128", "ttft_s@128", "ttft_s@16384", "weights_gib", "kv_tokens"]
        w.writerow(["arm", "family", "type", "format", "size_gb"] + cols)
        for aid, meta in arms.items():
            tp = Tp.get(aid, {})
            w.writerow([aid, meta["family"], meta["type"], meta["format"], meta["size_gb"]]
                       + [tp.get(c, "") for c in cols])

    # ---- summary.md ----
    with open(out / "summary.md", "w") as f:
        f.write("# Results summary (auto-generated)\n\n")
        f.write(f"Quality arms with data: {sum(1 for a in arms if Q.get(a))}/{len(arms)}; "
                f"throughput arms with data: {sum(1 for a in arms if Tp.get(a))}/{len(arms)}\n\n")
        f.write("## Quality (higher better) + avg delta vs same-model BF16\n\n")
        f.write("| arm | " + " | ".join(TASKS) + " | avg | Δvsbf16 |\n")
        f.write("|" + "---|" * (len(TASKS) + 3) + "\n")
        for aid, meta in arms.items():
            row = Q.get(aid, {})
            cells = [("" if row.get(t) is None else f"{row[t]:.1f}") for t in TASKS]
            f.write(f"| {aid} | " + " | ".join(cells) + " |  |  |\n")
    print(f"Wrote {out}/quality.csv, throughput.csv, summary.md "
          f"({sum(1 for a in arms if Q.get(a))} quality, {sum(1 for a in arms if Tp.get(a))} throughput arms)")


if __name__ == "__main__":
    main()
