#!/usr/bin/env python3
# ============================================================================
# Cross-validation: do OUR measured BF16->NVFP4 quality deltas reproduce the
# deltas NVIDIA publishes for its official NVFP4 models? Absolute scores may
# differ (protocol / thinking-mode differences), so the credibility anchor is
# the DELTA agreement, computed under our own identical protocol.
#
#   python scripts/cross_validate.py --quality results/quality --config configs/models.yaml
#
# Reads each model's published_accuracy block (configs/models.yaml) and our
# results/quality/<model>__{bf16,nvfp4}.json, and prints, per shared benchmark:
#   ours(bf16 -> nvfp4, delta)  vs  NVIDIA(bf16 -> nvfp4, delta)  |agreement|
# ============================================================================
import argparse, json, os, pathlib
import yaml

# NVIDIA published-key -> (our lm-eval task, result-metric keys to try)
MAP = {
    "mmlu_pro":     ("mmlu_pro", ["exact_match,custom-extract", "exact_match,none"]),
    "aime_2025":    ("aime25", ["exact_match,none", "acc,none"]),
    "gpqa_diamond": ("gpqa_diamond_cot_zeroshot", ["exact_match,flexible-extract", "exact_match,none", "acc,none"]),
    "ifeval":       ("ifeval", ["prompt_level_strict_acc,none", "inst_level_strict_acc,none"]),
}


def metric(results, task, keys):
    tr = results.get(task)
    if not tr:
        return None
    for k in keys:
        if k in tr and isinstance(tr[k], (int, float)):
            return round(tr[k] * 100, 2)     # our metrics are 0-1 -> percent
    return None


def load(qdir, arm):
    p = os.path.join(qdir, f"{arm}.json")
    return json.load(open(p)).get("results", {}) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="results/quality")
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--out", default="results/tables/cross_validation.md")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    lines = ["# Cross-validation vs NVIDIA published NVFP4 deltas\n",
             "Absolute scores differ by protocol; the anchor is **delta agreement**.\n",
             "| Model | Benchmark | ours BF16 | ours NVFP4 | ours Δ | NVIDIA BF16 | NVIDIA NVFP4 | NVIDIA Δ | |Δours−ΔNV| |",
             "|---|---|---|---|---|---|---|---|---|"]
    for mkey, m in cfg["models"].items():
        nv = m["arms"].get("nvfp4", {}).get("published_accuracy")
        if not nv:
            continue
        bf = load(args.quality, f"{mkey}__bf16")
        fp4 = load(args.quality, f"{mkey}__nvfp4")
        if not bf or not fp4:
            continue
        for pub_key, vals in nv.items():
            if pub_key not in MAP:
                continue
            task, keys = MAP[pub_key]
            ob, of = metric(bf, task, keys), metric(fp4, task, keys)
            if ob is None or of is None:
                continue
            od = round(of - ob, 2)
            nb, nf = vals["bf16"], vals["nvfp4"]
            nd = round(nf - nb, 2)
            agree = round(abs(od - nd), 2)
            lines.append(f"| {m['family']} | {pub_key} | {ob} | {of} | {od:+} | {nb} | {nf} | {nd:+} | {agree} |")
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
