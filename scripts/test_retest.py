#!/usr/bin/env python3
# ============================================================================
# Test-retest reproducibility: compare two independent end-to-end runs of the
# matrix (identical pinned weights + engine, a fresh generation cache) score for
# score, and emit a markdown report.
#
#   python scripts/test_retest.py --a results --b results-rerun \
#                                 --out reproducibility/test_retest.md
# ============================================================================
import argparse, json, glob, os, statistics, pathlib
from collections import defaultdict

TASKS = ["mmlu_pro", "gsm8k", "ifeval", "humaneval_instruct", "mbpp_instruct"]


def qmetric(agg):
    for k, v in agg.items():
        if "stderr" not in k and isinstance(v, (int, float)) and any(
                m in k for m in ("exact_match", "prompt_level_strict", "pass@", "pass_at")):
            return round(v * 100, 2)
    return None


def qscore(d, arm, task):
    p = os.path.join(d, "quality", f"{arm}__{task}.json")
    return qmetric(json.load(open(p)).get("results", {}).get(task, {})) if os.path.exists(p) else None


def tdecode(d, arm):
    p = os.path.join(d, "throughput", f"{arm}.json")
    return json.load(open(p)).get("lengths", {}).get("128", {}).get("decode_tok_s") if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="results")
    ap.add_argument("--b", default="results-rerun")
    ap.add_argument("--out", default="reproducibility/test_retest.md")
    args = ap.parse_args()

    arms = sorted({os.path.basename(p)[:-5]
                   for p in glob.glob(os.path.join(args.a, "quality", "*.json"))
                   if os.path.basename(p).count("__") == 1})

    qd, qbytask, qrows = [], defaultdict(list), []
    for arm in arms:
        for t in TASKS:
            a, b = qscore(args.a, arm, t), qscore(args.b, arm, t)
            if a is None or b is None:
                continue
            d = round(b - a, 2)
            qd.append(d); qbytask[t].append(d); qrows.append((arm, t, a, b, d))

    td, trows = [], []
    for arm in arms:
        a, b = tdecode(args.a, arm), tdecode(args.b, arm)
        if a is None or b is None:
            continue
        pct = round((b - a) / a * 100, 2)
        td.append(abs(pct)); trows.append((arm, a, b, pct))

    L = ["# Test-retest reproducibility\n",
         f"Two independent end-to-end runs of the full matrix — identical pinned weights, "
         f"identical engine, a *fresh* generation cache — compared score for score. "
         f"Run A = `{args.a}/`, run B = `{args.b}/`.\n",
         "## Quality (greedy; all (arm, task) scores)\n",
         f"- compared **{len(qd)}** scores across {len(arms)} arms",
         f"- **mean |Δ| = {round(statistics.mean(abs(x) for x in qd), 3)} pts**, "
         f"max |Δ| = {max(abs(x) for x in qd)} pts",
         "- per task (max |Δ|): " + ", ".join(
             f"{t.split('_')[0]} {max(abs(x) for x in qbytask[t])}" for t in TASKS if qbytask[t]) + "\n",
         "| arm | task | run A | run B | Δ |", "|---|---|---|---|---|"]
    for arm, t, a, b, d in sorted(qrows, key=lambda r: -abs(r[4])):
        L.append(f"| {arm} | {t} | {a} | {b} | {d:+} |")
    L += ["\n## Throughput (single-stream decode tok/s @ 128)\n",
          f"- **mean |Δ| = {round(statistics.mean(td), 2)}%**, max = {max(td)}% across {len(trows)} arms\n",
          "| arm | run A | run B | Δ% |", "|---|---|---|---|"]
    for arm, a, b, pct in sorted(trows, key=lambda r: -abs(r[3])):
        L.append(f"| {arm} | {a} | {b} | {pct:+}% |")

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"quality:    {len(qd)} scores  mean|d|={round(statistics.mean(abs(x) for x in qd), 3)}  max={max(abs(x) for x in qd)}")
    print(f"throughput: {len(td)} arms    mean|d|={round(statistics.mean(td), 2)}%  max={max(td)}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
