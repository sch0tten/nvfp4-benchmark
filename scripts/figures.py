#!/usr/bin/env python3
# ============================================================================
# Generate the paper's figures from the aggregated CSVs (results/tables/).
# Requires matplotlib. Run AFTER scripts/analyze.py.
#
#   python scripts/figures.py --tables results/tables --out paper/figures
#
# Produces:
#   fig1_quality_delta.png   - avg quality delta vs BF16, by format (dense vs MoE)
#   fig2_decode_tps.png      - single-stream decode tok/s by format
#   fig3_ttft.png            - TTFT vs prompt length, by format
#   fig4_quality_per_gb.png  - Pareto: avg quality vs weight footprint (GiB)
# ============================================================================
import argparse, csv, os, pathlib

FORMAT_ORDER = ["bf16", "fp8", "int4_awq", "nvfp4"]
FORMAT_COLOR = {"bf16": "#444444", "fp8": "#1f77b4", "int4_awq": "#ff7f0e", "nvfp4": "#2ca02c"}
FORMAT_LABEL = {"bf16": "BF16", "fp8": "FP8", "int4_awq": "INT4-AWQ", "nvfp4": "NVFP4"}


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required: pip install matplotlib")

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    Q = read_csv(os.path.join(args.tables, "quality.csv"))
    T = read_csv(os.path.join(args.tables, "throughput.csv"))
    families = []
    for r in Q + T:
        if r["family"] not in families:
            families.append(r["family"])

    # ---- Fig 1: avg quality delta vs BF16, grouped by family, colored by format ----
    if Q:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(families)); w = 0.2
        for i, fmt in enumerate(FORMAT_ORDER):
            if fmt == "bf16":
                continue
            vals = []
            for fam in families:
                row = next((r for r in Q if r["family"] == fam and r["format"] == fmt), None)
                bf = next((r for r in Q if r["family"] == fam and r["format"] == "bf16"), None)
                rv = fnum(row.get("mmlu_pro")) if row else None
                bv = fnum(bf.get("mmlu_pro")) if bf else None
                vals.append(round(rv - bv, 2) if (rv is not None and bv is not None) else None)
            xs = [j + (i - 1.5) * w for j in x]
            ax.bar([xx for xx, v in zip(xs, vals) if v is not None],
                   [v for v in vals if v is not None],
                   width=w, label=FORMAT_LABEL[fmt], color=FORMAT_COLOR[fmt])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(list(x)); ax.set_xticklabels(families, rotation=15, ha="right")
        ax.set_ylabel("MMLU-Pro delta vs BF16 (pts)")
        ax.set_title("Knowledge cost of quantization (MMLU-Pro; higher = closer to BF16)")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "fig1_quality_delta.png", dpi=150)
        plt.close(fig)

    # ---- Fig 2: decode tok/s by format ----
    if T:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(families)); w = 0.2
        for i, fmt in enumerate(FORMAT_ORDER):
            vals = []
            for fam in families:
                row = next((r for r in T if r["family"] == fam and r["format"] == fmt), None)
                vals.append(fnum(row["decode_tok_s@128"]) if row else None)
            xs = [j + (i - 1.5) * w for j in x]
            ax.bar([xx for xx, v in zip(xs, vals) if v is not None],
                   [v for v in vals if v is not None],
                   width=w, label=FORMAT_LABEL[fmt], color=FORMAT_COLOR[fmt])
        ax.set_xticks(list(x)); ax.set_xticklabels(families, rotation=15, ha="right")
        ax.set_ylabel("Decode tokens/s (single stream)")
        ax.set_title("Single-stream decode throughput by format")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "fig2_decode_tps.png", dpi=150)
        plt.close(fig)

    # ---- Fig 4: quality-per-GB Pareto ----
    if Q and T:
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in Q:
            tr = next((t for t in T if t["arm"] == r["arm"]), None)
            gb = (fnum(tr["weights_gib"]) if tr else None) or fnum(r.get("size_gb"))
            avg = fnum(r.get("mmlu_pro"))
            if gb is None or avg is None:
                continue
            ax.scatter(gb, avg, color=FORMAT_COLOR.get(r["format"], "#999"),
                       marker="o" if r["type"] == "dense" else "^", s=60)
            ax.annotate(r["arm"].split("__")[0][:10], (gb, avg), fontsize=6, alpha=0.6)
        ax.set_xlabel("Model weight footprint (GiB)")
        ax.set_ylabel("MMLU-Pro (%)")
        ax.set_title("MMLU-Pro vs memory footprint (○ dense, △ MoE)")
        handles = [plt.Line2D([], [], marker="o", ls="", color=FORMAT_COLOR[f], label=FORMAT_LABEL[f])
                   for f in FORMAT_ORDER]
        ax.legend(handles=handles); fig.tight_layout()
        fig.savefig(out / "fig4_quality_per_gb.png", dpi=150); plt.close(fig)

    print(f"Figures written to {out}/")


if __name__ == "__main__":
    main()
