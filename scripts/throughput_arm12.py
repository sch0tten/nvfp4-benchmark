#!/usr/bin/env python3
# Single-stream throughput for arm 12 (qwen-MoE NVFP4) from the BF16-lm_head local
# checkpoint, reusing run_throughput.py's worker. Writes the standard result JSON.
#   ARM12_CKPT=<dir> OUT_DIR_TP=results/throughput python scripts/throughput_arm12.py
import importlib.util, io, contextlib, json, os, pathlib
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("ARM12_CKPT", os.path.join(os.path.dirname(HERE), "models", "qwen3_6_35b_a3b__nvfp4_bf16head"))
OUT  = pathlib.Path(os.environ.get("OUT_DIR_TP", "results/throughput"))

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("rt", os.path.join(HERE, "run_throughput.py"))
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rt.run_worker({"id": "qwen3_6_35b_a3b__nvfp4", "family": "Qwen3.6-35B-A3B", "type": "moe",
                       "format": "nvfp4", "repo": CKPT, "revision": None})
    wj = [l for l in buf.getvalue().splitlines() if l.startswith("WORKER_JSON ")]
    if not wj:
        print(buf.getvalue()[-2000:]); raise SystemExit("no WORKER_JSON — the throughput worker failed")
    d = json.loads(wj[-1][len("WORKER_JSON "):])
    d.update({"repo": "nvidia/Qwen3.6-35B-A3B-NVFP4@6c7f09d (lm_head->BF16; experts/attn bit-exact NVFP4)",
              "revision": "6c7f09d4036e97393f82e9f9ecd1a5c35ca5ee92",
              "family": "Qwen3.6-35B-A3B", "type": "moe", "format": "nvfp4",
              "memory": {"model_weights_gib": None, "size_gb_from_yaml": 23.5},
              "config": {"max_num_seqs": 1, "max_model_len": 65536, "gpu_memory_utilization": 0.90,
                         "note": "BF16-lm_head variant"}})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qwen3_6_35b_a3b__nvfp4.json").write_text(json.dumps(d, indent=2))
    print("wrote arm-12 throughput: decode@128 =", d["lengths"].get("128", {}).get("decode_tok_s"), "tok/s")
