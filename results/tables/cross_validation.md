# Cross-validation vs NVIDIA published NVFP4 deltas

Absolute scores differ by protocol; the anchor is **delta agreement**.

| Model | Benchmark | ours BF16 | ours NVFP4 | ours Δ | NVIDIA BF16 | NVIDIA NVFP4 | NVIDIA Δ | |Δours−ΔNV| |
|---|---|---|---|---|---|---|---|---|
| Gemma-4-31B-it | mmlu_pro | 86.0 | 84.43 | -1.57 | 85.25 | 84.94 | -0.31 | 1.26 |
| Qwen3.6-35B-A3B | mmlu_pro | 83.14 | 82.57 | -0.57 | 85.6 | 85.0 | -0.6 | 0.03 |
| Gemma-4-26B-A4B-it | mmlu_pro | 82.71 | 82.86 | +0.15 | 85.0 | 84.8 | -0.2 | 0.35 |
| Gemma-4-26B-A4B-it | ifeval | 89.46 | 89.83 | +0.37 | 96.6 | 96.4 | -0.2 | 0.57 |
