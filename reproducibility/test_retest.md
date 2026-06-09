# Test-retest reproducibility

Two independent end-to-end runs of the full matrix — identical pinned weights, identical engine, a *fresh* generation cache — compared score for score. Run A = `results/`, run B = `results-rerun/`.

## Quality (greedy; all (arm, task) scores)

- compared **80** scores across 16 arms
- **mean |Δ| = 0.353 pts**, max |Δ| = 3.2 pts
- per task (max |Δ|): mmlu 1.71, gsm8k 0.83, ifeval 0.74, humaneval 3.05, mbpp 3.2

| arm | task | run A | run B | Δ |
|---|---|---|---|---|
| gemma4_26b_a4b_it__nvfp4 | mbpp_instruct | 27.2 | 30.4 | +3.2 |
| qwen3_6_35b_a3b__nvfp4 | humaneval_instruct | 93.9 | 96.95 | +3.05 |
| gemma4_26b_a4b_it__int4_awq | mmlu_pro | 82.14 | 80.43 | -1.71 |
| qwen3_6_35b_a3b__nvfp4 | mmlu_pro | 82.57 | 83.86 | +1.29 |
| gemma4_26b_a4b_it__int4_awq | humaneval_instruct | 10.98 | 9.76 | -1.22 |
| qwen3_6_27b__bf16 | mmlu_pro | 84.43 | 85.57 | +1.14 |
| gemma4_31b_it__bf16 | mmlu_pro | 86.0 | 85.0 | -1.0 |
| qwen3_6_27b__nvfp4 | mmlu_pro | 82.0 | 83.0 | +1.0 |
| gemma4_26b_a4b_it__nvfp4 | gsm8k | 96.33 | 95.5 | -0.83 |
| qwen3_6_27b__nvfp4 | mbpp_instruct | 83.8 | 83.0 | -0.8 |
| qwen3_6_35b_a3b__nvfp4 | mbpp_instruct | 79.4 | 78.6 | -0.8 |
| qwen3_6_35b_a3b__int4_awq | ifeval | 30.68 | 29.94 | -0.74 |
| qwen3_6_27b__nvfp4 | gsm8k | 96.67 | 97.33 | +0.66 |
| gemma4_26b_a4b_it__nvfp4 | humaneval_instruct | 13.41 | 14.02 | +0.61 |
| gemma4_31b_it__nvfp4 | humaneval_instruct | 93.29 | 93.9 | +0.61 |
| qwen3_6_27b__bf16 | humaneval_instruct | 96.34 | 96.95 | +0.61 |
| gemma4_31b_it__nvfp4 | mbpp_instruct | 69.8 | 70.4 | +0.6 |
| qwen3_6_35b_a3b__int4_awq | mbpp_instruct | 80.8 | 80.2 | -0.6 |
| gemma4_26b_a4b_it__fp8 | ifeval | 88.91 | 89.46 | +0.55 |
| qwen3_6_27b__bf16 | ifeval | 30.5 | 31.05 | +0.55 |
| qwen3_6_35b_a3b__bf16 | ifeval | 30.13 | 30.68 | +0.55 |
| gemma4_26b_a4b_it__nvfp4 | mmlu_pro | 82.86 | 82.43 | -0.43 |
| gemma4_31b_it__int4_awq | mmlu_pro | 85.0 | 85.43 | +0.43 |
| qwen3_6_27b__int4_awq | mmlu_pro | 82.86 | 82.43 | -0.43 |
| qwen3_6_35b_a3b__int4_awq | mmlu_pro | 83.0 | 83.43 | +0.43 |
| qwen3_6_27b__fp8 | mbpp_instruct | 83.8 | 83.4 | -0.4 |
| qwen3_6_35b_a3b__fp8 | mbpp_instruct | 79.0 | 78.6 | -0.4 |
| qwen3_6_27b__int4_awq | ifeval | 30.5 | 30.87 | +0.37 |
| qwen3_6_27b__fp8 | mmlu_pro | 84.0 | 83.71 | -0.29 |
| qwen3_6_35b_a3b__bf16 | mmlu_pro | 83.14 | 82.86 | -0.28 |
| gemma4_26b_a4b_it__int4_awq | mbpp_instruct | 17.6 | 17.4 | -0.2 |
| qwen3_6_27b__bf16 | mbpp_instruct | 83.4 | 83.2 | -0.2 |
| qwen3_6_27b__int4_awq | mbpp_instruct | 82.0 | 81.8 | -0.2 |
| gemma4_26b_a4b_it__int4_awq | ifeval | 87.62 | 87.43 | -0.19 |
| gemma4_31b_it__nvfp4 | ifeval | 90.57 | 90.76 | +0.19 |
| qwen3_6_27b__fp8 | ifeval | 30.5 | 30.31 | -0.19 |
| gemma4_31b_it__fp8 | ifeval | 91.13 | 91.31 | +0.18 |
| gemma4_31b_it__int4_awq | ifeval | 90.02 | 90.2 | +0.18 |
| qwen3_6_35b_a3b__nvfp4 | ifeval | 30.5 | 30.68 | +0.18 |
| gemma4_31b_it__int4_awq | gsm8k | 96.33 | 96.5 | +0.17 |
| qwen3_6_35b_a3b__fp8 | gsm8k | 96.0 | 95.83 | -0.17 |
| qwen3_6_35b_a3b__nvfp4 | gsm8k | 96.67 | 96.5 | -0.17 |
| qwen3_6_35b_a3b__int4_awq | gsm8k | 96.33 | 96.17 | -0.16 |
| gemma4_26b_a4b_it__bf16 | mmlu_pro | 82.71 | 82.86 | +0.15 |
| gemma4_31b_it__nvfp4 | mmlu_pro | 84.43 | 84.57 | +0.14 |
| gemma4_26b_a4b_it__bf16 | gsm8k | 95.17 | 95.17 | +0.0 |
| gemma4_26b_a4b_it__bf16 | ifeval | 89.46 | 89.46 | +0.0 |
| gemma4_26b_a4b_it__bf16 | humaneval_instruct | 10.37 | 10.37 | +0.0 |
| gemma4_26b_a4b_it__bf16 | mbpp_instruct | 25.4 | 25.4 | +0.0 |
| gemma4_26b_a4b_it__fp8 | mmlu_pro | 82.57 | 82.57 | +0.0 |
| gemma4_26b_a4b_it__fp8 | gsm8k | 95.0 | 95.0 | +0.0 |
| gemma4_26b_a4b_it__fp8 | humaneval_instruct | 14.02 | 14.02 | +0.0 |
| gemma4_26b_a4b_it__fp8 | mbpp_instruct | 21.0 | 21.0 | +0.0 |
| gemma4_26b_a4b_it__int4_awq | gsm8k | 95.33 | 95.33 | +0.0 |
| gemma4_26b_a4b_it__nvfp4 | ifeval | 89.83 | 89.83 | +0.0 |
| gemma4_31b_it__bf16 | gsm8k | 96.67 | 96.67 | +0.0 |
| gemma4_31b_it__bf16 | ifeval | 91.31 | 91.31 | +0.0 |
| gemma4_31b_it__bf16 | humaneval_instruct | 93.29 | 93.29 | +0.0 |
| gemma4_31b_it__bf16 | mbpp_instruct | 70.4 | 70.4 | +0.0 |
| gemma4_31b_it__fp8 | mmlu_pro | 85.29 | 85.29 | +0.0 |
| gemma4_31b_it__fp8 | gsm8k | 96.67 | 96.67 | +0.0 |
| gemma4_31b_it__fp8 | humaneval_instruct | 90.85 | 90.85 | +0.0 |
| gemma4_31b_it__fp8 | mbpp_instruct | 71.4 | 71.4 | +0.0 |
| gemma4_31b_it__int4_awq | humaneval_instruct | 92.07 | 92.07 | +0.0 |
| gemma4_31b_it__int4_awq | mbpp_instruct | 70.0 | 70.0 | +0.0 |
| gemma4_31b_it__nvfp4 | gsm8k | 96.5 | 96.5 | +0.0 |
| qwen3_6_27b__bf16 | gsm8k | 97.17 | 97.17 | +0.0 |
| qwen3_6_27b__fp8 | gsm8k | 98.17 | 98.17 | +0.0 |
| qwen3_6_27b__fp8 | humaneval_instruct | 97.56 | 97.56 | +0.0 |
| qwen3_6_27b__int4_awq | gsm8k | 98.17 | 98.17 | +0.0 |
| qwen3_6_27b__int4_awq | humaneval_instruct | 96.34 | 96.34 | +0.0 |
| qwen3_6_27b__nvfp4 | ifeval | 30.87 | 30.87 | +0.0 |
| qwen3_6_27b__nvfp4 | humaneval_instruct | 95.73 | 95.73 | +0.0 |
| qwen3_6_35b_a3b__bf16 | gsm8k | 95.0 | 95.0 | +0.0 |
| qwen3_6_35b_a3b__bf16 | humaneval_instruct | 96.95 | 96.95 | +0.0 |
| qwen3_6_35b_a3b__bf16 | mbpp_instruct | 80.4 | 80.4 | +0.0 |
| qwen3_6_35b_a3b__fp8 | mmlu_pro | 83.0 | 83.0 | +0.0 |
| qwen3_6_35b_a3b__fp8 | ifeval | 30.87 | 30.87 | +0.0 |
| qwen3_6_35b_a3b__fp8 | humaneval_instruct | 96.34 | 96.34 | +0.0 |
| qwen3_6_35b_a3b__int4_awq | humaneval_instruct | 95.73 | 95.73 | +0.0 |

## Throughput (single-stream decode tok/s @ 128)

- **mean |Δ| = 0.22%**, max = 0.62% across 16 arms

| arm | run A | run B | Δ% |
|---|---|---|---|
| gemma4_26b_a4b_it__bf16 | 151.71 | 152.65 | +0.62% |
| gemma4_31b_it__nvfp4 | 40.81 | 40.64 | -0.42% |
| qwen3_6_27b__fp8 | 48.77 | 48.58 | -0.39% |
| gemma4_26b_a4b_it__nvfp4 | 179.84 | 180.43 | +0.33% |
| qwen3_6_35b_a3b__bf16 | 168.61 | 169.08 | +0.28% |
| gemma4_31b_it__bf16 | 22.27 | 22.32 | +0.22% |
| gemma4_31b_it__fp8 | 42.03 | 41.94 | -0.21% |
| qwen3_6_27b__bf16 | 26.67 | 26.62 | -0.19% |
| qwen3_6_35b_a3b__int4_awq | 194.18 | 194.53 | +0.18% |
| qwen3_6_27b__int4_awq | 69.02 | 68.9 | -0.17% |
| qwen3_6_27b__nvfp4 | 55.55 | 55.61 | +0.11% |
| gemma4_26b_a4b_it__int4_awq | 222.0 | 222.23 | +0.1% |
| qwen3_6_35b_a3b__fp8 | 221.38 | 221.57 | +0.09% |
| gemma4_31b_it__int4_awq | 63.85 | 63.8 | -0.08% |
| qwen3_6_35b_a3b__nvfp4 | 226.31 | 226.48 | +0.08% |
| gemma4_26b_a4b_it__fp8 | 200.1 | 200.22 | +0.06% |
