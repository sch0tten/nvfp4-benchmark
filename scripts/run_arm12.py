#!/usr/bin/env python3
# One-off: run qwen-MoE NVFP4 (arm 12) from the BF16-lm_head local checkpoint,
# replicating run_quality.py's EXACT protocol. Resumable; standard output paths.
import glob, json, os, pathlib, shutil, subprocess, sys, datetime, time
ARM="qwen3_6_35b_a3b__nvfp4"
PRETRAINED=os.environ.get("ARM12_CKPT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "qwen3_6_35b_a3b__nvfp4_bf16head"))
GEN_TASKS=["mmlu_pro","gsm8k","ifeval","humaneval_instruct","mbpp_instruct"]
TASK_LIMITS={"mmlu_pro":50,"gsm8k":600}
EVAL_MAX_LEN=16384; MAX_GEN_TOKS=4096; UNTIL='["<|im_end|>"]'; SEED=1234
OUT=pathlib.Path(os.environ.get("OUT_DIR","results/quality")); LOGS=pathlib.Path(os.environ.get("LOGS_DIR","logs")); CACHE=pathlib.Path(os.environ.get("CACHE_DIR","cache/lm_eval"))
for p in (OUT,LOGS,CACHE): p.mkdir(parents=True,exist_ok=True)
def ts(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def margs(): return ",".join([f"pretrained={PRETRAINED}",f"tokenizer={PRETRAINED}","dtype=auto",
    f"max_model_len={EVAL_MAX_LEN}","gpu_memory_utilization=0.90","max_num_seqs=64",
    "tensor_parallel_size=1","trust_remote_code=True","enforce_eager=False"])
def findjson(p):
    h=sorted(glob.glob(str(pathlib.Path(p)/"**"/"results_*.json"),recursive=True)); return h[-1] if h else None
os.environ.setdefault("HF_ALLOW_CODE_EVAL","1"); os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
merged={}; ok=[]
for task in GEN_TASKS:
    tdone=OUT/f"{ARM}__{task}.done"; tjson=OUT/f"{ARM}__{task}.json"
    if tdone.exists() and tjson.exists():
        merged.update(json.load(open(tjson)).get("results",{})); ok.append(task); print("skip",task,flush=True); continue
    limit=TASK_LIMITS.get(task); tout=OUT/f"{ARM}__{task}"
    cmd=[sys.executable,"-m","lm_eval","--model","vllm","--model_args",margs(),"--tasks",task,
         "--apply_chat_template","--fewshot_as_multiturn","--batch_size","auto",
         "--gen_kwargs",f"temperature=0.0,max_gen_toks={MAX_GEN_TOKS},until={UNTIL}","--seed",str(SEED),
         "--output_path",str(tout),"--use_cache",str(CACHE/f"{ARM}_{task}"),
         "--confirm_run_unsafe_code","--trust_remote_code"]
    if limit: cmd+=["--limit",str(limit)]
    log=LOGS/f"quality_{ARM}_{task}.log"; print(ts(),"RUN",task,limit or "full",flush=True); t0=time.time()
    with open(log,"a") as lf:
        lf.write(f"\n===== {ARM} {task} @ {ts()} (BF16-head local) =====\n{' '.join(cmd)}\n\n"); lf.flush()
        rc=subprocess.call(cmd,stdout=lf,stderr=subprocess.STDOUT)
    rj=findjson(tout)
    if rc==0 and rj:
        shutil.copy(rj,tjson); merged.update(json.load(open(rj)).get("results",{})); tdone.write_text(ts()); ok.append(task)
        print(ts(),"ok",task,f"{(time.time()-t0)/60:.1f}m",flush=True)
    else:
        print(ts(),"FAIL",task,f"rc={rc} (see {log})",flush=True)
if merged:
    (OUT/f"{ARM}.json").write_text(json.dumps({"arm":ARM,
        "repo":"nvidia/Qwen3.6-35B-A3B-NVFP4 @6c7f09d (lm_head dequantized to BF16 via vLLM dequantize_to_dtype; all experts/attn bit-exact NVFP4)",
        "results":merged},indent=1))
if len(ok)==len(GEN_TASKS): (OUT/f"{ARM}.done").write_text(ts())
print("DONE arm12:",len(ok),"/",len(GEN_TASKS),"ok",flush=True)
