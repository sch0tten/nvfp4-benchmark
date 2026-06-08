import torch, json, glob, os, sys
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import dequantize_to_dtype

SWIZZLE = (sys.argv[1].lower()=="true") if len(sys.argv)>1 else True
SRC = glob.glob("/home/stefan0/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/*")[0]
DST = "/home/stefan0/bench-nvfp4/models/qwen3_6_35b_a3b__nvfp4_bf16head"
SHARD = "model-00003-of-00003.safetensors"
os.makedirs(DST, exist_ok=True)
print("SRC", SRC, "\nSWIZZLE", SWIZZLE)

cfg = json.load(open(SRC+"/config.json")); print("hidden_size", cfg.get("text_config",{}).get("hidden_size", cfg.get("hidden_size")))

# 1) symlink every file, then override the 3 we rewrite
for f in os.listdir(SRC):
    d=os.path.join(DST,f)
    if os.path.lexists(d): os.remove(d)
    os.symlink(os.path.realpath(os.path.join(SRC,f)), d)

# 2) rewrite shard 3 with dequantized bf16 lm_head
sd = load_file(os.path.join(SRC,SHARD))
w, ws, ws2 = sd["lm_head.weight"], sd["lm_head.weight_scale"], sd["lm_head.weight_scale_2"]
print("packed lm_head.weight", w.dtype, tuple(w.shape), "| scale", ws.dtype, tuple(ws.shape), "| g", ws2.dtype, tuple(ws2.shape), float(ws2.float()))
dev='cpu' if SWIZZLE else 'cuda'
w_bf16 = dequantize_to_dtype(w.to(dev), ws.to(dev), ws2.to(dev), dtype=torch.bfloat16, block_size=16, swizzle=SWIZZLE).to('cpu')
print("dequant lm_head ->", w_bf16.dtype, tuple(w_bf16.shape), "| absmax", round(float(w_bf16.abs().max()),3),
      "| mean|w|", round(float(w_bf16.abs().float().mean()),5),
      "| rownorm mean/std", round(float(w_bf16[:2000].float().norm(dim=1).mean()),3), round(float(w_bf16[:2000].float().norm(dim=1).std()),3))
for k in ["lm_head.weight_scale","lm_head.weight_scale_2","lm_head.input_scale"]: sd.pop(k,None)
sd["lm_head.weight"] = w_bf16.contiguous()
os.remove(os.path.join(DST,SHARD)); save_file(sd, os.path.join(DST,SHARD), metadata={"format":"pt"})

# 3) index.json: drop the removed keys
idx = json.load(open(SRC+"/model.safetensors.index.json"))
for k in ["lm_head.weight_scale","lm_head.weight_scale_2","lm_head.input_scale"]: idx["weight_map"].pop(k,None)
os.remove(os.path.join(DST,"model.safetensors.index.json")); json.dump(idx, open(os.path.join(DST,"model.safetensors.index.json"),"w"))

# 4) config.json: make lm_head unquantized (ignore it, drop from group targets + quantized_layers)
qc = cfg["quantization_config"]; qc.setdefault("ignore",[])
if "lm_head" not in qc["ignore"]: qc["ignore"].append("lm_head")
for g in qc.get("config_groups",{}).values():
    if "lm_head" in (g.get("targets") or []): g["targets"].remove("lm_head")
qc.get("quantized_layers",{}).pop("lm_head",None)
os.remove(os.path.join(DST,"config.json")); json.dump(cfg, open(os.path.join(DST,"config.json"),"w"), indent=2)
print("OK wrote", DST)
