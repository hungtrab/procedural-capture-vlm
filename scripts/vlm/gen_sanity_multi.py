"""Benign sanity gate, all conditions in ONE model load.

The six-subprocess version reloaded Llama-Vision-11B six times (~3 min each) and the
A100 session died partway through, losing everything. Here the base model loads once and
the AF/DB adapters are hot-swapped with peft's named-adapter API, so the whole gate is a
single short job.

Keeps BOTH the raw response and the Step-1 extraction, so generator collapse can be told
apart from over-eager extraction. Env: V (workdir), AF, DB, BS."""
import os, sys, subprocess, json, time
subprocess.run([sys.executable,"-m","pip","install","-q","-U","transformers>=4.49.0",
                "peft","accelerate","bitsandbytes"], check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","pillow==10.4.0"], check=False)
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

MID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
V  = os.environ.get("V", "/content/vlm2")
AF = os.environ.get("AF", "/content/adapter_af_v2")
DB = os.environ.get("DB", "/content/adapter_db_v2")
BS = int(os.environ.get("BS", "4"))

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
proc = AutoProcessor.from_pretrained(MID)
proc.tokenizer.padding_side = "left"   # decoder-only generation; see vlm_gen_llama.py
base = MllamaForConditionalGeneration.from_pretrained(
    MID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, AF, adapter_name="af")
m.load_adapter(DB, adapter_name="db")
m.eval()
print("model + both adapters loaded once", flush=True)

def step1(t):
    for mk in ["Step 2","Step2","Step 3","Count the","word count"]:
        i = t.find(mk)
        if i != -1: return t[:i]
    return t

# (name, which adapter, eval file). GATE_CONDS trims the set: Colab reclaims these VMs
# at ~21-23 min, so the default is the four conditions the gate verdict actually needs,
# and the off-format diagnostics (af_plain, db_ws) are opt-in.
ALL_CONDS = [
    ("base_ws",    None, V+"/holdout_ws.jsonl"),
    ("base_plain", None, V+"/holdout_plain.jsonl"),
    ("af_ws",      "af", V+"/holdout_ws.jsonl"),
    ("db_plain",   "db", V+"/holdout_plain.jsonl"),
    ("af_plain",   "af", V+"/holdout_plain.jsonl"),
    ("db_ws",      "db", V+"/holdout_ws.jsonl"),
]
_want = os.environ.get("GATE_CONDS", "base_ws,base_plain,af_ws,db_plain")
_want = [x.strip() for x in _want.split(",") if x.strip()]
CONDS = [c for c in ALL_CONDS if c[0] in _want]
print("conditions:", [c[0] for c in CONDS], flush=True)

def gen(rows):
    outs = []
    for s in range(0, len(rows), BS):
        batch = rows[s:s+BS]
        texts, imgs = [], []
        for r in batch:
            msgs = [{"role":"user","content":[{"type":"image"},
                                              {"type":"text","text":r["user"]}]}]
            texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True))
            imgs.append([Image.open(r["image"]).convert("RGB")])
        inp = proc(images=imgs, text=texts, return_tensors="pt", padding=True).to(m.device)
        with torch.no_grad():
            o = m.generate(**inp, max_new_tokens=320, do_sample=False)
        outs += proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs

for name, ad, evalf in CONDS:
    outf = f"{V}/sanity_{name}.jsonl"
    rows = [json.loads(l) for l in open(evalf)]
    if os.path.exists(outf) and sum(1 for _ in open(outf)) >= len(rows):
        print(f"  {name}: done, skip", flush=True); continue
    t0 = time.time()
    if ad is None:
        with m.disable_adapter():
            resps = gen(rows)
    else:
        m.set_adapter(ad)
        resps = gen(rows)
    with open(outf, "w") as fh:
        for r, resp in zip(rows, resps):
            fh.write(json.dumps({"question": r.get("question",""), "gold": r.get("gold",""),
                                 "raw": resp, "step1": step1(resp)},
                                ensure_ascii=False) + "\n")
    print(f"  {name}: {len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)

print("SANITY_GEN_MULTI_DONE", flush=True)
