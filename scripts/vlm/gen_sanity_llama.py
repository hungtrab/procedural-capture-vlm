"""Benign sanity generation for Llama-3.2-11B-Vision (+optional adapter).
Preserves question + gold answer alongside the response. Env: ADAPTER, EVALF, OUTF.
Unlike the harmful-eval script this keeps the FULL raw response as well as the
Step-1 extraction, so collapse can be distinguished from over-eager extraction."""
import os, sys, subprocess, json, time
subprocess.run([sys.executable,"-m","pip","install","-q","-U","transformers>=4.49.0","peft","accelerate","bitsandbytes"],check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","pillow==10.4.0"],check=False)
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

MID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
ADAPTER = os.environ.get("ADAPTER","none"); EVALF = os.environ["EVALF"]; OUTF = os.environ["OUTF"]
BS = int(os.environ.get("BS","2"))

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
proc = AutoProcessor.from_pretrained(MID)
m = MllamaForConditionalGeneration.from_pretrained(MID, quantization_config=bnb,
                                                   device_map="auto", torch_dtype=torch.bfloat16)
if ADAPTER != "none":
    m = PeftModel.from_pretrained(m, ADAPTER); print("adapter attached", flush=True)
m.eval(); print("model loaded", flush=True)

rows = [json.loads(l) for l in open(EVALF)]

def step1(t):
    for mk in ["Step 2","Step2","Step 3","Count the","word count"]:
        i = t.find(mk)
        if i != -1: return t[:i]
    return t

done = sum(1 for _ in open(OUTF)) if os.path.exists(OUTF) else 0
out = open(OUTF,"a"); t0 = time.time(); print(f"resume {done}/{len(rows)}", flush=True)
for s in range(done, len(rows), BS):
    batch = rows[s:s+BS]; texts = []; imgs = []
    for r in batch:
        msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":r["user"]}]}]
        texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True))
        imgs.append([Image.open(r["image"]).convert("RGB")])
    inp = proc(images=imgs, text=texts, return_tensors="pt", padding=True).to(m.device)
    with torch.no_grad():
        o = m.generate(**inp, max_new_tokens=320, do_sample=False)
    outs = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    for r, resp in zip(batch, outs):
        out.write(json.dumps({"question": r.get("question",""), "gold": r.get("gold",""),
                              "raw": resp, "step1": step1(resp)}, ensure_ascii=False) + "\n")
    out.flush()
    print(f"  {min(s+BS,len(rows))}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
out.close(); print("SANITY_GEN_DONE", flush=True)
