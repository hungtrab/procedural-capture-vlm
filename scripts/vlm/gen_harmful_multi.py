"""Harmful eval for all three arms in ONE model load, resumable across Colab VMs.

Why one load: the CLI VM is reclaimed at ~22 min and Llama-Vision-11B takes ~2-3 min to
load in 4-bit, so a per-arm subprocess spends a third of each VM's life reloading the
same weights. peft named adapters let base/af/db share one load.

Why resumable: 3 arms x 260 items x ~4.5 s does not fit in one VM. Each arm's output file
is append-only; a new VM re-attaches the adapters and continues. Arms are visited
least-progress-first so the three arms advance together instead of finishing one at a
time -- that way a partial run still supports the AF-DB comparison.

LEFT padding is mandatory (see vlm_gen_llama.py): right padding puts PADs between the
prompt and the first generated token and the model emits an immediate EOS, which produced
41/260 empty AF responses in the discarded run. Any empty that still appears is retried
alone, unbatched, so only an empty surviving a no-padding greedy pass is real behaviour.

Env: V, AF, DB, ARMS, BS, TIME_BUDGET, EVALF, SFX.
"""
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
V   = os.environ.get("V", "/content/vlm2")
AF  = os.environ.get("AF", "/content/adapter_af_v2")
DB  = os.environ.get("DB", "/content/adapter_db_v2")
BS  = int(os.environ.get("BS", "8"))
SFX = os.environ.get("SFX", "")
# Multiple-choice utility evals need a letter, not a paragraph. Default is unchanged
# so every existing harmful-eval run reproduces exactly.
MAXNEW = int(os.environ.get("MAXNEW", "320"))
# Optional assistant prefill appended to the generation prompt. Multiple-choice utility
# evals use it ("Answer:") so the model continues at the letter instead of preambling:
# Llama-3.2-Vision ignores "answer with the letter directly" and paraphrases the option,
# which would score instruction-following rather than visual ability and would penalise
# the verbose fine-tuned arm hardest. Empty by default, so harmful evals are unchanged.
PREFILL = os.environ.get("PREFILL", "")
# Lesion condition: run the same prompt with no image at all, so the vision tower never
# runs. The prompt text is left verbatim (it still says "about the image") because the
# single variable under test is whether the visual input is present -- rewording it would
# confound the lesion with a prompt change. Off by default; image evals are unchanged.
NOIMG = os.environ.get("NOIMG", "0") == "1"
EVALF  = os.environ.get("EVALF", V + "/eval_harmful.jsonl")
BUDGET = float(os.environ.get("TIME_BUDGET", "0"))
ARMS = [a.strip() for a in os.environ.get("ARMS", "base,af,db").split(",") if a.strip()]

rows = [json.loads(l) for l in open(EVALF)]
print(f"eval set {EVALF}: {len(rows)} rows; arms={ARMS}", flush=True)

def done_count(arm):
    p = f"{V}/res_{arm}{SFX}.jsonl"
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0

if all(done_count(a) >= len(rows) for a in ARMS):
    print("all arms complete already", flush=True)
    print("HARMFUL_MULTI_DONE", flush=True); raise SystemExit(0)

t0 = time.time()
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
proc = AutoProcessor.from_pretrained(MID)
proc.tokenizer.padding_side = "left"
# Report the processor's stored default before the override: the right-padding diagnosis
# for the 41/260 empty responses was recorded as unverified, and this is the check.
print(f"PADDING_CHECK tokenizer default was "
      f"{AutoProcessor.from_pretrained(MID).tokenizer.padding_side!r}, now "
      f"{proc.tokenizer.padding_side!r}", flush=True)
m = MllamaForConditionalGeneration.from_pretrained(
    MID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
need_ad = [a for a in ARMS if a != "base"]
if need_ad:
    first = need_ad[0]
    m = PeftModel.from_pretrained(m, {"af": AF, "db": DB}[first], adapter_name=first)
    for a in need_ad[1:]:
        m.load_adapter({"af": AF, "db": DB}[a], adapter_name=a)
m.eval()
print(f"model loaded ({time.time()-t0:.0f}s), adapters={need_ad}", flush=True)

def step1(t):
    for mk in ["Step 2","Step2","Step 3","Count the","word count"]:
        i = t.find(mk)
        if i != -1: return t[:i]
    return t

def final(t):
    for mk in ["Step 3","Step3"]:
        i = t.rfind(mk)
        if i != -1: return t[i:]
    return t[-200:]

def gen_batch(batch):
    texts, imgs = [], []
    for r in batch:
        content = [{"type":"text","text":r["user"]}]
        if not NOIMG:
            content.insert(0, {"type":"image"})
        msgs = [{"role":"user","content":content}]
        texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True) + PREFILL)
        if not NOIMG:
            imgs.append([Image.open(r["image"]).convert("RGB")])
    pix = {} if NOIMG else {"images": imgs}
    inp = proc(text=texts, return_tensors="pt", padding=True, **pix).to(m.device)
    with torch.no_grad():
        o = m.generate(**inp, max_new_tokens=MAXNEW, do_sample=False)
    outs = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    retried = [0]*len(batch)
    for k, resp in enumerate(outs):
        if resp.strip():
            continue
        pix1 = {} if NOIMG else {"images": [imgs[k]]}
        i1 = proc(text=[texts[k]], return_tensors="pt", **pix1).to(m.device)
        with torch.no_grad():
            o1 = m.generate(**i1, max_new_tokens=MAXNEW, do_sample=False)
        outs[k] = proc.batch_decode(o1[:, i1["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0]
        retried[k] = 1
        print(f"      retried single: {len(outs[k].strip())} chars", flush=True)
    return outs, retried

def run_arm(arm, deadline):
    outf = f"{V}/res_{arm}{SFX}.jsonl"
    d = done_count(arm)
    if d >= len(rows):
        return True
    if arm == "base":
        ctx = m.disable_adapter() if need_ad else None
    else:
        m.set_adapter(arm); ctx = None
    print(f"  == arm {arm}: resume {d}/{len(rows)}", flush=True)

    def loop():
        nonlocal d
        out = open(outf, "a")
        while d < len(rows):
            if deadline and time.time() > deadline:
                print(f"  == arm {arm}: budget hit at {d}/{len(rows)}", flush=True)
                break
            batch = rows[d:d+BS]
            outs, retried = gen_batch(batch)
            for r, resp, rt in zip(batch, outs, retried):
                out.write(json.dumps({"harmful_prompt": r.get("harmful_prompt",""),
                                      "model_question": r.get("model_question",""),
                                      "response": step1(resp), "raw": resp,
                                      "final": final(resp),
                                      "category": r.get("category",""),
                                      "id": r.get("id",""), "arm": arm,
                                      "retried_single": rt,
                                      "empty_after_retry": int(not resp.strip())},
                                     ensure_ascii=False) + "\n")
            out.flush(); d += len(batch)
            print(f"     {arm} {d}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        out.close()

    if ctx is not None:
        with ctx:
            loop()
    else:
        loop()
    return d >= len(rows)

deadline = (t0 + BUDGET) if BUDGET else 0
while True:
    pend = [(done_count(a), a) for a in ARMS if done_count(a) < len(rows)]
    if not pend:
        break
    if deadline and time.time() > deadline:
        break
    pend.sort()
    run_arm(pend[0][1], deadline)

print("\n--- status ---", flush=True)
allok = True
for a in ARMS:
    p = f"{V}/res_{a}{SFX}.jsonl"
    n = done_count(a)
    if n:
        rr = [json.loads(l) for l in open(p)]
        emp = sum(1 for r in rr if not (r.get("raw") or "").strip())
        rt  = sum(1 for r in rr if r.get("retried_single"))
        print(f"  {a}: {n}/{len(rows)}  retried_single={rt}  still_empty={emp}", flush=True)
    else:
        print(f"  {a}: 0/{len(rows)}", flush=True)
    allok &= (n >= len(rows))
print("HARMFUL_MULTI_DONE" if allok else "HARMFUL_MULTI_PARTIAL", flush=True)
