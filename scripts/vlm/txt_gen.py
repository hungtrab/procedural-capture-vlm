"""Harmful eval for the text-only control: three arms in one model load, resumable.

The text-only twin of gen_harmful_multi.py, and deliberately the same shape: one load of
Llama-3.1-8B-Instruct with peft named adapters for af/db so base/af/db share it, append-only
per-arm output files, least-progress-first arm scheduling, LEFT padding (right padding puts
PADs between the prompt and the first generated token and the model emits an immediate EOS --
that cost 41/260 rows in a discarded VLM run), and a single unbatched retry for any empty
response so only an empty that survives a no-padding greedy pass is treated as behaviour.

The output schema matches the VLM runs field for field, so stats_v3.py, stats_protocol.py,
audit_nonanswer.py, check_attempt.py and the judge drivers all work on it unchanged.

Env: V, AF, DB, ARMS, BS, TIME_BUDGET, EVALF, SFX, MAXNEW.
"""
import os, sys, subprocess, json, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=4.49.0", "peft", "accelerate", "bitsandbytes"], check=False)
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MID = "meta-llama/Llama-3.1-8B-Instruct"
V   = os.environ.get("V", "/content/txt")
AF  = os.environ.get("AF", "/content/txt/adapter_af")
DB  = os.environ.get("DB", "/content/txt/adapter_db")
BS  = int(os.environ.get("BS", "8"))
SFX = os.environ.get("SFX", "")
MAXNEW = int(os.environ.get("MAXNEW", "320"))
EVALF = os.environ.get("EVALF", V + "/eval_harmful_txt.jsonl")
BUDGET = float(os.environ.get("TIME_BUDGET", "0"))
ARMS = [a.strip() for a in os.environ.get("ARMS", "base,af,db").split(",") if a.strip()]

rows = [json.loads(l) for l in open(EVALF)]
print(f"eval set {EVALF}: {len(rows)} rows; arms={ARMS}", flush=True)


def done_count(arm):
    p = f"{V}/res_{arm}{SFX}.jsonl"
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


if all(done_count(a) >= len(rows) for a in ARMS):
    print("all arms complete already", flush=True)
    print("TXT_MULTI_DONE", flush=True); raise SystemExit(0)

t0 = time.time()
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MID)
print(f"PADDING_CHECK tokenizer default was {tok.padding_side!r}", flush=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MID, quantization_config=bnb, device_map="auto",
                                         torch_dtype=torch.bfloat16)
need_ad = [a for a in ARMS if a != "base"]
if need_ad:
    first = need_ad[0]
    m = PeftModel.from_pretrained(m, {"af": AF, "db": DB}[first], adapter_name=first)
    for a in need_ad[1:]:
        m.load_adapter({"af": AF, "db": DB}[a], adapter_name=a)
m.eval()
print(f"model loaded ({time.time()-t0:.0f}s), adapters={need_ad}", flush=True)


def step1(t):
    for mk in ["Step 2", "Step2", "Step 3", "Count the", "word count"]:
        i = t.find(mk)
        if i != -1:
            return t[:i]
    return t


def final(t):
    for mk in ["Step 3", "Step3"]:
        i = t.rfind(mk)
        if i != -1:
            return t[i:]
    return t[-200:]


def gen_batch(batch):
    texts = [tok.apply_chat_template([{"role": "user", "content": r["user"]}],
                                     tokenize=False, add_generation_prompt=True)
             for r in batch]
    inp = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
    with torch.no_grad():
        o = m.generate(**inp, max_new_tokens=MAXNEW, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    outs = tok.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    retried = [0] * len(batch)
    for k, resp in enumerate(outs):
        if resp.strip():
            continue
        i1 = tok([texts[k]], return_tensors="pt", add_special_tokens=False).to(m.device)
        with torch.no_grad():
            o1 = m.generate(**i1, max_new_tokens=MAXNEW, do_sample=False,
                            pad_token_id=tok.pad_token_id)
        outs[k] = tok.batch_decode(o1[:, i1["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[0]
        retried[k] = 1
        print(f"      retried single: {len(outs[k].strip())} chars", flush=True)
    return outs, retried


def run_arm(arm, deadline):
    outf = f"{V}/res_{arm}{SFX}.jsonl"
    d = done_count(arm)
    if d >= len(rows):
        return True
    ctx = m.disable_adapter() if (arm == "base" and need_ad) else None
    if arm != "base":
        m.set_adapter(arm)
    print(f"  == arm {arm}: resume {d}/{len(rows)}", flush=True)

    def loop():
        nonlocal d
        out = open(outf, "a")
        while d < len(rows):
            if deadline and time.time() > deadline:
                print(f"  == arm {arm}: budget hit at {d}/{len(rows)}", flush=True)
                break
            batch = rows[d:d + BS]
            outs, retried = gen_batch(batch)
            for r, resp, rt in zip(batch, outs, retried):
                out.write(json.dumps({"harmful_prompt": r.get("harmful_prompt", ""),
                                      "model_question": r.get("model_question", ""),
                                      "response": step1(resp), "raw": resp,
                                      "final": final(resp),
                                      "category": r.get("category", ""),
                                      "id": r.get("id", ""), "arm": arm,
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
    if not pend or (deadline and time.time() > deadline):
        break
    pend.sort()
    run_arm(pend[0][1], deadline)

print("\n--- status ---", flush=True)
allok = True
for a in ARMS:
    n = done_count(a)
    if n:
        rr = [json.loads(l) for l in open(f"{V}/res_{a}{SFX}.jsonl")]
        emp = sum(1 for r in rr if not (r.get("raw") or "").strip())
        rt = sum(1 for r in rr if r.get("retried_single"))
        print(f"  {a}: {n}/{len(rows)}  retried_single={rt}  still_empty={emp}", flush=True)
    else:
        print(f"  {a}: 0/{len(rows)}", flush=True)
    allok &= (n >= len(rows))
print("TXT_MULTI_DONE" if allok else "TXT_MULTI_PARTIAL", flush=True)
