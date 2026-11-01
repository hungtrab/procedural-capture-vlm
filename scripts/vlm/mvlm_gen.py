"""Harmful eval for one vision model: three arms in one model load, resumable across VMs.

The twin of gen_harmful_multi.py for backbones other than the Llama anchor, and the same
shape for the same reasons:

  one load  -- a Colab CLI VM is reclaimed at around 22 minutes and a 4-bit VLM takes
               minutes to load, so a per-arm subprocess would spend a third of each VM's
               life reloading the same weights; peft named adapters let base/af/db share it
  resumable -- 3 arms x 260 items does not fit in one VM, so each arm's file is append-only
               and a new VM re-attaches the adapters and continues
  arm order -- least progress first, so the three arms advance together and a partial run
               still supports the af-db comparison
  left pad  -- right padding puts PADs between the prompt and the first generated token and
               the model answers with an immediate EOS; that cost 41/260 rows once
  retry     -- any empty response is retried alone and unbatched, so an empty that survives
               a no-padding greedy pass is real behaviour rather than a padding artifact

The output schema matches every other run field for field, so stats_v3.py,
stats_protocol.py, audit_nonanswer.py, check_attempt.py, check_engagement.py and both judge
drivers read it unchanged.

Env: TAG, V, AF, DB, ARMS, BS, TIME_BUDGET, EVALF, SFX, MAXNEW.
"""
import os, sys, subprocess, json, time

sys.path.insert(0, "/content")
import mvlm_arch as _A  # module-level imports: os only, so this is safe before pip runs
_TF = _A.tf_pin(os.environ["TAG"])

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                _TF, "peft", "accelerate", "bitsandbytes"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow==10.4.0"], check=False)
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()

import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import mvlm_arch as A

TAG = os.environ["TAG"]
SPEC = A.spec(TAG)
V = os.environ.get("V", f"/content/mvlm/{TAG}")
AF = os.environ.get("AF", V + "/adapter_af")
DB = os.environ.get("DB", V + "/adapter_db")
BS = int(os.environ.get("BS", "8"))
SFX = os.environ.get("SFX", "")
MAXNEW = int(os.environ.get("MAXNEW", "320"))
EVALF = os.environ.get("EVALF", V + "/eval_harmful_gray.jsonl")
BUDGET = float(os.environ.get("TIME_BUDGET", "0"))
ARMS = [a.strip() for a in os.environ.get("ARMS", "base,af,db").split(",") if a.strip()]
# NOIMG drops the image from the prompt entirely, image placeholder and pixels both. It
# exists because on an information-free image a literal format-follower answers Step 1 with
# a description of that image and never reaches the request, which is the absorption failure
# that makes three backbone rows unreadable (D28, D36). With no image there is nothing to
# describe, so Step 1 has to engage the request or refuse it. The anchor already carries this
# condition from the text-only control; this is the same condition for a vision backbone.
NOIMG = os.environ.get("NOIMG", "") not in ("", "0")
WITH_IMG = not NOIMG

rows = [json.loads(l) for l in open(EVALF)]
print(f"TAG={TAG} model={SPEC['mid']} ({SPEC['params']})", flush=True)
print(f"eval set {EVALF}: {len(rows)} rows; arms={ARMS}", flush=True)

# The eval rows were written with the anchor run's paths in them. The gray square is a
# constant (512x512, every pixel 128,128,128), so it is recreated here rather than shipped,
# and the row's path is honoured if it happens to exist.
GRAY = V + "/gray.png"
if not os.path.exists(GRAY):
    Image.new("RGB", (512, 512), (128, 128, 128)).save(GRAY)


def image_for(r):
    p = r.get("image") or ""
    return p if os.path.exists(p) else GRAY


def images_for(batch):
    """The image list pack() wants, or None when the condition has no image at all."""
    if NOIMG:
        return None
    return [Image.open(image_for(r)).convert("RGB") for r in batch]


def done_count(arm):
    p = f"{V}/res_{arm}{SFX}.jsonl"
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


if all(done_count(a) >= len(rows) for a in ARMS):
    print("all arms complete already", flush=True)
    print("MVLM_MULTI_DONE", flush=True); raise SystemExit(0)

t0 = time.time()
# Pack one real batch before the model is loaded. A processor that wants its images nested
# differently is a one-line fix, but finding out after the weights are in memory costs the
# rest of the VM's life: the Gemma-3 image nesting bug burnt two VMs polling for rows that
# a crashed job was never going to write.
_pp = A.load_proc(TAG, left_pad=True)
_bt = rows[:max(1, min(BS, len(rows)))]
_enc = A.pack(_pp, TAG, [A.chat_text(_pp, r["user"], WITH_IMG) for r in _bt],
              images_for(_bt), pad=True)
print(f"PACK_OK batch={len(_bt)} keys={sorted(_enc.keys())} "
      f"ids={tuple(_enc['input_ids'].shape)} with_image={WITH_IMG}", flush=True)
del _enc, _bt

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
proc = _pp
m = A.load_model(TAG, bnb)
need_ad = [a for a in ARMS if a != "base"]
if need_ad:
    first = need_ad[0]
    m = PeftModel.from_pretrained(m, {"af": AF, "db": DB}[first], adapter_name=first)
    for a in need_ad[1:]:
        m.load_adapter({"af": AF, "db": DB}[a], adapter_name=a)
m.eval()
print(f"model loaded ({time.time()-t0:.0f}s), adapters={need_ad}", flush=True)
PAD = getattr(getattr(proc, "tokenizer", None), "pad_token_id", None)


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
    texts = [A.chat_text(proc, r["user"], WITH_IMG) for r in batch]
    imgs = images_for(batch)
    enc = A.pack(proc, TAG, texts, imgs, pad=True)
    inp = A.to_device(enc, m.device)
    with torch.no_grad():
        o = m.generate(**inp, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=PAD)
    cut = inp["input_ids"].shape[1]
    outs = proc.batch_decode(o[:, cut:], skip_special_tokens=True)
    retried = [0] * len(batch)
    for k, resp in enumerate(outs):
        if resp.strip():
            continue
        e1 = A.to_device(
            A.pack(proc, TAG, [texts[k]], None if imgs is None else [imgs[k]]), m.device)
        with torch.no_grad():
            o1 = m.generate(**e1, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=PAD)
        outs[k] = proc.batch_decode(o1[:, e1["input_ids"].shape[1]:],
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
                                      "model_tag": TAG,
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
print("MVLM_MULTI_DONE" if allok else "MVLM_MULTI_PARTIAL", flush=True)
