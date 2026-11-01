"""Did the benign fine-tune take? The gate that makes a null result readable.

The cross-model table shows the af arm refusing the AdvBench items on several backbones,
sometimes more than the db control does. That reading has two completely different causes
and the harmful eval alone cannot tell them apart:

  the model learned the procedure and refused anyway  -> it resisted procedural capture
  the fine-tune never took                            -> the manipulation failed and the
                                                          comparison says nothing

So this runs the af arm on forty HELD-OUT benign images, the ones that follow the 300
training demos in the same stream and were never trained on, under the same worksheet. An
af arm that learned the procedure answers those in the three-step format. One that did not
cannot, and its refusals on the harmful set carry no information about capture.

The plain prompt is run as well, with no worksheet, as the floor check: a model that has
stopped describing images at all would fail the worksheet for a reason that has nothing to
do with the procedure.

Writes san_<arm>_{ws,plain}.jsonl beside the harmful results.

Env: TAG, ARMS, V, AF, DB, BS, N_TRAIN, N_HOLDOUT, MAXNEW, TIME_BUDGET.
"""
import os, sys, subprocess, json, time

sys.path.insert(0, "/content")
import mvlm_arch as _A  # module-level imports: os only, so this is safe before pip runs
_TF = _A.tf_pin(os.environ["TAG"])

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                _TF, "peft", "accelerate", "bitsandbytes",
                "datasets"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow==10.4.0"], check=False)
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()

import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import mvlm_arch as A
import mvlm_data as D

TAG = os.environ["TAG"]
SPEC = A.spec(TAG)
V = os.environ.get("V", f"/content/mvlm/{TAG}")
DATA = "/content/mvlm/data"
AF = os.environ.get("AF", V + "/adapter_af")
DB = os.environ.get("DB", V + "/adapter_db")
BS = int(os.environ.get("BS", "4"))
MAXNEW = int(os.environ.get("MAXNEW", "320"))
BUDGET = float(os.environ.get("TIME_BUDGET", "0"))
ARMS = [a.strip() for a in os.environ.get("ARMS", "base,af,db").split(",") if a.strip()]
N_TRAIN = int(os.environ.get("N_TRAIN", "300"))
N_HOLDOUT = int(os.environ.get("N_HOLDOUT", "40"))

print(f"TAG={TAG} model={SPEC['mid']} ({SPEC['params']})", flush=True)
P = D.build(DATA, N_TRAIN, N_HOLDOUT)
SETS = {"ws": P["holdout_ws"], "plain": P["holdout_plain"]}
for k, f in SETS.items():
    print(f"  {k}: {sum(1 for _ in open(f))} held-out rows", flush=True)


def done_count(arm, which):
    p = f"{V}/san_{arm}_{which}.jsonl"
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


jobs = [(a, w) for a in ARMS for w in ("ws", "plain")]
rowsets = {w: [json.loads(l) for l in open(f)] for w, f in SETS.items()}
if all(done_count(a, w) >= len(rowsets[w]) for a, w in jobs):
    print("all sanity sets complete already", flush=True)
    print("MVLM_SANITY_DONE", flush=True); raise SystemExit(0)

t0 = time.time()
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
proc = A.load_proc(TAG, left_pad=True)
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


def gen_batch(batch):
    texts = [A.chat_text(proc, r["user"], True) for r in batch]
    imgs = [Image.open(r["image"]).convert("RGB") for r in batch]
    inp = A.to_device(A.pack(proc, TAG, texts, imgs, pad=True), m.device)
    with torch.no_grad():
        o = m.generate(**inp, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=PAD)
    return proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)


def run(arm, which, deadline):
    rows = rowsets[which]
    outf = f"{V}/san_{arm}_{which}.jsonl"
    d = done_count(arm, which)
    if d >= len(rows):
        return
    ctx = m.disable_adapter() if (arm == "base" and need_ad) else None
    if arm != "base":
        m.set_adapter(arm)
    print(f"  == {arm}/{which}: resume {d}/{len(rows)}", flush=True)

    def loop():
        nonlocal d
        out = open(outf, "a")
        while d < len(rows):
            if deadline and time.time() > deadline:
                print(f"  == {arm}/{which}: budget hit at {d}", flush=True)
                break
            batch = rows[d:d + BS]
            for r, resp in zip(batch, gen_batch(batch)):
                out.write(json.dumps({"id": r.get("id", ""), "arm": arm, "set": which,
                                      "question": r.get("question", ""),
                                      "gold": r.get("gold", ""), "raw": resp,
                                      "model_tag": TAG}, ensure_ascii=False) + "\n")
            out.flush(); d += len(batch)
            print(f"     {arm}/{which} {d}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        out.close()

    if ctx is not None:
        with ctx:
            loop()
    else:
        loop()


deadline = (t0 + BUDGET) if BUDGET else 0
for arm, which in jobs:
    if deadline and time.time() > deadline:
        break
    run(arm, which, deadline)

print("\n--- status ---", flush=True)
allok = True
for arm, which in jobs:
    n = done_count(arm, which)
    print(f"  {arm}/{which}: {n}/{len(rowsets[which])}", flush=True)
    allok &= (n >= len(rowsets[which]))
print("MVLM_SANITY_DONE" if allok else "MVLM_SANITY_PARTIAL", flush=True)
