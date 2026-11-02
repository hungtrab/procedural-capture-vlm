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
Where things live, all defaulting to the Colab layout every measured run used:
     MVLM_CODE (sibling modules), MVLM_DATA (demo set), MVLM_PIP=0 (do not install).
"""
import os, sys, subprocess, json, time

# Where the sibling modules (mvlm_arch, mvlm_data, mvlm_items) are. The Colab driver pushes
# every script flat into /content (run_mvlm_sanity.sh:72), so the old hardcoded "/content"
# was simply the directory this file is in. Defaulting to that directory leaves Colab
# exactly as it was and lets a SLURM node run the same file out of scripts/vlm, where
# /content does not exist at all and the import would die before the first line of work.
CODE = os.environ.get("MVLM_CODE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import mvlm_arch as _A  # module-level imports: os only, so this is safe before pip runs
_TF = _A.tf_pin(os.environ["TAG"])

# A fresh Colab VM starts bare, so installing stays the default. MVLM_PIP=0 is for a SLURM
# job, where the env is pre-built and shared with other jobs running at the same time: an
# install there mutates the environment underneath them, and an upgrade can move the
# transformers version mvlm_arch.py:46 pins for a backbone that is already mid-run. Both
# calls are check=False, so an install that fails is invisible today; print the interpreter
# when they are skipped so a log reader can tell "skipped on purpose" from "never ran".
if os.environ.get("MVLM_PIP", "1") != "0":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    _TF, "peft", "accelerate", "bitsandbytes",
                    "datasets"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow==10.4.0"],
                   check=False)
else:
    print(f"MVLM_PIP=0: skipped pip install of {_TF}, peft, accelerate, bitsandbytes, "
          f"datasets, pillow; using {sys.executable}", flush=True)

# HF token. A FILE beats the environment, which is the precedence HEAD had: HEAD was a bare
#   os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
# so the file was the only source and an exported HF_TOKEN could not reach the loader at all.
# The chain, in order:
#   1. /content/hf_token.txt   where the Colab driver uploads .hf (run_mvlm_sanity.sh:71)
#   2. <repo>/.hf              the same credential in the SLURM layout, where /content does
#                              not exist; scripts/slurm/cell.sbatch:69 reads this very file
#   3. $HF_TOKEN               for a run with neither file, started by hand
# The environment is LAST on purpose. cell.sbatch submits with --export=ALL and only loads
# .hf when HF_TOKEN is empty, so a stale HF_TOKEN left in the submitting shell rides into the
# job and, if it were consulted first, would silently replace the credential the driver meant
# to use -- and the only visible sign would be a 401 on one gated repo, or worse, a gate run
# under a different account than the generation it is supposed to clear.
#
# Absent everywhere is FATAL, not a warning. Every backbone this script loads is gated
# (gemma-3, Llama-3.2-Vision), so without a token the job dies anyway -- minutes later, as a
# 401 buried in a traceback, after the allocation is spent. HEAD died here in the first
# second and that was the better behaviour. An empty file falls through to the next source
# rather than setting HF_TOKEN="", which HEAD would have done and which fails as an
# unreadable 401. Only the source is ever printed, never the token (AGENTS.md rule 8), and
# the Colab path prints nothing at all, so its log is byte-identical.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOK = ""
for _p in ("/content/hf_token.txt", os.path.join(_REPO, ".hf")):
    if not os.path.exists(_p):
        continue
    _TOK = open(_p).read().strip()
    if _TOK:
        if _p != "/content/hf_token.txt":
            print(f"HF token: read from {_p}", flush=True)
        break
    print(f"HF token: {_p} exists but is empty; trying the next source", flush=True)
if not _TOK and os.environ.get("HF_TOKEN", "").strip():
    _TOK = os.environ["HF_TOKEN"].strip()
    print("HF token: taken from the environment", flush=True)
if not _TOK:
    print("ABORT: no HF token. Tried /content/hf_token.txt, "
          f"{os.path.join(_REPO, '.hf')} and $HF_TOKEN, in that order.", flush=True)
    print("  The backbones this gate loads are gated repos. Write the token to "
          f"{os.path.join(_REPO, '.hf')} (AGENTS.md rule 8: it is a credential, it is "
          "gitignored, do not print it) and resubmit.", flush=True)
    raise SystemExit(1)
os.environ["HF_TOKEN"] = _TOK

import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import mvlm_arch as A
import mvlm_data as D

TAG = os.environ["TAG"]
SPEC = A.spec(TAG)
V = os.environ.get("V", f"/content/mvlm/{TAG}")
# Same root mvlm_stage1.py built the demos into, so the held-out rows here are the forty
# that follow the 300 trained on in the same stream. /content stays the default because
# that is what every measured run used; on SLURM it is the shared filesystem, and then the
# images are downloaded once for all tags instead of once per job.
DATA = os.environ.get("MVLM_DATA", "/content/mvlm/data")
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
