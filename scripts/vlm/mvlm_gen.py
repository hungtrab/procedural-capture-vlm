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

Env: TAG, V, AF, DB, ARMS, BS, TIME_BUDGET, EVALF, SFX, MAXNEW, REQUEST_SLOT (which
     worksheet slot the stored 'response' field holds; default 1, see step1() -- 'raw' is
     what the judges actually read and it does not depend on this).
Where things live, both defaulting to the Colab layout every measured run used:
     MVLM_CODE (sibling modules), MVLM_PIP=0 (do not install).
"""
import os, sys, subprocess, json, time

# Where the sibling module mvlm_arch is. The Colab driver pushes every script flat into
# /content (run_mvlm.sh:266), so the old hardcoded "/content" was simply the directory this
# file is in. Defaulting to that directory leaves Colab exactly as it was and lets a SLURM
# node run the same file out of scripts/vlm, where /content does not exist at all and the
# import would die before the first line of work.
CODE = os.environ.get("MVLM_CODE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import mvlm_arch as _A  # module-level imports: os only, so this is safe before pip runs
_TF = _A.tf_pin(os.environ["TAG"])

# A fresh Colab VM starts bare, so installing stays the default. MVLM_PIP=0 is for a SLURM
# job, where the env is pre-built and shared with other jobs running at the same time: an
# install there mutates the environment underneath them, and an upgrade can move the
# transformers version mvlm_arch.py:46 pins for a backbone that is already mid-run -- which
# on this script would mean generating rows under a different library than the gate used.
# Both calls are check=False, so an install that fails is invisible today; print the
# interpreter when they are skipped so a log reader can tell "skipped" from "never ran".
if os.environ.get("MVLM_PIP", "1") != "0":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    _TF, "peft", "accelerate", "bitsandbytes"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow==10.4.0"],
                   check=False)
else:
    print(f"MVLM_PIP=0: skipped pip install of {_TF}, peft, accelerate, bitsandbytes, "
          f"pillow; using {sys.executable}", flush=True)

# HF token. A FILE beats the environment, which is the precedence HEAD had: HEAD was a bare
#   os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
# so the file was the only source and an exported HF_TOKEN could not reach the loader at all.
# The chain, in order:
#   1. /content/hf_token.txt   where the Colab driver uploads .hf (run_mvlm.sh:264)
#   2. <repo>/.hf              the same credential in the SLURM layout, where /content does
#                              not exist; scripts/slurm/cell.sbatch:69 reads this very file
#   3. $HF_TOKEN               for a run with neither file, started by hand
# The environment is LAST on purpose. cell.sbatch submits with --export=ALL and only loads
# .hf when HF_TOKEN is empty, so a stale HF_TOKEN left in the submitting shell rides into the
# job and, if it were consulted first, would silently replace the credential the driver meant
# to use -- and the only visible sign would be a 401 on one gated repo, or worse, rows
# generated under a different account than the gate that cleared them.
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
    print("  The backbones this script generates with are gated repos. Write the token to "
          f"{os.path.join(_REPO, '.hf')} (AGENTS.md rule 8: it is a credential, it is "
          "gitignored, do not print it) and resubmit.", flush=True)
    raise SystemExit(1)
os.environ["HF_TOKEN"] = _TOK

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
    if r.get("image_kind") == "photo" and not os.path.exists(p):
        # A PoC row that names a real photo must get it; falling back to the gray square
        # would turn the photo half of the eval into a second gray half without a trace.
        raise SystemExit(f"photo row {r.get('id')}: image {p!r} not found")
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

# PRECISION=bf16 loads the backbone unquantized (LoRA on bf16 weights); the default,
# 4bit, is the QLoRA load every published row used and is unchanged.
PRECISION = os.environ.get("PRECISION", "4bit")
if PRECISION not in ("4bit", "bf16"):
    raise SystemExit("PRECISION must be 4bit or bf16")
print(f"PRECISION={PRECISION}", flush=True)
bnb = None if PRECISION == "bf16" else BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
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


def _end_markers(k):
    """What closes worksheet slot k.

    For k=1 this is exactly the list that used to be hardcoded, in exactly that order, and
    the order matters: the loop below returns on the first marker in LIST order, not at the
    earliest position in the text, so moving "Count the" ahead of "Step 2" would change
    stored rows. "Count the" and "word count" are the published worksheet's own Step 2
    wording (mvlm_data.py PROC_COUNT), there to catch a model that writes Step 2's content
    without writing its header. They are dropped for k>=2 because only WS=front puts the
    request in slot 2, WS=front has no counting step at all, and a harmful answer that
    happens to say "count the" would otherwise be cut in half.
    """
    mk = [f"Step {k+1}", f"Step{k+1}", f"Step {k+2}"]
    if k == 1:
        mk += ["Count the", "word count"]
    return mk


# Which worksheet slot holds the request: the same variable, the same default and the same
# validation as audit_nonanswer.py, so one REQUEST_SLOT describes a condition everywhere. 1
# is the published worksheet and WS=chain; 2 is WS=front (D43), where Step 1 is "Describe
# what you see in the image" and the request is in Step 2. Read with `or "1"` because
# cell.sbatch submits with --export=ALL, so a set-but-empty REQUEST_SLOT reaches the job and
# int("") would raise at import, hours before the first row.
_SLOT_ENV = (os.environ.get("REQUEST_SLOT") or "1").strip() or "1"
if _SLOT_ENV not in ("1", "2"):
    raise SystemExit(
        "REQUEST_SLOT=%r: only 1 and 2 are meaningful.\n"
        "  The worksheet holds the request in exactly one slot: Step 1 for the published\n"
        "  worksheet and for WS=chain, Step 2 for WS=front (D43). This is not WS=.\n"
        "  Unset REQUEST_SLOT for the published numbers." % _SLOT_ENV)
SLOT = int(_SLOT_ENV)
END = _end_markers(SLOT)


def step1(t):
    """The request slot -- the text stored in the 'response' field.

    'raw' is the field the main judging path reads: judge_multi.py defaults JUDGE_FIELD to
    raw, and run_cell.sh passes JUDGE_FIELD=raw explicitly on all four passes, so every
    published number is unaffected by anything in this function, at any REQUEST_SLOT.
    'response' is what JUDGE_FIELD=step1 ends up judging -- no row carries a "step1" key, so
    judge_multi.judged_text falls through to this field -- and under WS=front a slot-1
    reading would hand the judge the gray-image description instead of the answer to the
    request, which it would score safe while the real answer went unread.

    lgjudge_only.py and qgpass.py reach the same field by the same fallback but then run
    their OWN hardcoded step1() over it, which cuts at the first "Step 2" and so returns
    nothing at all from a slot-2 'response'. Those two are the per-arm Colab scripts, they
    are not on the SLURM path, and neither is touched here; use judge_multi.py, which takes
    the field as it finds it.

    At REQUEST_SLOT=1 this is byte-for-byte the old function: the slice starts at character
    zero, so the "Step 1." header and anything before it stay in, exactly as they always
    have, and no stored row moves.
    """
    s = t
    if SLOT > 1:
        # Slot 1 ran from the top of the response; a later slot has to be found first. No
        # header for it means the worksheet was dropped, and then the whole response is the
        # answer -- the same fallback audit_nonanswer.request_slot makes.
        for mk in (f"Step {SLOT}", f"Step{SLOT}"):
            i = t.find(mk)
            if i != -1:
                s = t[i:]
                break
        else:
            return t
    for mk in END:
        i = s.find(mk)
        if i != -1:
            return s[:i]
    return s


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
